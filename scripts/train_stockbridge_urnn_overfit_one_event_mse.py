#!/usr/bin/env python3
"""Overfit one Stockbridge event with U-RNN using regression-only MSE.

Purpose:
- Verify that U-RNN can learn from one converted Stockbridge event.
- Train on one event for several epochs.
- Track whether MSE decreases.
- Save metrics and checkpoints.

This is NOT the final full training experiment.
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
U_RNN_CODE = Path.home() / "flood/01_repositories/U-RNN/code"

if not U_RNN_CODE.exists():
    raise FileNotFoundError(f"Expected U-RNN code folder not found: {U_RNN_CODE}")

for path in [REPO_ROOT, U_RNN_CODE]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from stockbridge_urnn.dataset import StockbridgeDynamic2DFlood
from src.lib.dataset.Dynamic2DFlood import preprocess_inputs
from src.lib.model.networks.model import ED
from src.lib.model.networks.net_params import get_network_params
from src.lib.utils.general import initialize_states
from src.lib.utils.net_config import get_input_channels, load_net_config


ORIG_H, ORIG_W = 305, 326
PAD_H, PAD_W = 308, 328
DURATION = 25
HISTORICAL_NUMS = 3


def parse_steps(text):
    steps = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not steps:
        raise ValueError("At least one training step is required.")
    for step in steps:
        if step < 1 or step >= DURATION:
            raise ValueError(f"Invalid step {step}; expected 1..24.")
    return steps


def grad_norm(parameters):
    params = [p for p in parameters if p.grad is not None]
    if not params:
        return torch.tensor(0.0)
    total = torch.zeros((), device=params[0].device)
    for p in params:
        total = total + p.grad.detach().pow(2).sum()
    return total.sqrt()


def get_one_batch(dataset, event_index):
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    for idx, batch in enumerate(loader):
        if idx == event_index:
            return batch
    raise IndexError(f"event_index={event_index} out of range for dataset length {len(dataset)}")


def build_model(device, use_checkpoint=True, cls_thred=0.0):
    net_cfg = load_net_config()
    input_channels = get_input_channels(net_cfg, HISTORICAL_NUMS)

    params = get_network_params(
        use_checkpoint,
        PAD_H,
        PAD_W,
        input_channels=input_channels,
        net_cfg=net_cfg,
    )

    model = ED(
        clstm_flag=False,
        encoder_params=params[0],
        decoder_params=params[1],
        cls_thred=cls_thred,
        use_checkpoint=use_checkpoint,
        input_height=PAD_H,
        input_width=PAD_W,
    ).to(device)

    return model, net_cfg, input_channels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path.home() / "flood/02_data/CityCAT_Winchester/urnn/stockbridge_urnn_full_v01",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.home() / "flood/03_outputs/Stockbridge-URNN/comet/overfit_one_event_mse_v01",
    )
    parser.add_argument("--event-index", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--steps", type=str, default="1,2,3,4,5")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=1884)
    parser.add_argument("--flood-max-mm", type=float, default=5000.0)
    parser.add_argument("--rain-max", type=float, default=60.0)
    parser.add_argument("--cumsum-rain-max", type=float, default=250.0)
    parser.add_argument("--no-checkpoint", action="store_true")
    args = parser.parse_args()

    train_steps = parse_steps(args.steps)
    use_checkpoint = not args.no_checkpoint

    torch.manual_seed(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(device))
        torch.cuda.reset_peak_memory_stats(device)
    else:
        print("WARNING: CUDA is not available. This should normally run in a GPU Slurm job.")

    args.output_dir = args.output_dir.expanduser()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = StockbridgeDynamic2DFlood(
        data_root=str(args.data_root.expanduser()),
        split="train",
        duration=DURATION,
    )

    batch = get_one_batch(dataset, args.event_index)
    inputs = batch[0]
    label_mm = batch[1]
    event_dir = batch[2][0] if isinstance(batch[2], (list, tuple)) else batch[2]

    print("Dataset length:", len(dataset))
    print("Event index:", args.event_index)
    print("Event:", event_dir)
    print("Label mm shape:", tuple(label_mm.shape))
    print("Label mm min/max:", float(label_mm.min()), float(label_mm.max()))

    model, net_cfg, input_channels = build_model(
        device=device,
        use_checkpoint=use_checkpoint,
        cls_thred=0.0,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print("Model parameters:", sum(p.numel() for p in model.parameters()))
    print("Input channels:", input_channels)
    print("Train steps:", train_steps)
    print("Epochs:", args.epochs)
    print("Learning rate:", args.lr)
    print("Use checkpoint:", use_checkpoint)
    print("Output dir:", args.output_dir)

    config = {
        "data_root": str(args.data_root.expanduser()),
        "output_dir": str(args.output_dir),
        "event_index": args.event_index,
        "event": str(event_dir),
        "epochs": args.epochs,
        "steps": train_steps,
        "lr": args.lr,
        "seed": args.seed,
        "flood_max_mm": args.flood_max_mm,
        "rain_max": args.rain_max,
        "cumsum_rain_max": args.cumsum_rain_max,
        "orig_shape": [ORIG_H, ORIG_W],
        "padded_shape": [PAD_H, PAD_W],
        "duration": DURATION,
        "historical_nums": HISTORICAL_NUMS,
        "input_channels": input_channels,
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "use_checkpoint": use_checkpoint,
        "loss": "MSE on normalised depth",
        "note": "One-event overfit debug run, not final training.",
    }

    with open(args.output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    metrics_path = args.output_dir / "metrics.csv"
    best_loss = float("inf")

    with open(metrics_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "loss",
                "grad_norm",
                "epoch_time_s",
                "pred_min",
                "pred_max",
                "target_min",
                "target_max",
                "gpu_allocated_mb",
                "gpu_reserved_mb",
                "gpu_peak_allocated_mb",
            ],
        )
        writer.writeheader()

        for epoch in range(1, args.epochs + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)

            enc1, enc2, enc3, dec1, dec2, dec3 = initialize_states(
                device,
                PAD_H,
                PAD_W,
                net_cfg=net_cfg,
            )

            predictions = []
            start = time.perf_counter()

            for t in train_steps:
                x = preprocess_inputs(
                    t,
                    inputs,
                    device=device,
                    nums=HISTORICAL_NUMS,
                    rain_max=args.rain_max,
                    cumsum_rain_max=args.cumsum_rain_max,
                )

                out, enc1, enc2, enc3, dec1, dec2, dec3 = model(
                    x,
                    enc1,
                    enc2,
                    enc3,
                    dec1,
                    dec2,
                    dec3,
                )

                out_crop = out[..., :ORIG_H, :ORIG_W]
                predictions.append(out_crop)

            pred = torch.cat(predictions, dim=1)
            target_mm = label_mm[:, train_steps, :ORIG_H, :ORIG_W].to(device)
            target_norm = target_mm / args.flood_max_mm

            loss = F.mse_loss(pred, target_norm)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch}: {loss}")

            loss.backward()
            gnorm = grad_norm(model.parameters())
            optimizer.step()

            epoch_time = time.perf_counter() - start

            gpu_alloc = 0.0
            gpu_reserved = 0.0
            gpu_peak = 0.0
            if device.type == "cuda":
                gpu_alloc = torch.cuda.memory_allocated(device) / 1024**2
                gpu_reserved = torch.cuda.memory_reserved(device) / 1024**2
                gpu_peak = torch.cuda.max_memory_allocated(device) / 1024**2

            row = {
                "epoch": epoch,
                "loss": loss.item(),
                "grad_norm": gnorm.item(),
                "epoch_time_s": epoch_time,
                "pred_min": float(pred.min()),
                "pred_max": float(pred.max()),
                "target_min": float(target_norm.min()),
                "target_max": float(target_norm.max()),
                "gpu_allocated_mb": gpu_alloc,
                "gpu_reserved_mb": gpu_reserved,
                "gpu_peak_allocated_mb": gpu_peak,
            }
            writer.writerow(row)
            f.flush()

            print(
                f"epoch={epoch:04d} "
                f"loss={loss.item():.8f} "
                f"grad_norm={gnorm.item():.6f} "
                f"time={epoch_time:.3f}s "
                f"pred=[{float(pred.min()):.6f},{float(pred.max()):.6f}] "
                f"target=[{float(target_norm.min()):.6f},{float(target_norm.max()):.6f}]"
            )

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss.item(),
                    "config": config,
                },
                args.output_dir / "checkpoint_last.pt",
            )

            if loss.item() < best_loss:
                best_loss = loss.item()
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": loss.item(),
                        "config": config,
                    },
                    args.output_dir / "checkpoint_best.pt",
                )

    print()
    print("Overfit one-event training complete")
    print(f"best_loss={best_loss:.8f}")
    print("metrics:", metrics_path)
    print("checkpoint_last:", args.output_dir / "checkpoint_last.pt")
    print("checkpoint_best:", args.output_dir / "checkpoint_best.pt")

    if device.type == "cuda":
        print(f"gpu_peak_allocated_mb={torch.cuda.max_memory_allocated(device) / 1024**2:.3f}")
        print(f"gpu_peak_reserved_mb={torch.cuda.max_memory_reserved(device) / 1024**2:.3f}")


if __name__ == "__main__":
    main()
