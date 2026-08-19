#!/usr/bin/env python3
"""Formal Stockbridge U-RNN training with regression-only MSE.

This script trains U-RNN on the converted Stockbridge dataset using:
- train split: 70 events
- valid split: 15 events
- full forecast horizon by default: T1–T24
- regression-only MSE on normalised water depth

This is the first formal baseline training script for the Stockbridge U-RNN
adaptation. It deliberately avoids the original combined regression/classification
loss until the U-RNN wet/dry branch is reviewed for this dataset.
"""

import argparse
import csv
import json
import math
import random
import subprocess
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
        raise ValueError("At least one timestep is required.")
    for step in steps:
        if step < 1 or step >= DURATION:
            raise ValueError(f"Invalid step {step}; expected 1..24.")
    return steps


def get_git_commit():
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
        )
        return out.strip()
    except Exception:
        return "unknown"


def grad_norm(parameters):
    params = [p for p in parameters if p.grad is not None]
    if not params:
        return torch.tensor(0.0)
    total = torch.zeros((), device=params[0].device)
    for p in params:
        total = total + p.grad.detach().pow(2).sum()
    return total.sqrt()


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


def unpack_event_path(batch_event):
    if isinstance(batch_event, (list, tuple)):
        return str(batch_event[0])
    return str(batch_event)


def run_one_event(
    *,
    model,
    net_cfg,
    inputs,
    label_mm,
    steps,
    device,
    flood_max_mm,
    rain_max,
    cumsum_rain_max,
    train_mode,
):
    enc1, enc2, enc3, dec1, dec2, dec3 = initialize_states(
        device,
        PAD_H,
        PAD_W,
        net_cfg=net_cfg,
    )

    predictions = []

    for t in steps:
        x = preprocess_inputs(
            t,
            inputs,
            device=device,
            nums=HISTORICAL_NUMS,
            rain_max=rain_max,
            cumsum_rain_max=cumsum_rain_max,
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
    target_mm = label_mm[:, steps, :ORIG_H, :ORIG_W].to(device)
    target_norm = target_mm / flood_max_mm

    loss = F.mse_loss(pred, target_norm)

    if not torch.isfinite(loss):
        raise RuntimeError("Non-finite loss detected.")

    with torch.no_grad():
        pred_min = float(pred.min())
        pred_max = float(pred.max())
        target_min = float(target_norm.min())
        target_max = float(target_norm.max())

    return loss, pred_min, pred_max, target_min, target_max


def make_loader(dataset, shuffle, seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=shuffle,
        num_workers=0,
        generator=generator if shuffle else None,
    )


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
        default=Path.home() / "flood/03_outputs/Stockbridge-URNN/comet/full_mse_v01",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--steps", type=str, default=",".join(str(i) for i in range(1, 25)))
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1884)
    parser.add_argument("--flood-max-mm", type=float, default=5000.0)
    parser.add_argument("--rain-max", type=float, default=60.0)
    parser.add_argument("--cumsum-rain-max", type=float, default=250.0)
    parser.add_argument("--max-train-events", type=int, default=0, help="0 means use full train split.")
    parser.add_argument("--max-valid-events", type=int, default=0, help="0 means use full valid split.")
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument("--no-checkpoint", action="store_true")
    args = parser.parse_args()

    steps = parse_steps(args.steps)
    use_checkpoint = not args.no_checkpoint

    random.seed(args.seed)
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

    train_dataset = StockbridgeDynamic2DFlood(
        data_root=str(args.data_root.expanduser()),
        split="train",
        duration=DURATION,
    )
    valid_dataset = StockbridgeDynamic2DFlood(
        data_root=str(args.data_root.expanduser()),
        split="valid",
        duration=DURATION,
    )

    train_loader = make_loader(
        train_dataset,
        shuffle=not args.no_shuffle,
        seed=args.seed,
    )
    valid_loader = make_loader(
        valid_dataset,
        shuffle=False,
        seed=args.seed,
    )

    model, net_cfg, input_channels = build_model(
        device=device,
        use_checkpoint=use_checkpoint,
        cls_thred=0.0,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    model_params = sum(p.numel() for p in model.parameters())

    config = {
        "script": "train_stockbridge_urnn_mse.py",
        "git_commit": get_git_commit(),
        "data_root": str(args.data_root.expanduser()),
        "output_dir": str(args.output_dir),
        "epochs": args.epochs,
        "steps": steps,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "flood_max_mm": args.flood_max_mm,
        "rain_max": args.rain_max,
        "cumsum_rain_max": args.cumsum_rain_max,
        "train_events_available": len(train_dataset),
        "valid_events_available": len(valid_dataset),
        "max_train_events": args.max_train_events,
        "max_valid_events": args.max_valid_events,
        "shuffle_train": not args.no_shuffle,
        "orig_shape": [ORIG_H, ORIG_W],
        "padded_shape": [PAD_H, PAD_W],
        "duration": DURATION,
        "historical_nums": HISTORICAL_NUMS,
        "input_channels": input_channels,
        "model_parameters": model_params,
        "use_checkpoint": use_checkpoint,
        "loss": "MSE on normalised flood depth",
        "note": "Formal baseline MSE training; not wet-weighted and not final evaluation.",
    }

    with open(args.output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("Output dir:", args.output_dir)
    print("Git commit:", config["git_commit"])
    print("Train events available:", len(train_dataset))
    print("Valid events available:", len(valid_dataset))
    print("Max train events:", args.max_train_events or "full")
    print("Max valid events:", args.max_valid_events or "full")
    print("Model parameters:", model_params)
    print("Input channels:", input_channels)
    print("Steps:", steps)
    print("Epochs:", args.epochs)
    print("Learning rate:", args.lr)
    print("Weight decay:", args.weight_decay)
    print("Use checkpoint:", use_checkpoint)

    metrics_path = args.output_dir / "metrics.csv"
    best_valid_loss = math.inf
    best_epoch = None

    with open(metrics_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "train_loss",
                "valid_loss",
                "train_events",
                "valid_events",
                "mean_grad_norm",
                "epoch_time_s",
                "train_pred_min",
                "train_pred_max",
                "valid_pred_min",
                "valid_pred_max",
                "target_min",
                "target_max",
                "gpu_peak_allocated_mb",
                "gpu_peak_reserved_mb",
                "best_valid_loss",
                "best_epoch",
            ],
        )
        writer.writeheader()

        for epoch in range(1, args.epochs + 1):
            epoch_start = time.perf_counter()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)

            model.train()

            train_loss_sum = 0.0
            train_count = 0
            grad_norm_sum = 0.0
            train_pred_min = math.inf
            train_pred_max = -math.inf
            target_min_seen = math.inf
            target_max_seen = -math.inf

            for batch_idx, batch in enumerate(train_loader, start=1):
                if args.max_train_events and train_count >= args.max_train_events:
                    break

                inputs = batch[0]
                label_mm = batch[1]

                optimizer.zero_grad(set_to_none=True)

                loss, pred_min, pred_max, target_min, target_max = run_one_event(
                    model=model,
                    net_cfg=net_cfg,
                    inputs=inputs,
                    label_mm=label_mm,
                    steps=steps,
                    device=device,
                    flood_max_mm=args.flood_max_mm,
                    rain_max=args.rain_max,
                    cumsum_rain_max=args.cumsum_rain_max,
                    train_mode=True,
                )

                loss.backward()
                gnorm = grad_norm(model.parameters())
                optimizer.step()

                train_loss_sum += loss.item()
                grad_norm_sum += gnorm.item()
                train_count += 1

                train_pred_min = min(train_pred_min, pred_min)
                train_pred_max = max(train_pred_max, pred_max)
                target_min_seen = min(target_min_seen, target_min)
                target_max_seen = max(target_max_seen, target_max)

            train_loss = train_loss_sum / max(train_count, 1)
            mean_grad = grad_norm_sum / max(train_count, 1)

            model.eval()

            valid_loss_sum = 0.0
            valid_count = 0
            valid_pred_min = math.inf
            valid_pred_max = -math.inf

            with torch.no_grad():
                for batch_idx, batch in enumerate(valid_loader, start=1):
                    if args.max_valid_events and valid_count >= args.max_valid_events:
                        break

                    inputs = batch[0]
                    label_mm = batch[1]

                    loss, pred_min, pred_max, target_min, target_max = run_one_event(
                        model=model,
                        net_cfg=net_cfg,
                        inputs=inputs,
                        label_mm=label_mm,
                        steps=steps,
                        device=device,
                        flood_max_mm=args.flood_max_mm,
                        rain_max=args.rain_max,
                        cumsum_rain_max=args.cumsum_rain_max,
                        train_mode=False,
                    )

                    valid_loss_sum += loss.item()
                    valid_count += 1

                    valid_pred_min = min(valid_pred_min, pred_min)
                    valid_pred_max = max(valid_pred_max, pred_max)
                    target_min_seen = min(target_min_seen, target_min)
                    target_max_seen = max(target_max_seen, target_max)

            valid_loss = valid_loss_sum / max(valid_count, 1)
            epoch_time = time.perf_counter() - epoch_start

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                best_epoch = epoch
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "train_loss": train_loss,
                        "valid_loss": valid_loss,
                        "config": config,
                    },
                    args.output_dir / "checkpoint_best.pt",
                )

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "valid_loss": valid_loss,
                    "config": config,
                },
                args.output_dir / "checkpoint_last.pt",
            )

            gpu_peak_alloc = 0.0
            gpu_peak_reserved = 0.0
            if device.type == "cuda":
                gpu_peak_alloc = torch.cuda.max_memory_allocated(device) / 1024**2
                gpu_peak_reserved = torch.cuda.max_memory_reserved(device) / 1024**2

            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "valid_loss": valid_loss,
                "train_events": train_count,
                "valid_events": valid_count,
                "mean_grad_norm": mean_grad,
                "epoch_time_s": epoch_time,
                "train_pred_min": train_pred_min,
                "train_pred_max": train_pred_max,
                "valid_pred_min": valid_pred_min,
                "valid_pred_max": valid_pred_max,
                "target_min": target_min_seen,
                "target_max": target_max_seen,
                "gpu_peak_allocated_mb": gpu_peak_alloc,
                "gpu_peak_reserved_mb": gpu_peak_reserved,
                "best_valid_loss": best_valid_loss,
                "best_epoch": best_epoch,
            }
            writer.writerow(row)
            f.flush()

            print(
                f"epoch={epoch:04d} "
                f"train_loss={train_loss:.8f} "
                f"valid_loss={valid_loss:.8f} "
                f"best_valid={best_valid_loss:.8f} "
                f"best_epoch={best_epoch} "
                f"train_events={train_count} "
                f"valid_events={valid_count} "
                f"grad_norm={mean_grad:.6f} "
                f"time={epoch_time:.2f}s "
                f"train_pred=[{train_pred_min:.5f},{train_pred_max:.5f}] "
                f"valid_pred=[{valid_pred_min:.5f},{valid_pred_max:.5f}] "
                f"target=[{target_min_seen:.5f},{target_max_seen:.5f}] "
                f"gpu_peak_mb={gpu_peak_alloc:.1f}"
            )

    print()
    print("Formal Stockbridge U-RNN MSE training complete")
    print(f"best_valid_loss={best_valid_loss:.8f}")
    print(f"best_epoch={best_epoch}")
    print("metrics:", metrics_path)
    print("checkpoint_last:", args.output_dir / "checkpoint_last.pt")
    print("checkpoint_best:", args.output_dir / "checkpoint_best.pt")

    if device.type == "cuda":
        print(f"gpu_peak_allocated_mb={torch.cuda.max_memory_allocated(device) / 1024**2:.3f}")
        print(f"gpu_peak_reserved_mb={torch.cuda.max_memory_reserved(device) / 1024**2:.3f}")


if __name__ == "__main__":
    main()
