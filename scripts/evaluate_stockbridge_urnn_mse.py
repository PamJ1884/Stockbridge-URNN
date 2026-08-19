#!/usr/bin/env python3
"""Evaluate trained Stockbridge U-RNN MSE checkpoint on test split.

Outputs:
- aggregate_metrics.json
- per_event_metrics.csv
- selected max-depth maps as .npz files for later plotting

Metrics are computed in metres.
Predictions are evaluated both raw and clamped at >= 0.
CSI is computed using clamped predictions.
"""

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
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


def parse_float_list(text):
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_int_list(text):
    if not text.strip():
        return []
    return [int(x.strip()) for x in text.split(",") if x.strip()]


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


def run_inference_event(
    *,
    model,
    net_cfg,
    inputs,
    steps,
    device,
    flood_max_mm,
    rain_max,
    cumsum_rain_max,
):
    enc1, enc2, enc3, dec1, dec2, dec3 = initialize_states(
        device,
        PAD_H,
        PAD_W,
        net_cfg=net_cfg,
    )

    predictions = []

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()

    with torch.no_grad():
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

        pred_norm = torch.cat(predictions, dim=1)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_s = time.perf_counter() - start

    pred_mm = pred_norm * flood_max_mm
    pred_m = pred_mm / 1000.0

    return pred_m, elapsed_s


def metric_dict(pred_m, target_m, thresholds_m):
    pred_raw = pred_m
    pred_clamped = torch.clamp(pred_m, min=0.0)

    target = target_m

    out = {}

    for name, pred in [("raw", pred_raw), ("clamped", pred_clamped)]:
        err = pred - target
        mse = torch.mean(err ** 2).item()
        mae = torch.mean(torch.abs(err)).item()

        target_mean = torch.mean(target)
        denom = torch.sum((target - target_mean) ** 2).item()
        sse = torch.sum(err ** 2).item()
        nse = float("nan") if denom <= 0 else 1.0 - sse / denom

        pred_flat = pred.reshape(-1)
        target_flat = target.reshape(-1)

        pred_centered = pred_flat - torch.mean(pred_flat)
        target_centered = target_flat - torch.mean(target_flat)

        corr_denom = torch.sqrt(
            torch.sum(pred_centered ** 2) * torch.sum(target_centered ** 2)
        ).item()
        corr_num = torch.sum(pred_centered * target_centered).item()
        corr = float("nan") if corr_denom <= 0 else corr_num / corr_denom

        out[f"{name}_rmse_m"] = math.sqrt(mse)
        out[f"{name}_mae_m"] = mae
        out[f"{name}_nse"] = nse
        out[f"{name}_corr"] = corr
        out[f"{name}_pred_min_m"] = torch.min(pred).item()
        out[f"{name}_pred_max_m"] = torch.max(pred).item()

    out["target_min_m"] = torch.min(target).item()
    out["target_max_m"] = torch.max(target).item()

    for threshold in thresholds_m:
        pred_wet = pred_clamped >= threshold
        target_wet = target >= threshold

        tp = torch.logical_and(pred_wet, target_wet).sum().item()
        fp = torch.logical_and(pred_wet, ~target_wet).sum().item()
        fn = torch.logical_and(~pred_wet, target_wet).sum().item()

        denom = tp + fp + fn
        csi = float("nan") if denom == 0 else tp / denom

        key = str(threshold).replace(".", "p")
        out[f"csi_{key}m"] = csi
        out[f"tp_{key}m"] = int(tp)
        out[f"fp_{key}m"] = int(fp)
        out[f"fn_{key}m"] = int(fn)

    return out


class Aggregate:
    def __init__(self, thresholds_m):
        self.thresholds_m = thresholds_m
        self.n = 0

        self.raw_sse = 0.0
        self.raw_sae = 0.0
        self.clamped_sse = 0.0
        self.clamped_sae = 0.0

        self.target_sum = 0.0
        self.target_sq_sum = 0.0

        self.raw_pred_sum = 0.0
        self.raw_pred_sq_sum = 0.0
        self.raw_pred_target_sum = 0.0

        self.clamped_pred_sum = 0.0
        self.clamped_pred_sq_sum = 0.0
        self.clamped_pred_target_sum = 0.0

        self.target_min = math.inf
        self.target_max = -math.inf
        self.raw_pred_min = math.inf
        self.raw_pred_max = -math.inf
        self.clamped_pred_min = math.inf
        self.clamped_pred_max = -math.inf

        self.csi_counts = {
            th: {"tp": 0, "fp": 0, "fn": 0}
            for th in thresholds_m
        }

        self.inference_s = 0.0
        self.events = 0

    def update(self, pred_m, target_m, inference_s):
        pred_raw = pred_m
        pred_clamped = torch.clamp(pred_m, min=0.0)
        target = target_m

        pred_raw_f = pred_raw.reshape(-1).double()
        pred_clamped_f = pred_clamped.reshape(-1).double()
        target_f = target.reshape(-1).double()

        n = target_f.numel()
        self.n += n
        self.events += 1
        self.inference_s += inference_s

        raw_err = pred_raw_f - target_f
        clamped_err = pred_clamped_f - target_f

        self.raw_sse += torch.sum(raw_err ** 2).item()
        self.raw_sae += torch.sum(torch.abs(raw_err)).item()
        self.clamped_sse += torch.sum(clamped_err ** 2).item()
        self.clamped_sae += torch.sum(torch.abs(clamped_err)).item()

        self.target_sum += torch.sum(target_f).item()
        self.target_sq_sum += torch.sum(target_f ** 2).item()

        self.raw_pred_sum += torch.sum(pred_raw_f).item()
        self.raw_pred_sq_sum += torch.sum(pred_raw_f ** 2).item()
        self.raw_pred_target_sum += torch.sum(pred_raw_f * target_f).item()

        self.clamped_pred_sum += torch.sum(pred_clamped_f).item()
        self.clamped_pred_sq_sum += torch.sum(pred_clamped_f ** 2).item()
        self.clamped_pred_target_sum += torch.sum(pred_clamped_f * target_f).item()

        self.target_min = min(self.target_min, torch.min(target_f).item())
        self.target_max = max(self.target_max, torch.max(target_f).item())
        self.raw_pred_min = min(self.raw_pred_min, torch.min(pred_raw_f).item())
        self.raw_pred_max = max(self.raw_pred_max, torch.max(pred_raw_f).item())
        self.clamped_pred_min = min(self.clamped_pred_min, torch.min(pred_clamped_f).item())
        self.clamped_pred_max = max(self.clamped_pred_max, torch.max(pred_clamped_f).item())

        for th in self.thresholds_m:
            pred_wet = pred_clamped >= th
            target_wet = target >= th

            self.csi_counts[th]["tp"] += torch.logical_and(pred_wet, target_wet).sum().item()
            self.csi_counts[th]["fp"] += torch.logical_and(pred_wet, ~target_wet).sum().item()
            self.csi_counts[th]["fn"] += torch.logical_and(~pred_wet, target_wet).sum().item()

    def finish(self):
        target_mean = self.target_sum / self.n
        target_denom = self.target_sq_sum - (self.target_sum ** 2) / self.n

        def corr(pred_sum, pred_sq_sum, pred_target_sum):
            pred_denom = pred_sq_sum - (pred_sum ** 2) / self.n
            numerator = pred_target_sum - (pred_sum * self.target_sum) / self.n
            denominator = math.sqrt(max(pred_denom, 0.0) * max(target_denom, 0.0))
            if denominator <= 0:
                return float("nan")
            return numerator / denominator

        out = {
            "events": self.events,
            "n_values": self.n,
            "inference_total_s": self.inference_s,
            "inference_s_per_event": self.inference_s / max(self.events, 1),
            "inference_s_per_timestep": self.inference_s / max(self.events * 24, 1),
            "raw_rmse_m": math.sqrt(self.raw_sse / self.n),
            "raw_mae_m": self.raw_sae / self.n,
            "raw_nse": float("nan") if target_denom <= 0 else 1.0 - self.raw_sse / target_denom,
            "raw_corr": corr(self.raw_pred_sum, self.raw_pred_sq_sum, self.raw_pred_target_sum),
            "raw_pred_min_m": self.raw_pred_min,
            "raw_pred_max_m": self.raw_pred_max,
            "clamped_rmse_m": math.sqrt(self.clamped_sse / self.n),
            "clamped_mae_m": self.clamped_sae / self.n,
            "clamped_nse": float("nan") if target_denom <= 0 else 1.0 - self.clamped_sse / target_denom,
            "clamped_corr": corr(
                self.clamped_pred_sum,
                self.clamped_pred_sq_sum,
                self.clamped_pred_target_sum,
            ),
            "clamped_pred_min_m": self.clamped_pred_min,
            "clamped_pred_max_m": self.clamped_pred_max,
            "target_min_m": self.target_min,
            "target_max_m": self.target_max,
        }

        for th, counts in self.csi_counts.items():
            denom = counts["tp"] + counts["fp"] + counts["fn"]
            csi = float("nan") if denom == 0 else counts["tp"] / denom
            key = str(th).replace(".", "p")
            out[f"csi_{key}m"] = csi
            out[f"tp_{key}m"] = counts["tp"]
            out[f"fp_{key}m"] = counts["fp"]
            out[f"fn_{key}m"] = counts["fn"]

        return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path.home() / "flood/02_data/CityCAT_Winchester/urnn/stockbridge_urnn_full_v01",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path.home() / "flood/03_outputs/Stockbridge-URNN/comet/full_mse_v01/checkpoint_best.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.home() / "flood/03_outputs/Stockbridge-URNN/comet/full_mse_v01/test_eval",
    )
    parser.add_argument("--split", type=str, default="test", choices=["train", "valid", "test"])
    parser.add_argument("--steps", type=str, default=",".join(str(i) for i in range(1, 25)))
    parser.add_argument("--flood-max-mm", type=float, default=5000.0)
    parser.add_argument("--rain-max", type=float, default=60.0)
    parser.add_argument("--cumsum-rain-max", type=float, default=250.0)
    parser.add_argument("--thresholds-m", type=str, default="0.05,0.10,0.30")
    parser.add_argument("--sample-event-indices", type=str, default="0,7,14")
    parser.add_argument("--no-checkpoint", action="store_true")
    args = parser.parse_args()

    steps = parse_steps(args.steps)
    thresholds_m = parse_float_list(args.thresholds_m)
    sample_indices = set(parse_int_list(args.sample_event_indices))
    use_checkpoint = not args.no_checkpoint

    args.output_dir = args.output_dir.expanduser()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample_dir = args.output_dir / "sample_maps_npz"
    sample_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(device))
        torch.cuda.reset_peak_memory_stats(device)

    dataset = StockbridgeDynamic2DFlood(
        data_root=str(args.data_root.expanduser()),
        split=args.split,
        duration=DURATION,
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    model, net_cfg, input_channels = build_model(
        device=device,
        use_checkpoint=use_checkpoint,
        cls_thred=0.0,
    )

    checkpoint = torch.load(args.checkpoint.expanduser(), map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Checkpoint:", args.checkpoint.expanduser())
    print("Checkpoint epoch:", checkpoint.get("epoch", "unknown"))
    print("Checkpoint valid_loss:", checkpoint.get("valid_loss", "unknown"))
    print("Split:", args.split)
    print("Events:", len(dataset))
    print("Steps:", steps)
    print("Thresholds m:", thresholds_m)
    print("Input channels:", input_channels)
    print("Output dir:", args.output_dir)

    config = {
        "script": "evaluate_stockbridge_urnn_mse.py",
        "data_root": str(args.data_root.expanduser()),
        "checkpoint": str(args.checkpoint.expanduser()),
        "checkpoint_epoch": checkpoint.get("epoch", "unknown"),
        "checkpoint_train_loss": checkpoint.get("train_loss", "unknown"),
        "checkpoint_valid_loss": checkpoint.get("valid_loss", "unknown"),
        "output_dir": str(args.output_dir),
        "split": args.split,
        "events": len(dataset),
        "steps": steps,
        "thresholds_m": thresholds_m,
        "sample_event_indices": sorted(sample_indices),
        "flood_max_mm": args.flood_max_mm,
        "rain_max": args.rain_max,
        "cumsum_rain_max": args.cumsum_rain_max,
        "orig_shape": [ORIG_H, ORIG_W],
        "padded_shape": [PAD_H, PAD_W],
        "duration": DURATION,
        "historical_nums": HISTORICAL_NUMS,
        "input_channels": input_channels,
        "use_checkpoint": use_checkpoint,
        "metrics_note": "Metrics in metres. CSI uses clamped predictions.",
    }

    with open(args.output_dir / "eval_config.json", "w") as f:
        json.dump(config, f, indent=2)

    per_event_csv = args.output_dir / "per_event_metrics.csv"
    aggregate = Aggregate(thresholds_m=thresholds_m)

    with open(per_event_csv, "w", newline="") as f:
        fieldnames = [
            "event_index",
            "event_name",
            "inference_s",
            "raw_rmse_m",
            "raw_mae_m",
            "raw_nse",
            "raw_corr",
            "clamped_rmse_m",
            "clamped_mae_m",
            "clamped_nse",
            "clamped_corr",
            "raw_pred_min_m",
            "raw_pred_max_m",
            "clamped_pred_min_m",
            "clamped_pred_max_m",
            "target_min_m",
            "target_max_m",
        ]

        for th in thresholds_m:
            key = str(th).replace(".", "p")
            fieldnames += [f"csi_{key}m", f"tp_{key}m", f"fp_{key}m", f"fn_{key}m"]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for event_index, batch in enumerate(loader):
            inputs = batch[0]
            label_mm = batch[1]
            event_path = unpack_event_path(batch[2])
            event_name = os.path.basename(event_path)

            pred_m, inference_s = run_inference_event(
                model=model,
                net_cfg=net_cfg,
                inputs=inputs,
                steps=steps,
                device=device,
                flood_max_mm=args.flood_max_mm,
                rain_max=args.rain_max,
                cumsum_rain_max=args.cumsum_rain_max,
            )

            target_mm = label_mm[:, steps, :ORIG_H, :ORIG_W].to(device)
            target_m = target_mm / 1000.0

            event_metrics = metric_dict(pred_m, target_m, thresholds_m)
            aggregate.update(pred_m, target_m, inference_s)

            row = {
                "event_index": event_index,
                "event_name": event_name,
                "inference_s": inference_s,
            }
            row.update(event_metrics)
            writer.writerow(row)
            f.flush()

            print(
                f"event={event_index:03d} {event_name} "
                f"rmse={event_metrics['clamped_rmse_m']:.5f}m "
                f"mae={event_metrics['clamped_mae_m']:.5f}m "
                f"nse={event_metrics['clamped_nse']:.4f} "
                f"corr={event_metrics['clamped_corr']:.4f} "
                f"time={inference_s:.3f}s "
                f"pred_max={event_metrics['clamped_pred_max_m']:.3f}m "
                f"target_max={event_metrics['target_max_m']:.3f}m"
            )

            if event_index in sample_indices:
                pred_np = torch.clamp(pred_m, min=0.0).squeeze(0).detach().cpu().numpy()
                target_np = target_m.squeeze(0).detach().cpu().numpy()
                diff_np = pred_np - target_np

                np.savez_compressed(
                    sample_dir / f"{event_index:03d}_{event_name}_maps.npz",
                    pred_m=pred_np.astype(np.float32),
                    target_m=target_np.astype(np.float32),
                    diff_m=diff_np.astype(np.float32),
                    steps=np.array(steps, dtype=np.int32),
                    event_name=np.array(event_name),
                )

    aggregate_metrics = aggregate.finish()

    if device.type == "cuda":
        aggregate_metrics["gpu_peak_allocated_mb"] = torch.cuda.max_memory_allocated(device) / 1024**2
        aggregate_metrics["gpu_peak_reserved_mb"] = torch.cuda.max_memory_reserved(device) / 1024**2

    with open(args.output_dir / "aggregate_metrics.json", "w") as f:
        json.dump(aggregate_metrics, f, indent=2)

    print()
    print("Evaluation complete")
    print("Aggregate metrics:")
    print(json.dumps(aggregate_metrics, indent=2))
    print("per_event_metrics:", per_event_csv)
    print("aggregate_metrics:", args.output_dir / "aggregate_metrics.json")
    print("sample maps:", sample_dir)


if __name__ == "__main__":
    main()
