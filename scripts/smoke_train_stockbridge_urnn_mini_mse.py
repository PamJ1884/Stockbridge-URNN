#!/usr/bin/env python3
"""Tiny U-RNN trainability smoke test on converted Stockbridge mini dataset.

Purpose:
- Use real converted Stockbridge data.
- Run a short recurrent sequence through U-RNN.
- Compute simple MSE on normalised depth.
- Check loss, gradients, and optimizer.step().

This is NOT a performance experiment.
This deliberately avoids the original FocalBCE_and_WMSE path until the
reg/cls output handling is reviewed.
"""

import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

U_RNN_CODE = Path.home() / "flood/01_repositories/U-RNN/code"
if not U_RNN_CODE.exists():
    raise FileNotFoundError(f"Expected U-RNN code folder not found: {U_RNN_CODE}")
if str(U_RNN_CODE) not in sys.path:
    sys.path.insert(0, str(U_RNN_CODE))

from src.lib.dataset.Dynamic2DFlood import Dynamic2DFlood, preprocess_inputs
from src.lib.model.networks.model import ED
from src.lib.utils.net_config import load_net_config, get_input_channels
from src.lib.model.networks.net_params import get_network_params
from src.lib.utils.general import initialize_states


DATA_ROOT = Path.home() / "flood/02_data/CityCAT_Winchester/urnn/stockbridge_urnn_mini_v01"
LOCATION = "stockbridge"
EVENT_LIST = DATA_ROOT / "stockbridge_train.txt"

ORIG_H, ORIG_W = 305, 326
PAD_H, PAD_W = 308, 328
DURATION = 25
HISTORICAL_NUMS = 3
FLOOD_MAX_MM = 5000.0

# Short window only. Enough to test gradients without stressing memory.
TRAIN_STEPS = [1, 2, 3, 4, 5]


def grad_norm(parameters):
    params = [p for p in parameters if p.grad is not None]
    if not params:
        return torch.tensor(0.0)
    total = torch.zeros((), device=params[0].device)
    for p in params:
        total = total + p.grad.detach().pow(2).sum()
    return total.sqrt()


def parameter_changed(before, after):
    return any(not torch.equal(b, a) for b, a in zip(before, after))


def main():
    torch.manual_seed(42)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(device))
        torch.cuda.reset_peak_memory_stats(device)

    dataset = Dynamic2DFlood(
        data_root=str(DATA_ROOT),
        split="train",
        event_list_file=str(EVENT_LIST),
        duration=DURATION,
        location=LOCATION,
    )

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    batch = next(iter(loader))

    inputs = batch[0]
    label_mm = batch[1]  # (B,T,308,328), already converted to mm by loader
    event_dir = batch[2]

    print("Event:", event_dir)
    print("Label mm shape:", tuple(label_mm.shape))
    print("Label mm min/max:", float(label_mm.min()), float(label_mm.max()))

    net_cfg = load_net_config()
    input_channels = get_input_channels(net_cfg, HISTORICAL_NUMS)

    params = get_network_params(
        False,
        PAD_H,
        PAD_W,
        input_channels=input_channels,
        net_cfg=net_cfg,
    )

    # cls_thred=0.0 avoids zeroing most regression outputs during this debug smoke test.
    # This is not the final scientific training setting.
    model = ED(
        clstm_flag=False,
        encoder_params=params[0],
        decoder_params=params[1],
        cls_thred=0.0,
        use_checkpoint=False,
        input_height=PAD_H,
        input_width=PAD_W,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    print("Model parameters:", sum(p.numel() for p in model.parameters()))
    print("Input channels:", input_channels)
    print("Train steps:", TRAIN_STEPS)

    model.train()
    optimizer.zero_grad(set_to_none=True)

    before_params = [p.detach().clone() for p in model.parameters()]

    enc1, enc2, enc3, dec1, dec2, dec3 = initialize_states(
        device,
        PAD_H,
        PAD_W,
        net_cfg=net_cfg,
    )

    predictions = []

    start = time.perf_counter()

    for t in TRAIN_STEPS:
        x = preprocess_inputs(
            t,
            inputs,
            device=device,
            nums=HISTORICAL_NUMS,
            rain_max=60.0,
            cumsum_rain_max=250.0,
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

        # out: (B,1,308,328), normalised depth-like prediction
        out_crop = out[..., :ORIG_H, :ORIG_W]
        predictions.append(out_crop)

    pred = torch.cat(predictions, dim=1)  # (B,S,305,326)

    target_mm = label_mm[:, TRAIN_STEPS, :ORIG_H, :ORIG_W].to(device)
    target_norm = target_mm / FLOOD_MAX_MM

    print("Prediction shape:", tuple(pred.shape))
    print("Target shape:", tuple(target_norm.shape))
    print("Prediction range:", float(pred.min()), float(pred.max()))
    print("Target norm range:", float(target_norm.min()), float(target_norm.max()))

    loss = F.mse_loss(pred, target_norm)

    if not torch.isfinite(loss):
        raise RuntimeError("Loss is not finite")

    loss.backward()
    optimizer.step()

    elapsed = time.perf_counter() - start

    grad_params = [p for p in model.parameters() if p.grad is not None]
    nonzero_grad_params = [
        p for p in grad_params
        if p.grad is not None and p.grad.detach().abs().sum().item() > 0.0
    ]

    gnorm = grad_norm(model.parameters())

    after_params = [p.detach().clone() for p in model.parameters()]
    changed = parameter_changed(before_params, after_params)

    print()
    print("Tiny training/update smoke test complete")
    print(f"loss={loss.item():.8f}")
    print("grad_params:", len(grad_params))
    print("nonzero_grad_params:", len(nonzero_grad_params))
    print(f"grad_norm={gnorm.item():.8f}")
    print("parameters_changed:", changed)
    print(f"step_time={elapsed:.3f}s")

    if device.type == "cuda":
        print(f"gpu_allocated_mb={torch.cuda.memory_allocated(device) / 1024**2:.3f}")
        print(f"gpu_reserved_mb={torch.cuda.memory_reserved(device) / 1024**2:.3f}")
        print(f"gpu_peak_allocated_mb={torch.cuda.max_memory_allocated(device) / 1024**2:.3f}")


if __name__ == "__main__":
    main()
