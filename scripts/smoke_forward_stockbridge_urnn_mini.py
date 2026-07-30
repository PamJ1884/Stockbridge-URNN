#!/usr/bin/env python3
"""Real-data U-RNN forward-pass smoke test using converted Stockbridge mini dataset.

This script:
1. Loads one Stockbridge event through the original Dynamic2DFlood loader.
2. Builds U-RNN inputs using original preprocess_inputs().
3. Runs the ED model recurrently for T=25 timesteps.
4. Crops predictions from padded 308x328 back to original 305x326.
5. Prints prediction/label shapes and basic ranges.

No training is performed.
"""

import sys
from pathlib import Path

import torch
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


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("Data root:", DATA_ROOT)
    print("Event list:", EVENT_LIST)
    print("Device:", device)

    dataset = Dynamic2DFlood(
        data_root=str(DATA_ROOT),
        split="train",
        event_list_file=str(EVENT_LIST),
        duration=DURATION,
        location=LOCATION,
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    batch = next(iter(loader))
    inputs = batch[0]
    label = batch[1]
    event_dir = batch[2]

    print("Event:", event_dir)
    print("Label padded shape:", tuple(label.shape))
    print("Label padded min/max mm:", float(label.min()), float(label.max()))

    net_cfg = load_net_config()
    input_channels = get_input_channels(net_cfg, HISTORICAL_NUMS)

    params = get_network_params(
        False,
        PAD_H,
        PAD_W,
        input_channels=input_channels,
        net_cfg=net_cfg,
    )

    model = ED(
        clstm_flag=False,
        encoder_params=params[0],
        decoder_params=params[1],
        cls_thred=0.5,
        use_checkpoint=False,
        input_height=PAD_H,
        input_width=PAD_W,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())

    print("Model: ED")
    print("Input channels:", input_channels)
    print(f"Parameters: {total_params / 1e6:.2f} million")

    enc1, enc2, enc3, dec1, dec2, dec3 = initialize_states(
        device,
        PAD_H,
        PAD_W,
        net_cfg=net_cfg,
    )

    predictions = []

    model.eval()
    with torch.no_grad():
        for t in range(DURATION):
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

            # out shape: (B, 1, PAD_H, PAD_W)
            out_crop = out[..., :ORIG_H, :ORIG_W]
            predictions.append(out_crop.detach().cpu())

            if t in [0, 1, 2, 24]:
                print(
                    f"t={t:02d}",
                    "input=", tuple(x.shape),
                    "raw_out=", tuple(out.shape),
                    "crop=", tuple(out_crop.shape),
                    "pred_min/max=", float(out_crop.min()), float(out_crop.max()),
                )

    pred = torch.cat(predictions, dim=1)  # (B, T, ORIG_H, ORIG_W)
    label_crop = label[:, :, :ORIG_H, :ORIG_W]

    print()
    print("Forward pass complete")
    print("Prediction cropped shape:", tuple(pred.shape))
    print("Label cropped shape:", tuple(label_crop.shape))
    print("Expected shape:", (1, DURATION, ORIG_H, ORIG_W))

    print()
    print("Prediction range:", float(pred.min()), float(pred.max()))
    print("Label range mm:", float(label_crop.min()), float(label_crop.max()))

    # Shape check for DNO-style forecast comparison.
    pred_forecast = pred[:, 1:, :, :]
    label_forecast = label_crop[:, 1:, :, :]

    print()
    print("Forecast comparison window:")
    print("Prediction T1-T24:", tuple(pred_forecast.shape))
    print("Label T1-T24:", tuple(label_forecast.shape))


if __name__ == "__main__":
    main()
