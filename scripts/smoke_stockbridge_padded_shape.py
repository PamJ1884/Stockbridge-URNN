#!/usr/bin/env python3
"""Smoke test U-RNN forward pass using padded Stockbridge dimensions."""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.lib.model.networks.model import ED
from src.lib.utils.net_config import load_net_config, get_input_channels
from src.lib.model.networks.net_params import get_network_params
from src.lib.utils.general import initialize_states


def main():
    original_H, original_W = 305, 326
    padded_H, padded_W = 308, 328
    T = 2
    historical_nums = 3

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net_cfg = load_net_config()
    input_channels = get_input_channels(net_cfg, historical_nums)

    params = get_network_params(
        False,
        padded_H,
        padded_W,
        input_channels=input_channels,
        net_cfg=net_cfg,
    )

    model = ED(
        clstm_flag=False,
        encoder_params=params[0],
        decoder_params=params[1],
        cls_thred=0.5,
        use_checkpoint=False,
        input_height=padded_H,
        input_width=padded_W,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())

    print("U-RNN Stockbridge padded-shape smoke test")
    print("Device:", device)
    print("Input channels:", input_channels)
    print(f"Parameters: {total_params / 1e6:.2f} million")
    print("Original grid:", original_H, original_W)
    print("Padded grid:", padded_H, padded_W)

    enc1, enc2, enc3, dec1, dec2, dec3 = initialize_states(
        device,
        padded_H,
        padded_W,
        net_cfg=net_cfg,
    )

    outputs = []

    model.eval()
    with torch.no_grad():
        for t in range(T):
            x = torch.randn(
                1,
                1,
                input_channels,
                padded_H,
                padded_W,
                device=device,
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

            cropped = out[..., :original_H, :original_W]

            print(f"Step {t}: raw output shape     = {tuple(out.shape)}")
            print(f"Step {t}: cropped output shape = {tuple(cropped.shape)}")

            outputs.append(cropped.detach().cpu())

    outputs = torch.cat(outputs, dim=1)

    print()
    print("Smoke test complete")
    print("Cropped output sequence shape:", tuple(outputs.shape))
    print("Expected cropped shape: (1, 2, 305, 326)")


if __name__ == "__main__":
    main()
