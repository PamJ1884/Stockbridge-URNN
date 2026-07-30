#!/usr/bin/env python3
"""Smoke test U-RNN forward pass with Stockbridge raster dimensions."""

import sys
from pathlib import Path

import torch

U_RNN_CODE = Path.home() / "flood/01_repositories/U-RNN/code"
if not U_RNN_CODE.exists():
    raise FileNotFoundError(f"Expected U-RNN code folder not found: {U_RNN_CODE}")
if str(U_RNN_CODE) not in sys.path:
    sys.path.insert(0, str(U_RNN_CODE))

from src.lib.model.networks.model import ED
from src.lib.utils.net_config import load_net_config, get_input_channels
from src.lib.model.networks.net_params import get_network_params
from src.lib.utils.general import initialize_states


def main():
    H, W = 305, 326
    T = 2
    historical_nums = 3

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net_cfg = load_net_config()
    input_channels = get_input_channels(net_cfg, historical_nums)

    params = get_network_params(
        False,
        H,
        W,
        input_channels=input_channels,
        net_cfg=net_cfg,
    )

    model = ED(
        clstm_flag=False,
        encoder_params=params[0],
        decoder_params=params[1],
        cls_thred=0.5,
        use_checkpoint=False,
        input_height=H,
        input_width=W,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())

    print("U-RNN Stockbridge shape smoke test")
    print("Device:", device)
    print("Input channels:", input_channels)
    print(f"Parameters: {total_params / 1e6:.2f} million")
    print("Grid:", H, W)

    enc1, enc2, enc3, dec1, dec2, dec3 = initialize_states(
        device,
        H,
        W,
        net_cfg=net_cfg,
    )

    outputs = []

    model.eval()
    with torch.no_grad():
        for t in range(T):
            x = torch.randn(1, 1, input_channels, H, W, device=device)

            out, enc1, enc2, enc3, dec1, dec2, dec3 = model(
                x,
                enc1,
                enc2,
                enc3,
                dec1,
                dec2,
                dec3,
            )

            outputs.append(out.detach().cpu())
            print(f"Step {t}: output shape = {tuple(out.shape)}")

    outputs = torch.cat(outputs, dim=1)

    print()
    print("Smoke test complete")
    print("Output sequence shape:", tuple(outputs.shape))


if __name__ == "__main__":
    main()
