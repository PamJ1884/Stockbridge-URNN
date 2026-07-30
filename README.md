# Stockbridge U-RNN

Independent repository for adapting and testing U-RNN on the Stockbridge CityCAT dataset.

This repository keeps the U-RNN Stockbridge workflow separate from the DNO/TL-DNO UrbanFloodCast work.

## Purpose

Compare U-RNN against the existing Stockbridge DNO baseline using the same dataset, split, forecast horizon, and evaluation metrics.

## Baseline comparison target

- Dataset: 100 CityCAT simulations
- Split: 70 train / 15 validation / 15 test
- Original tensor shape: [305, 326, 25, 5]
- Channels: H, Vx, Vy, rainfall, DEM
- Forecast: T0 input, T1-T24 output
- Evaluation: NSE, correlation, CSI, RMSE, MAE

## Current technical findings

- Original U-RNN can run on the Ubuntu desktop GPU.
- Raw Stockbridge grid 305x326 fails because of encoder-decoder size mismatch.
- Padding to 308x328 and cropping back to 305x326 works.
- Converted Stockbridge mini dataset loads through Dynamic2DFlood.
- preprocess_inputs produces tensors of shape (1, 1, 9, 308, 328).
- Real-data forward pass works.
- One-step training/update smoke test works.
- Multi-step full-resolution local training exceeded RTX 3050 6 GB memory.

## External dependency

This repo assumes the original U-RNN repository is available at:

~/flood/01_repositories/U-RNN

The original U-RNN source code is kept separate and mostly unmodified.
