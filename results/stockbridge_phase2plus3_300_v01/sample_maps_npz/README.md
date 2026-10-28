# Stockbridge R019 — U-RNN Phase 2+3

## Experiment

- Dataset: Stockbridge Phase 2+3 (300 events)
- Split: frozen test set (45 events)
- Event: R019_stockbridge_T000_T024
- Event index: 1
- Model: U-RNN, 16,583,986 parameters
- Best checkpoint: epoch 100
- CRS: EPSG:27700

## File

001_R019_stockbridge_T000_T024_maps.npz

Source evaluation:
test_eval_r019_maps_v02

## Arrays

- pred_m: (24, 305, 326), float32
- target_m: (24, 305, 326), float32
- diff_m: (24, 305, 326), float32
- steps: T1–T24
- event_name: R019_stockbridge_T000_T024

Depth units: metres.
Predictions are nonnegative (clamped).
Temporal resolution: 5 minutes.

## Reference maxima

- CityCAT maximum depth: 2.260 m
- U-RNN maximum depth: 2.351 m

The file supports spatial and temporal comparison
with DNO and SWE-GNN on the same event.

The original model outputs and canonical datasets
remain archived separately on Comet.
