# Stockbridge U-RNN Phase 2 — 100 events

Historical baseline:
- Train: 70 events
- Validation: 15 events
- Test: 15 events

## Results provenance

- metrics.csv:
  Original full_mse_v01/metrics.csv

- aggregate_metrics.json:
  Original full_mse_v01/test_eval/aggregate_metrics.json

- per_event_metrics.csv:
  Recovered from full_mse_v01/test_eval_r019/per_event_metrics.csv

## Recovery verification

The original test_eval/per_event_metrics.csv contained only
column headers.

The recovered CSV contains all 15 unique historical test events.

Its event membership was verified against stockbridge_test.txt.

Its accuracy metrics and confusion counts were compared against
the original aggregate_metrics.json and matched.

Inference timing may differ between evaluation runs.

The original files were preserved without modification.
