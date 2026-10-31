# Sample data

`synthetic_sample.npz` is an openly generated two-window sample with shape **2 × 9 × 3 × 17 × 19**. It contains artificial weather fields, moving binary lightning patterns, synthetic timestamps, and an illustrative grid. It has no observed lightning records and supports the reduced, untrained demo configuration only.

Recreate it with `python -m scripts.make_synthetic_sample --output samples/new_synthetic_sample.npz`.

The real-data sample for the frozen checkpoint is not yet included. The HPC exporter will select the first two chronologically eligible, consecutive-hour windows in the original test split, independent of lightning activity and prediction quality. Full 129 × 135 input maps are retained because this FNO uses global spatial operations; a cropped map is not equivalent input.

`real_sample.npz` will contain standardized weather channels plus binary past lightning, an observed binary target, coordinates and exact timestamps. Its small sample scores do not reproduce the 4,036-window paper scores.
