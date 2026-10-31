# Sample data

`synthetic_sample.npz` is an openly generated two-window sample with shape **2 × 9 × 3 × 17 × 19**. It contains artificial weather fields, moving binary lightning patterns, synthetic timestamps, and an illustrative grid. It has no observed lightning records and supports the reduced, untrained demo configuration only.

Recreate it with `python -m scripts.make_synthetic_sample --output samples/new_synthetic_sample.npz`.

`real_sample.npz` contains the first two chronologically eligible, consecutive-hour windows in the original test split, independent of lightning activity and prediction quality. The target hours are **13 July 2023 at 03:00 and 04:00**. Each uses three past hours. Full 129 × 135 input maps are retained because this FNO uses global spatial operations; a cropped map is not equivalent input. The file is about 2.1 MB and is included directly in the repository as well as in the release.

The sample contains standardized weather channels plus binary past lightning, observed binary targets, coordinates and exact timestamps. `expected_prediction.npz` contains the HPC reference probabilities and binary labels; `real_sample_manifest.json` records the artifact hashes, environment and selection rule. Its small sample scores do not reproduce the 4,036-window paper scores. [Verification details](../docs/SAMPLE_VERIFICATION.md) explain the measured CPU/GPU differences.
