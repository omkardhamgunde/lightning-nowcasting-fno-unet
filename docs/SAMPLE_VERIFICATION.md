# Verification of the released real sample

The released sample contains two original-test targets, **13 July 2023 at 03:00 and 04:00**, selected as the first two chronologically eligible examples with consecutive hourly histories. Each uses three input maps. Full 129 × 135 maps are retained. These are not selected by lightning activity or model performance.

The checkpoint is the original validation-selected corrected BCE epoch 3, with SHA256 `7990dbe199157499d4a70da37eb7e486b6980ad5404f91d6d438033e00b55ce6`. The model source, normalization, coordinates and reference arrays are checksum-verified. No weights were retrained, converted or resaved. The archive also includes the HPC export manifest.

The reference prediction was generated on an NVIDIA H100 using PyTorch 2.7.1+cu118 and NumPy 1.26.4. Local Windows CPU checks used both Python 3.11/PyTorch 2.7.1+cpu/NumPy 1.26.4 and Python 3.13/PyTorch 2.8.0+cpu. Both CPU runs produced the same measured differences from the reference:

| Check | Result |
|---|---:|
| Maximum absolute probability difference | 0.0002186894 |
| Mean absolute probability difference | 0.0000011888 |
| Binary differences at strict probability > 0.30 | 0 of 34,830 grid-cell predictions |

The original strict `allclose(rtol=1e-4, atol=2e-5)` test failed. This is recorded rather than presented as a bitwise match. The checkpoint, inputs and source hashes match exactly, and the two CPU versions agree. CPU/GPU numerical behavior can differ; [PyTorch documents this explicitly](https://docs.pytorch.org/docs/2.7/notes/numerical_accuracy.html). The precise contributing kernel was not isolated.

The released cross-backend check therefore uses explicit bounds of **maximum absolute error ≤ 0.0005**, **mean absolute error ≤ 0.000005**, and **exact binary equality at > 0.30 for these two sample cases**. Any threshold crossing is rejected. The original tighter test remains available:

```bash
python -m scripts.verify_sample --assets sample_assets --device cpu
python -m scripts.verify_sample --assets sample_assets --device cpu --strict-probabilities
```

These bounds describe sample numerical reproducibility, not forecasting accuracy, statistical uncertainty or a guarantee of identical predictions on every platform or test case. The recorded full-test benchmark remains separate.
