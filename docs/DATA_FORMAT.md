# Input and output format

## Prepared inference input

Use a compressed NPZ readable with `numpy.load(..., allow_pickle=False)`:

| Field | Shape and meaning |
|---|---|
| `x` | float32 `[N, 9, 3, H, W]`; weather is already standardized, lightning is binary |
| `y` | Optional binary `[N, 1, H, W]`; next-hour occurrence |
| `channel_names` | `cape, cp, clwc, r, q, t2m, d2m, w, lightning`, in this order |
| `input_times` | datetime64 `[N, 3]`; three consecutive hourly timestamps |
| `target_times` | datetime64 `[N]`; one hour after the latest input |
| `latitude` | `[H]`; descending north to south |
| `longitude` | `[W]`; ascending west to east |
| `normalization_id` | Scalar Unicode string matching the configuration |
| `sample_kind` | Scalar Unicode string `observed` or `synthetic` |

For the frozen research model, H = 129, W = 135, latitude 38.5° to 6.5° N and longitude 66.5° to 100° E, at 0.25° spacing. The three inputs at t−2, t−1 and t predict the map at t+1. The encoded projection has 56 channels (27 flattened data channels, 27 repeated time-encoding channels and two position channels).

## Physical weather inputs

`prepare_input.py` accepts already aligned maps with `weather` float32 `[N,8,3,129,135]`, `lightning_history` binary `[N,1,3,129,135]`, the coordinates and timestamps above, and optional `y`. All variables must represent matching valid hours. This helper does not download ERA5, infer pressure levels, regrid arbitrary source data, or convert WRF fields.

| Weather variable | Physical units | Level |
|---|---|---|
| CAPE | J/kg | Single level |
| Convective precipitation | m | ERA5 accumulation convention used in the original cache |
| Specific cloud liquid water | kg/kg | 500 hPa |
| Relative humidity | % | 700 hPa |
| Specific humidity | kg/kg | 700 hPa |
| Air temperature | K | 2 m |
| Dewpoint temperature | K | 2 m |
| Pressure vertical velocity | Pa/s | 500 hPa |

The published affine transformation is `(weather − mean) / scale`, independently per weather variable. Do not fit new statistics on a WRF case, new input, validation set or test set. Do not normalize binary lightning. The configuration preserves the frozen model's legacy full-cache normalization, which is not a leakage-free training-only preprocessing pipeline. The legacy cache has standardized-zero CP in 2021–2022; missing CP must not be presented as a measured zero.

By default missing physical inputs raise an error. The explicit `--missing-policy legacy_raw_zero` option reproduces the original raw-zero fill before standardization for nonfinite/fill values; document its use. It does not implement the special missing-year CP policy.

```bash
python prepare_input.py --raw aligned_physical_inputs.npz --output prepared_inputs.npz
```

## Predictions

`prediction.npz` contains float32 `probability` and uint8 `binary`, both `[N,1,H,W]`, coordinates, target timestamps and the strict cutoff. `binary = probability > threshold`. These are cell-hour occurrence labels, not flash counts or flash-rate density. The command also writes `run.json` with settings, checkpoint hash, versions and optional sample scores.

Pattern correlation uses probabilities; it does not change just because a binary cutoff changes. Undefined scores are JSON null. TP/TN/FP/FN are grid-cell-hour counts, not counts of physical flashes.
