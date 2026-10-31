# Corrected FNO–U-Net BCE model

## Intended output

Next-hour probability of at least one retained lightning observation in each 0.25° grid cell. A strict cutoff `p > 0.30` produces binary occurrence. This is not a flash-rate-density or individual-flash-count regression model.

## Architecture

Input: `[B,9,3,129,135]`. Flatten channels and time, concatenate the original relative-time and position encodings, then apply a 1 × 1 projection from 56 to 128 channels. Four corrected FNO blocks and a U-Net branch operate in parallel. U-Net widths are 32, 64, 128 and 256, with three 2 × 2 max-pooling steps and three transposed-convolution decoder stages. Skip features are concatenated; bilinear interpolation aligns odd dimensions. Fusion maps 160 → 80 → 1 channel.

The corrected spectral convolution uses `bchw,cohw->bohw` and two disjoint Fourier-row bands. Each retained frequency has its own learned input-to-output-channel matrix. Twenty-four modes are retained in each spectral direction for each band at the published grid size. This differs from the legacy single-corner weight-summation implementation.

The executable `model.py` is byte-identical to the corrected source used for the recorded BCE run. Full model parameter elements, counting each complex parameter once: **77,546,753**. Architecture changes for the synthetic demo are explicit in `configs/demo.json`.

## Training and selection

Standard unweighted BCE; AdamW; initial learning rate 0.0003; batch size 8; seed 20261013; validation BCE controls scheduling, early stopping and checkpoint selection. The run completed 33 epochs and selected epoch 3. The recorded environment was Python 3.11, PyTorch 2.7.1+cu118, NumPy 1.26.4 and an NVIDIA H100 80 GB.

The legacy cache has 40,363 retained hours, producing 40,359 original windows split into 32,287 train, 4,036 validation and 4,036 test windows. The recorded test target period is 2023-07-13 03:00 through 2023-12-31 22:00 using the source timestamps. The source timezone and observation completeness are not independently certified.

## Reproducibility boundaries

- Weather scaling was retained from the original full cache for compatibility, not recomputed from training inputs only. The fixed values and units are in `configs/normalization.json`.
- CP is missing from the original 2021–2022 source files and is standardized zero in those cached years. The original timeline has gaps; the archived full-test benchmark includes gap-containing windows.
- This test period has been inspected in earlier experiments. Single-seed differences do not establish operational superiority. ERA5 reanalysis inputs do not demonstrate a real-time deployment pipeline.
- The synthetic demo has random weights and artificial data. The real sample, when available, uses two consecutive-hour targets; it cannot reproduce full-test aggregate scores by itself.

## Artifact identities

Corrected model source SHA256: `41160b2828c2ce3e965e6d3bb034dae0e643c4ca43a60afe6632943740662e98`.

Selected epoch-3 checkpoint SHA256: `7990dbe199157499d4a70da37eb7e486b6980ad5404f91d6d438033e00b55ce6`.

The weights/real-sample assets have not yet been attached. Download instructions must not claim they are available until the export, checksum check and sample inference have completed.
