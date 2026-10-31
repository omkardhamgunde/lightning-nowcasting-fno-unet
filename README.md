# Lightning occurrence nowcasting with corrected FNO–U-Net

Code and command-line workflows for a corrected two-band Fourier Neural Operator and U-Net hybrid trained with standard binary cross-entropy (BCE).

**Available now:** exact corrected-model source, normalization/grid configuration, a small synthetic sample, a CPU demo, training/inference scripts and automated checks. **Pending:** export and attachment of the trained epoch-3 weights and a small real-data input sample. The synthetic demo runs immediately; it is not a scientific lightning forecast. See [trained weights](checkpoints/README.md).

## Run the small demo

Use Python **3.11**. The recorded research environment used PyTorch 2.7.1 and NumPy 1.26.4. A GPU is not needed for the reduced demo.

```bash
python -m venv .venv
```

Activate the environment:

```bash
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install the CPU dependencies and run:

```bash
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install numpy==1.26.4
python demo.py
```

The command writes `outputs/demo/prediction.npz` and `outputs/demo/run.json`. It uses **untrained random weights, reduced channels/modes, and synthetic data**. It checks the input-to-output workflow only. A second run must use a new destination, for example `python demo.py --output outputs/demo2`.

Optional figure:

```bash
python -m pip install -r requirements-plot.txt
python plot_prediction.py --input outputs/demo/prediction.npz --output outputs/demo/figure.png
```

The probability panel has a 0–1 colorbar; the binary panel has an occurrence/no-occurrence legend and states its threshold.

For an NVIDIA GPU matching the recorded CUDA build, install `torch==2.7.1` from `https://download.pytorch.org/whl/cu118` instead. Select a wheel suitable for your system using the [official PyTorch instructions](https://pytorch.org/get-started/previous-versions/#v271). The supplied `environment.yml` is an alternative Python 3.11 environment definition.

## Run the trained research model

This command becomes runnable after the verified sample assets are attached. Do not replace missing weights with random initialization.

```bash
python predict.py --input sample_assets/real_sample.npz --checkpoint sample_assets/corrected_bce_epoch3.pt --config configs/corrected_bce.json --output outputs/real_sample --device auto
```

`predict.py` verifies the published checkpoint SHA256, channel order, history timing, normalization identifier and spatial dimensions. It then produces:

- `prediction.npz`: per-cell probabilities and strict `probability > 0.30` binary predictions.
- `run.json`: settings, environment, checkpoint identity and metrics for the supplied sample when observed labels are included.

Sample metrics cover only the supplied hours. They are **not** the paper's full-test results. The model predicts whether lightning occurs in a cell-hour; it does not predict individual flash counts or flash-rate density.

Once the assets are available, verify the sample against the exported reference prediction:

```bash
python -m scripts.verify_sample --assets sample_assets --device cpu
```

## Inputs

The research input is `N × 9 × 3 × 129 × 135`: eight weather variables and binary lightning for three hourly maps. Channel order is **CAPE, convective precipitation, cloud liquid water, relative humidity, specific humidity, 2 m temperature, 2 m dewpoint, pressure vertical velocity, lightning**. It predicts the next hour. Weather is standardized; lightning remains 0 or 1.

[Input units, levels, timestamps and NPZ fields](docs/DATA_FORMAT.md) explain how to supply another aligned sample. `prepare_input.py` applies the frozen normalization to already aligned physical weather arrays; it is not a raw NetCDF download/regridding pipeline.

## Model and results

Four FNO blocks operate in parallel with a 2D U-Net branch. Each spectral convolution uses separate positive/negative Fourier-row bands with a distinct channel-mixing matrix for every retained mode. The two branches are concatenated and projected to one output-logit channel. The released configuration uses width 128, 24 modes per band and U-Net base width 32.

The selected research checkpoint is **epoch 3**, chosen by validation BCE. Previously verified full-test scores at `probability > 0.30`, over 4,036 legacy test windows, are:

| CSI | POD | FAR | MCC | Pattern correlation |
|---:|---:|---:|---:|---:|
| 0.31866 | 0.46065 | 0.49168 | 0.48039 | 0.55707 |

These are recorded benchmark results, not results of the synthetic demo. The original cache retains full-cache normalization, missing-year CP handling and time gaps; this inference release does not turn that benchmark into a new independent or operational test. See [model details](docs/MODEL_CARD.md).

## Train on your own prepared data

```bash
python train.py --train train.npz --validation validation.npz --output outputs/new_training --device cuda
```

This starts a fresh BCE run with validation-based checkpoint selection; it does not load the published trained weights or evaluate a test set. Training and validation bundles must be chronological and must not share input or target hours. See [training instructions](docs/TRAINING.md).

## Checks

```bash
python -m unittest discover -s tests -v
```

Checks cover the exact corrected-source hash, Fourier-mode behavior, both spectral bands, finite gradients, time/channel validation, normalization, strict cutoffs, checkpoint verification, output protection, training and inference. GitHub Actions runs them on Python 3.11 with the pinned CPU dependencies.

## Authors exporting the existing HPC assets

The public exporter requires explicit source paths and verifies the existing checkpoint and model hashes. It reads the cache privately, selects two test examples by chronological availability, runs frozen inference and writes a portable archive. It never retrains or modifies original files.

```bash
python -m scripts.export_hpc_assets --cache /path/to/preprocessed_lightning_data.pt --checkpoint /path/to/best_epoch_0003.pt --model-source /path/to/corrected_bce/model.py --output /new/path/reviewer_assets --device cuda
```

For this project's IITM execution, only nodes **iitm-ai-01** and **iitm-ai-02** are permitted. The project-specific terminal launcher supplied separately pins node 02 and checks the runtime host. An asset download URL will be added only after the files have actually been exported and verified.

## Attribution

The original FNO–U-Net baseline implementation was supplied by Karan Aditya Harshey. This repository packages the corrected spectral implementation, BCE experiment configuration and reproducible sample workflow. The model source is byte-identical to the verified corrected research source; its SHA256 is recorded in `configs/corrected_bce.json`. No claim is made that the combined architecture is an unchanged implementation of a single earlier paper. The U-Net encoder–decoder principle originates from [Ronneberger et al. (2015)](https://arxiv.org/abs/1505.04597).
