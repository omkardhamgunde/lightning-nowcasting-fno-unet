# Lightning nowcasting with FNO–U-Net

Trained corrected FNO–U-Net with BCE loss for next-hour lightning occurrence. Model code, two real sample cases and instructions are provided here; the saved weights are downloaded automatically.

## Run

Use **Python 3.11**.

```bash
git clone https://github.com/omkardhamgunde/lightning-nowcasting-fno-unet.git
cd lightning-nowcasting-fno-unet
python -m pip install -r requirements.txt
python run_model.py
```

The first run downloads and verifies the trained weights (~572 MB). Later runs reuse them. A GPU is not required.

## Files

- `model.py` — corrected FNO–U-Net architecture.
- `run_model.py` — downloads weights, runs prediction and saves results.
- `sample_data.npz` — two real examples with three-hour input histories and observed next-hour targets.
- `requirements.txt` — required dependencies.

The sample is already normalized and has input shape `2 × 9 × 3 × 129 × 135`. The nine channels are CAPE, convective precipitation, cloud liquid water, relative humidity, specific humidity, 2 m temperature, 2 m dewpoint, vertical velocity and past lightning. Target hours are **13 July 2023 at 03:00 and 04:00**.

## Output

- `output/prediction.npz` — probability maps and binary predictions for both examples.
- `output/prediction.png` — probability, binary prediction and observed occurrence for the first example.

Binary occurrence is **1 when probability > 0.30**, otherwise 0. These outputs are occurrence predictions, not flash counts or flash-rate density. Sample results cover only these two examples.

To run again without overwriting results:

```bash
python run_model.py --output output2
```

The unchanged epoch-3 weights are available in [release v1.0.0](https://github.com/omkardhamgunde/lightning-nowcasting-fno-unet/releases/tag/v1.0.0). Original baseline implementation: Karan Aditya Harshey; corrected model and sample workflow: Omkar Dhamgunde.
