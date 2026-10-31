# Training interface

`train.py` trains a new corrected FNO–U-Net with standard, unweighted `binary_cross_entropy_with_logits(..., reduction="mean")`. It is an explicit interface for new prepared datasets, not a claim to repeat the historical run from a two-window sample.

Supply separate standardized train and validation NPZ bundles following [DATA_FORMAT.md](DATA_FORMAT.md), with binary targets `y`. The loader checks channel order, timing and grid orientation. Split chronologically before preparing examples. It rejects shared input or target hours across training and validation, including overlapping temporal contexts.

The published legacy checkpoint was trained on its original cache and split, which contained adjacent temporal contexts and full-cache scaling. This stricter new-data interface does not hide those historical differences.

```bash
python train.py --train train.npz --validation validation.npz --config configs/corrected_bce.json --output outputs/new_training --device cuda
```

| Setting | Default |
|---|---|
| Initialization | Fresh random weights |
| Seed | 20261013 |
| Batch size | 8 |
| Shuffle | False |
| Optimizer | AdamW, learning rate 0.0003, weight decay 0.01 |
| Scheduler | ReduceLROnPlateau on validation BCE, factor 0.5, patience 2 |
| Gradient norm clipping | 1.0 |
| Maximum epochs | 300 |
| Early stopping | 30 consecutive epochs without a lower validation BCE |
| Checkpoint selection | Minimum validation BCE |
| Loss weights | No focal/Dice, positive-class, spatial or regional weights |

Outputs are `best.pt` (plain state_dict), `history.json`, `model_config.json` and `training_summary.json`. Use the generated configuration for subsequent inference:

```bash
python predict.py --input held_out.npz --checkpoint outputs/new_training/best.pt --config outputs/new_training/model_config.json --output outputs/held_out --device cuda
```

The existing epoch-3 research checkpoint has about 77.5 million parameter elements when each complex parameter is counted once. Full training needs substantially more memory than the reduced demo; the recorded run used an NVIDIA H100 80 GB. The supplied two-window sample is for inference only. Full-data training requires the separately obtained dataset and suitable computing resources.
