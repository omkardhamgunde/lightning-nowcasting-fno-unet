# Trained weights

The corrected BCE model uses the validation-selected epoch 3 checkpoint.

Expected SHA256:

```text
7990dbe199157499d4a70da37eb7e486b6980ad5404f91d6d438033e00b55ce6
```

The checkpoint is included in `reviewer_sample_assets.zip` in [release v1.0.0](https://github.com/omkardhamgunde/lightning-nowcasting-fno-unet/releases/tag/v1.0.0). The bundle is approximately 572 MB compressed and 615 MB expanded. It also includes the real sample, normalization/configuration, manifest and reference predictions. The original state_dict bytes were preserved; the model was not retrained or resaved.

Run `python -m scripts.download_assets --output sample_assets` from the repository root to download and verify it. The checkpoint will be installed as `sample_assets/corrected_bce_epoch3.pt`. Do not substitute the historical composite-loss checkpoint or random weights. Large weights remain in the release rather than ordinary Git history.
