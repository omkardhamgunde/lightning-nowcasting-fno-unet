# Trained weights

The corrected BCE model uses the validation-selected epoch 3 checkpoint.

Expected SHA256:

```text
7990dbe199157499d4a70da37eb7e486b6980ad5404f91d6d438033e00b55ce6
```

The checkpoint is not yet included in this repository or its releases. The author-side exporter creates a verified asset bundle from the existing HPC checkpoint; it does not retrain the model. After the assets are returned and verified, the checkpoint can be attached to a GitHub Release and the download instructions updated. Do not substitute the historical composite-loss checkpoint or random weights.

Once available, place the file here as `corrected_bce_epoch3.pt`. Large weights belong in a release asset, not ordinary Git history.
