# Code and sample availability

## Reviewer question

Please include a code repository with a small sample dataset, and instructions on how to run your model. Maybe even an executable, if your skillset allows for that.

## Response

The model code is available at https://github.com/omkardhamgunde/lightning-nowcasting-fno-unet. The repository includes a small real sample dataset with input maps and observed targets, the required environment configuration, and instructions for installation, preprocessing and inference. The trained model weights and reference predictions are provided in release v1.0.0. Command-line scripts run the model and generate lightning-probability maps and thresholded binary occurrence predictions. A lightweight synthetic demo is also included for checking installation without downloading the trained weights.
