"""Run the trained corrected FNO–U-Net on the included real sample."""
import argparse
import hashlib
from pathlib import Path
import shutil
import socket
import tempfile
import urllib.request
import zipfile

import numpy as np
import torch

from model import FNO2DLightning

ROOT = Path(__file__).resolve().parent
WEIGHTS_URL = "https://github.com/omkardhamgunde/lightning-nowcasting-fno-unet/releases/download/v1.0.0/reviewer_sample_assets.zip"
ARCHIVE_SHA256 = "7d2138a74338225021e97a22db85a60700c3664817e953ffb07379fd86b38e67"
WEIGHTS_SHA256 = "7990dbe199157499d4a70da37eb7e486b6980ad5404f91d6d438033e00b55ce6"
VARIABLES = ["cape", "cp", "clwc", "r", "q", "t2m", "d2m", "w", "lightning"]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def get_weights(path):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        pending = path.with_name(path.name + ".pending")
        if pending.exists():
            raise FileExistsError("An unfinished weight download exists: " + str(pending))
        print("Downloading saved model weights (~572 MB); this is needed only once.", flush=True)
        with tempfile.TemporaryDirectory(prefix="lightning-weights-") as temporary:
            archive = Path(temporary) / "weights.zip"
            request = urllib.request.Request(WEIGHTS_URL, headers={"User-Agent": "lightning-nowcasting"})
            with urllib.request.urlopen(request, timeout=120) as response, archive.open("xb") as stream:
                shutil.copyfileobj(response, stream, length=4 * 1024 * 1024)
            if sha256(archive) != ARCHIVE_SHA256:
                raise ValueError("Downloaded archive checksum does not match")
            try:
                with zipfile.ZipFile(archive) as bundle:
                    name = "corrected_bce_epoch3.pt"
                    if bundle.namelist().count(name) != 1:
                        raise ValueError("Expected trained weights are missing")
                    with bundle.open(name) as source, pending.open("xb") as destination:
                        shutil.copyfileobj(source, destination, length=4 * 1024 * 1024)
                if sha256(pending) != WEIGHTS_SHA256:
                    raise ValueError("Trained weight checksum does not match")
                pending.rename(path)
            except Exception:
                pending.unlink(missing_ok=True)
                raise
    if sha256(path) != WEIGHTS_SHA256:
        raise ValueError("Use the saved corrected BCE epoch-3 weights; checkpoint identity differs")
    return path


def load_sample(path):
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    x = data["x"]
    if x.ndim != 5 or x.shape[1:] != (9, 3, 129, 135) or len(x) < 1 or x.dtype != np.float32:
        raise ValueError("Expected preprocessed float32 input with shape N × 9 × 3 × 129 × 135")
    if not np.isfinite(x).all() or not np.isin(x[:, -1], [0, 1]).all():
        raise ValueError("Inputs must be finite and lightning history must be binary")
    if data["channel_names"].tolist() != VARIABLES:
        raise ValueError("Input variable order differs from the trained model")
    if str(data["normalization_id"].item()) != "corrected-bce-legacy-eight-weather-v1":
        raise ValueError("Input normalization differs from the trained model")
    for key, expected in [("latitude", np.linspace(38.5, 6.5, 129)),
                          ("longitude", np.linspace(66.5, 100., 135))]:
        if not np.array_equal(data[key], expected):
            raise ValueError("Input coordinates differ from the model grid")
    times, targets = data["input_times"], data["target_times"]
    if (times.shape != (len(x), 3) or targets.shape != (len(x),)
            or times.dtype.kind != "M" or targets.dtype.kind != "M"
            or np.isnat(times).any() or np.isnat(targets).any()
            or not (np.diff(times, axis=1) == np.timedelta64(1, "h")).all()
            or not (targets - times[:, -1] == np.timedelta64(1, "h")).all()):
        raise ValueError("Each sample needs three consecutive input hours and a next-hour target")
    if "y" in data and (data["y"].shape != (len(x), 1, 129, 135) or not np.isin(data["y"], [0, 1]).all()):
        raise ValueError("Observed targets must be binary maps")
    return data


def plot(data, probabilities, binary, threshold, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    count = 3 if "y" in data else 2
    fig, axes = plt.subplots(1, count, figsize=(5 * count, 5), constrained_layout=True)
    extent = [data["longitude"][0] - .125, data["longitude"][-1] + .125,
              data["latitude"][-1] - .125, data["latitude"][0] + .125]
    image = axes[0].imshow(probabilities[0, 0], origin="upper", extent=extent, vmin=0, vmax=1, cmap="viridis")
    axes[0].set_title("Lightning probability")
    fig.colorbar(image, ax=axes[0], label="Probability", shrink=.8)
    cmap = ListedColormap(["white", "#17324d"])
    axes[1].imshow(binary[0, 0], origin="upper", extent=extent, vmin=0, vmax=1, cmap=cmap)
    axes[1].set_title(f"Binary prediction > {threshold:.2f}")
    axes[1].legend(handles=[Patch(facecolor="white", edgecolor="black", label="No occurrence"),
                            Patch(facecolor="#17324d", label="Occurrence")], fontsize=8)
    if count == 3:
        axes[2].imshow(data["y"][0, 0], origin="upper", extent=extent, vmin=0, vmax=1, cmap=cmap)
        axes[2].set_title("Recorded occurrence")
    for ax in axes:
        ax.set_xlabel("Longitude (°E)"); ax.set_ylabel("Latitude (°N)")
    fig.suptitle(np.datetime_as_string(data["target_times"][0], unit="h"))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "sample_data.npz")
    parser.add_argument("--weights", type=Path, default=ROOT / "model_weights.pt")
    parser.add_argument("--output", type=Path, default=ROOT / "output")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    host = socket.gethostname().split(".")[0]
    if host.startswith("iitm-ai-") and host not in {"iitm-ai-01", "iitm-ai-02"}:
        raise RuntimeError("This project's HPC inference is restricted to nodes 01 and 02")
    if args.output.exists():
        raise FileExistsError("Output already exists; use --output with a new folder name")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use the default CPU setting")
    data = load_sample(args.input)
    weights = get_weights(args.weights)
    torch.set_num_threads(4)
    model = FNO2DLightning(in_channels=56, width=128, modes=24, unet_base=32)
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
    model = model.to(args.device).eval()
    print("Running the saved model on the real sample...", flush=True)
    with torch.inference_mode():
        probabilities = np.concatenate([
            torch.sigmoid(model(torch.from_numpy(data["x"][i:i + 1]).to(args.device))).cpu().numpy()
            for i in range(len(data["x"]))
        ])
    if not np.isfinite(probabilities).all():
        raise ValueError("Nonfinite model output")
    threshold = .30
    binary = (probabilities > np.float32(threshold)).astype(np.uint8)
    args.output.mkdir(parents=True, exist_ok=False)
    payload = dict(probability=probabilities, binary=binary, target_times=data["target_times"],
                   latitude=data["latitude"], longitude=data["longitude"], threshold=np.float32(threshold))
    if "y" in data:
        payload["target"] = data["y"]
    np.savez_compressed(args.output / "prediction.npz", **payload)
    plot(data, probabilities, binary, threshold, args.output / "prediction.png")
    print("Saved predictions:", args.output / "prediction.npz")
    print("Saved figure:", args.output / "prediction.png")


if __name__ == "__main__":
    main()
