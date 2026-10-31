"""Author-side read-only export of verified weights and two real inference samples."""
import argparse
import importlib
import json
from pathlib import Path
import shutil
import socket
import time
import zipfile

import numpy as np
import torch

from lightning_nowcasting.data import VARIABLES, read_bundle, read_json, save_json
from lightning_nowcasting.inference import load_model, predict, sha256, choose_device
from lightning_nowcasting.metrics import occurrence_metrics

ROOT = Path(__file__).resolve().parents[1]


def safe_cache(path):
    """Only the reviewed tensor/NumPy cache schema; never weights_only=False."""
    if not zipfile.is_zipfile(path): raise ValueError("Expected a ZIP-format PyTorch cache")
    if __import__("os").name != "nt":
        import mmap
        if torch.serialization.get_default_mmap_options() != mmap.MAP_PRIVATE:
            raise ValueError("Cache mapping must be private and read-only to the source")
    multiarray = importlib.import_module("numpy.core.multiarray")
    allowed = [np.ndarray, np.dtype,
               (multiarray._reconstruct, "numpy.core.multiarray._reconstruct"),
               (multiarray._reconstruct, "numpy._core.multiarray._reconstruct")]
    allowed.extend({type(np.dtype(n)) for n in ["float32", "float64", "int32", "int64", "bool", "datetime64[ns]"]})
    with torch.serialization.safe_globals(allowed):
        return torch.load(path, map_location="cpu", weights_only=True, mmap=True)


def identity(path):
    stat = Path(path).stat()
    return stat.st_size, stat.st_mtime_ns


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-source", type=Path, required=True, help="Reviewed HPC corrected_bce/model.py")
    parser.add_argument("--output", type=Path, required=True, help="New export folder")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()
    host = socket.gethostname().split(".")[0]
    if host.startswith("iitm-ai-") and host not in {"iitm-ai-01", "iitm-ai-02"}:
        raise RuntimeError("Only nodes 01 and 02 may run this project's HPC export")
    if args.output.exists(): raise FileExistsError("Export folder already exists; inspect it before retrying")
    config = read_json(ROOT / "configs/corrected_bce.json")
    norms = read_json(ROOT / "configs/normalization.json")
    source_hash = sha256(args.model_source)
    if source_hash != config["model_source_sha256"] or sha256(ROOT / "lightning_nowcasting/model.py") != source_hash:
        raise ValueError("HPC source and published model must match the pinned corrected model exactly")
    checkpoint_hash = sha256(args.checkpoint)
    if checkpoint_hash != config["checkpoint_sha256"]: raise ValueError("Selected epoch-3 checkpoint changed")
    cache_before = identity(args.cache)
    if cache_before != (25305628432, 1780054784000000000):
        raise ValueError("Original cache identity changed; stop before extracting any data")
    print("Verified original source and checkpoint. Loading the existing cache privately.", flush=True)
    data = safe_cache(args.cache)
    times = np.asarray(data["time_index"]).astype("datetime64[ns]")
    if (len(times) != 40363 or tuple(data["era5"].shape) != (40363, 8, 129, 135)
            or tuple(data["lightning"].shape) != (40363, 129, 135)):
        raise ValueError("Unexpected audited cache dimensions")
    if np.isnat(times).any() or not (np.diff(times) > np.timedelta64(0, "ns")).all():
        raise ValueError("Cache timestamps are missing or not increasing")
    for key, original in [("latitude", "lat_vals"), ("longitude", "lon_vals")]:
        if not np.array_equal(np.asarray(data[original]), np.asarray(norms[key])):
            raise ValueError("Model coordinates differ from the frozen grid")
    total = len(times) - 4
    start = int(.9 * total)
    if times[start + 3] != np.datetime64("2023-07-13T03:00:00", "ns"):
        raise ValueError("Test split boundary changed")
    selected = []
    for index in range(start, total):
        if (np.diff(times[index:index + 4]) == np.timedelta64(1, "h")).all():
            selected.append(index)
            if len(selected) == 2: break
    if len(selected) != 2: raise ValueError("Cannot find two consecutive-hour test examples")
    inputs = []
    for index in selected:
        weather = data["era5"][index:index + 3]
        lightning = data["lightning"][index:index + 3].unsqueeze(1)
        inputs.append(torch.cat([weather, lightning], 1).permute(1, 0, 2, 3).contiguous().numpy())
    payload = dict(x=np.stack(inputs).astype(np.float32),
                   y=np.stack([data["lightning"][i + 3].numpy()[None] for i in selected]).astype(np.uint8),
                   input_times=np.stack([times[i:i + 3] for i in selected]),
                   target_times=times[np.asarray(selected) + 3], latitude=np.asarray(data["lat_vals"]),
                   longitude=np.asarray(data["lon_vals"]), channel_names=np.array(VARIABLES),
                   normalization_id=np.array(config["normalization_id"]), sample_kind=np.array("observed"))
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.output / "real_sample.npz", **payload)
    bundle = read_bundle(args.output / "real_sample.npz", config, require_target=True)
    torch.set_num_threads(4)
    device = choose_device(args.device)
    model, _ = load_model(config, args.checkpoint, device)
    started = time.monotonic()
    probability = predict(model, bundle["x"], device)
    np.savez_compressed(args.output / "expected_prediction.npz", probability=probability,
                        binary=(probability > np.float32(.3)).astype(np.uint8),
                        target_times=bundle["target_times"], threshold=np.float32(.3))
    # Preserve the exact state_dict bytes and its published SHA256, not a converted checkpoint.
    shutil.copyfile(args.checkpoint, args.output / "corrected_bce_epoch3.pt")
    if sha256(args.output / "corrected_bce_epoch3.pt") != checkpoint_hash:
        raise ValueError("Checkpoint copy checksum failed")
    if identity(args.cache) != cache_before or sha256(args.checkpoint) != checkpoint_hash or sha256(args.model_source) != source_hash:
        raise ValueError("Protected source changed during the export")
    report = {"schema": "corrected-bce-sample-assets-v1", "source_files_unchanged": True,
              "checkpoint_sha256": checkpoint_hash, "model_source_sha256": source_hash,
              "selected_epoch": 3, "training_performed": False, "threshold_fitting": False,
              "sample_selection": "First two chronological original-test targets with consecutive hourly histories; no activity/skill filter",
              "target_times": [str(t) for t in payload["target_times"]], "input_shape": list(payload["x"].shape),
              "sample_scores": occurrence_metrics(probability, payload["y"], .3),
              "sample_scores_are_full_test_results": False, "prediction_seconds": time.monotonic() - started,
              "environment": {"torch": str(torch.__version__), "numpy": str(np.__version__),
                              "device": str(device), "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None},
              "asset_sha256": {name: sha256(args.output / name) for name in
                  ["corrected_bce_epoch3.pt", "real_sample.npz", "expected_prediction.npz"]}}
    save_json(args.output / "real_sample_manifest.json", report)
    shutil.copyfile(ROOT / "configs/corrected_bce.json", args.output / "model_config.json")
    shutil.copyfile(ROOT / "configs/normalization.json", args.output / "normalization.json")
    archive = args.output / "reviewer_sample_assets.zip"
    print("Compressing verified weights and small sample; no source data are modified.", flush=True)
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=2) as z:
        for name in ["corrected_bce_epoch3.pt", "real_sample.npz", "expected_prediction.npz",
                     "real_sample_manifest.json", "model_config.json", "normalization.json"]:
            z.write(args.output / name, name)
    print("EXPORT COMPLETE. Return:", archive, flush=True)


if __name__ == "__main__": main()
