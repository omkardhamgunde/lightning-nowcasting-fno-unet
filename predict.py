#!/usr/bin/env python3
"""Run the frozen corrected model, or an explicitly untrained synthetic demo."""
import argparse
from pathlib import Path
import platform
import sys

import numpy as np
import torch

from lightning_nowcasting.data import read_bundle, read_json, save_json
from lightning_nowcasting.inference import choose_device, load_model, predict
from lightning_nowcasting.metrics import occurrence_metrics

ROOT = Path(__file__).resolve().parent


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Standardized input bundle; see docs/DATA_FORMAT.md")
    parser.add_argument("--checkpoint", type=Path, help="Plain trained state_dict; required unless --demo")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/corrected_bce.json")
    parser.add_argument("--output", type=Path, required=True, help="New output directory; never overwrites an earlier run")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--threshold", type=float, help="Fixed cutoff; no threshold fitting is performed")
    parser.add_argument("--demo", action="store_true", help="Random weights and synthetic data only; not a scientific prediction")
    args = parser.parse_args(argv)
    if args.output.exists(): raise FileExistsError("Output already exists: " + str(args.output))
    if args.threads < 1: raise ValueError("Threads must be positive")
    torch.set_num_threads(args.threads)
    config = read_json(args.config)
    threshold = config["threshold"] if args.threshold is None else args.threshold
    if not np.isfinite(threshold) or not 0 <= threshold <= 1: raise ValueError("Cutoff must be between zero and one")
    data = read_bundle(args.input, config)
    device = choose_device(args.device)
    model, checkpoint_hash = load_model(config, args.checkpoint, device, demo=args.demo)
    probabilities = predict(model, data["x"], device, args.batch_size)
    binary = (probabilities > np.float32(threshold)).astype(np.uint8)
    report = {"model": config["name"], "sample_kind": str(data["sample_kind"].item()),
              "untrained_synthetic_demo": args.demo, "trained_checkpoint_loaded": args.checkpoint is not None,
              "checkpoint_sha256": checkpoint_hash, "windows": len(probabilities),
              "input_shape": list(data["x"].shape), "output_shape": list(probabilities.shape),
              "threshold": threshold, "operator": ">", "threshold_fitting": False,
              "device": str(device), "python": platform.python_version(), "torch": str(torch.__version__),
              "numpy": str(np.__version__), "normalization_id": config["normalization_id"]}
    if "y" in data:
        report["scores"] = occurrence_metrics(probabilities, data["y"], threshold)
        report["score_scope"] = "Synthetic workflow check only" if args.demo else "Only the supplied sample; not the full paper test set"
    args.output.mkdir(parents=True, exist_ok=False)
    payload = {"probability": probabilities, "binary": binary, "latitude": data["latitude"],
               "longitude": data["longitude"], "target_times": data["target_times"],
               "threshold": np.float32(threshold), "operator": np.array(">"),
               "sample_kind": data["sample_kind"], "untrained_demo": np.bool_(args.demo)}
    if "y" in data: payload["target"] = data["y"]
    np.savez_compressed(args.output / "prediction.npz", **payload)
    save_json(args.output / "run.json", report)
    print("UNTRAINED SYNTHETIC DEMO — not a lightning forecast" if args.demo else "TRAINED CHECKPOINT INFERENCE COMPLETE")
    print("Prediction:", args.output / "prediction.npz")
    print("Run details:", args.output / "run.json")
    return report


if __name__ == "__main__":
    try: main()
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as error:
        print("ERROR:", error, file=sys.stderr)
        raise SystemExit(2)
