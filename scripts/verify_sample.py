"""Verify exported real-sample inference without changing model weights or thresholds."""
import argparse
from pathlib import Path
import numpy as np
import torch
from lightning_nowcasting.data import read_bundle, read_json
from lightning_nowcasting.inference import choose_device, load_model, predict, sha256

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--assets", type=Path, required=True, help="Directory containing the six exported files")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = p.parse_args()
    config = read_json(ROOT / "configs/corrected_bce.json")
    manifest = read_json(args.assets / "real_sample_manifest.json")
    if manifest["checkpoint_sha256"] != config["checkpoint_sha256"]:
        raise ValueError("Manifest checkpoint is not the published model")
    for name, digest in manifest["asset_sha256"].items():
        if Path(name).name != name: raise ValueError("Invalid asset filename")
        if sha256(args.assets / name) != digest: raise ValueError("Asset checksum differs: " + name)
    data = read_bundle(args.assets / "real_sample.npz", config, require_target=True)
    device = choose_device(args.device); torch.set_num_threads(4)
    model, _ = load_model(config, args.assets / "corrected_bce_epoch3.pt", device)
    actual = predict(model, data["x"], device)
    with np.load(args.assets / "expected_prediction.npz", allow_pickle=False) as expected:
        reference = expected["probability"]
        if actual.shape != reference.shape or not np.allclose(actual, reference, rtol=1e-4, atol=2e-5):
            raise ValueError("Sample probabilities differ from the exported checkpoint inference")
        differences = int(((actual > np.float32(.3)) != expected["binary"]).sum())
    print("CHECKSUMS AND PROBABILITIES VERIFIED")
    print("Maximum probability error:", float(np.max(np.abs(actual - reference))))
    print("Binary differences at > 0.30:", differences)
    if differences: print("Small platform-dependent differences near the threshold must be inspected before reporting exact binary equality.")


if __name__ == "__main__": main()
