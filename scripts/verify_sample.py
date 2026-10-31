"""Verify exported real-sample inference without changing model weights or thresholds."""
import argparse
from pathlib import Path
import numpy as np
import torch
from lightning_nowcasting.data import read_bundle, read_json
from lightning_nowcasting.inference import choose_device, load_model, predict, sha256

ROOT = Path(__file__).resolve().parents[1]


def compare_reference(actual, reference, reference_binary, limits, *, strict=False):
    if actual.shape != reference.shape or reference_binary.shape != reference.shape:
        raise ValueError("Sample/reference dimensions differ")
    if not np.isfinite(actual).all() or not np.isfinite(reference).all():
        raise ValueError("Nonfinite reference comparison")
    if not np.array_equal(reference_binary, (reference > np.float32(.3)).astype(np.uint8)):
        raise ValueError("Reference binary labels do not match their declared cutoff")
    error = np.abs(actual.astype(np.float64) - reference.astype(np.float64))
    maximum, mean = float(error.max()), float(error.mean())
    differences = int(((actual > np.float32(.3)) != reference_binary).sum())
    report = {"maximum_probability_error": maximum, "mean_probability_error": mean,
              "binary_differences_at_0p30": differences,
              "original_strict_allclose": bool(np.allclose(actual, reference, rtol=1e-4, atol=2e-5))}
    if maximum > limits["maximum_absolute_probability_error"] or mean > limits["maximum_mean_probability_error"]:
        raise ValueError("Sample probabilities exceed the documented CPU/GPU tolerance: " + str(report))
    if limits["require_exact_binary_at_0p30"] and differences:
        raise ValueError("Binary sample predictions differ from the reference: " + str(report))
    if strict and not report["original_strict_allclose"]:
        raise ValueError("Strict probability comparison failed: " + str(report))
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--assets", type=Path, required=True, help="Directory containing the six exported files")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--strict-probabilities", action="store_true", help="Also require the original tighter allclose test")
    args = p.parse_args()
    config = read_json(ROOT / "configs/corrected_bce.json")
    manifest = read_json(args.assets / "real_sample_manifest.json")
    if manifest["checkpoint_sha256"] != config["checkpoint_sha256"]:
        raise ValueError("Manifest checkpoint is not the published model")
    for name, digest in manifest["asset_sha256"].items():
        if Path(name).name != name: raise ValueError("Invalid asset filename")
        if sha256(args.assets / name) != digest: raise ValueError("Asset checksum differs: " + name)
    data = read_bundle(args.assets / "real_sample.npz", config, require_target=True)
    limits = read_json(ROOT / "configs/release_assets.json")["verification"]
    device = choose_device(args.device); torch.set_num_threads(4)
    model, _ = load_model(config, args.assets / "corrected_bce_epoch3.pt", device)
    actual = predict(model, data["x"], device)
    with np.load(args.assets / "expected_prediction.npz", allow_pickle=False) as expected:
        reference = expected["probability"]
        if not np.array_equal(expected["target_times"], data["target_times"]) or float(expected["threshold"]) != float(np.float32(.3)):
            raise ValueError("Reference dates or cutoff differ from the published sample")
        comparison = compare_reference(actual, reference, expected["binary"], limits, strict=args.strict_probabilities)
    print("CHECKSUMS AND PROBABILITIES VERIFIED")
    print("Maximum probability error:", comparison["maximum_probability_error"])
    print("Mean probability error:", comparison["mean_probability_error"])
    print("Binary differences at > 0.30:", comparison["binary_differences_at_0p30"])
    print("Documented CPU/GPU probability limits:", limits)
    print("Original tighter allclose check:", comparison["original_strict_allclose"])


if __name__ == "__main__": main()
