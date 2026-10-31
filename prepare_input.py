#!/usr/bin/env python3
"""Standardize already aligned physical weather maps into the inference format."""
import argparse
from pathlib import Path
import numpy as np
from lightning_nowcasting.data import normalize_weather, read_bundle, read_json, VARIABLES

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", type=Path, required=True, help="NPZ with weather, lightning_history, coordinates and timestamps")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--normalization", type=Path, default=ROOT / "configs/normalization.json")
    p.add_argument("--config", type=Path, default=ROOT / "configs/corrected_bce.json")
    p.add_argument("--missing-policy", choices=["error", "legacy_raw_zero"], default="error")
    args = p.parse_args()
    if args.output.exists(): raise FileExistsError("Output exists")
    norms, config = read_json(args.normalization), read_json(args.config)
    if norms["normalization_id"] != config["normalization_id"]: raise ValueError("Normalization/config mismatch")
    with np.load(args.raw, allow_pickle=False) as archive: raw = {key: archive[key] for key in archive.files}
    required = {"weather", "lightning_history", "latitude", "longitude", "input_times", "target_times"}
    if required - raw.keys(): raise ValueError("Missing raw fields: " + str(required - raw.keys()))
    for key in ["latitude", "longitude"]:
        if not np.array_equal(raw[key], np.asarray(norms[key])):
            raise ValueError("Physical maps must already be aligned to the published " + key + " grid")
    weather = normalize_weather(raw["weather"], norms["mean"], norms["scale"], missing_policy=args.missing_policy)
    lightning = raw["lightning_history"]
    if lightning.shape != (len(weather), 1, 3, *weather.shape[-2:]) or not np.isin(lightning, [0, 1]).all():
        raise ValueError("Expected N,1,3,H,W binary lightning_history")
    payload = {key: raw[key] for key in ["latitude", "longitude", "input_times", "target_times"]}
    payload.update(x=np.concatenate([weather, lightning.astype(np.float32)], axis=1),
                   channel_names=np.array(VARIABLES), sample_kind=np.array("observed"),
                   normalization_id=np.array(norms["normalization_id"]))
    if "y" in raw: payload["y"] = raw["y"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".pending.npz")
    if temporary.exists(): raise FileExistsError("Pending output already exists")
    np.savez_compressed(temporary, **payload)
    try:
        read_bundle(temporary, config)
        temporary.rename(args.output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    print("Prepared:", args.output)


if __name__ == "__main__": main()
