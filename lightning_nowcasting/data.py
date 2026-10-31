"""Validated, pickle-free input bundles and explicit preprocessing."""
import json
from pathlib import Path

import numpy as np

VARIABLES = ("cape", "cp", "clwc", "r", "q", "t2m", "d2m", "w", "lightning")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_bundle(path, config, *, require_target=False):
    """Return standardized x as N,C,T,H,W; never silently standardize it twice."""
    with np.load(path, allow_pickle=False) as archive:
        values = {key: archive[key] for key in archive.files}
    required = {"x", "channel_names", "input_times", "target_times", "latitude", "longitude",
                "normalization_id", "sample_kind"}
    if required - values.keys():
        raise ValueError("Missing NPZ fields: " + ", ".join(sorted(required - values.keys())))
    x = values["x"]
    h, w = config["grid"]
    if x.ndim != 5 or x.shape[1:] != (9, 3, h, w) or x.shape[0] < 1 or x.dtype != np.float32:
        raise ValueError(f"Expected float32 x with shape (N,9,3,{h},{w})")
    if not np.isfinite(x).all():
        raise ValueError("Input contains nonfinite values; preprocessing must handle these explicitly")
    if tuple(values["channel_names"].tolist()) != VARIABLES:
        raise ValueError("Channel order differs from the trained model")
    if str(values["normalization_id"].item()) != config["normalization_id"]:
        raise ValueError("Normalization identity does not match the model configuration")
    kind = str(values["sample_kind"].item())
    if kind not in {"synthetic", "observed"} or (kind == "synthetic") != bool(config["synthetic_demo"]):
        raise ValueError("Synthetic and observed workflows cannot be mixed")
    if not np.isin(x[:, -1], [0, 1]).all():
        raise ValueError("Lightning history must contain binary occurrence labels")
    for key, size in [("latitude", h), ("longitude", w)]:
        a = values[key]
        if a.shape != (size,) or not np.isfinite(a).all():
            raise ValueError("Invalid " + key + " coordinate")
    if not (np.diff(values["latitude"]) < 0).all() or not (np.diff(values["longitude"]) > 0).all():
        raise ValueError("Latitude must descend north to south and longitude ascend west to east")
    for key, size in [("latitude", h), ("longitude", w)]:
        if key + "_bounds" in config:
            expected = np.linspace(*config[key + "_bounds"], size)
            if not np.allclose(values[key], expected, rtol=0, atol=1e-6):
                raise ValueError("Coordinates do not match the configured " + key + " grid")
    times, targets = values["input_times"], values["target_times"]
    if times.dtype.kind != "M" or targets.dtype.kind != "M":
        raise ValueError("Timestamps must be numpy datetime64 arrays")
    if times.shape != (len(x), 3) or targets.shape != (len(x),) or np.isnat(times).any() or np.isnat(targets).any():
        raise ValueError("Invalid timestamp dimensions or missing timestamps")
    # New sample workflows require actual hourly histories. Historical gap-containing
    # full-cache benchmark scores are documented separately, not silently reproduced here.
    if not (np.diff(times, axis=1) == np.timedelta64(1, "h")).all():
        raise ValueError("Each input history must contain three consecutive hours")
    if not (targets - times[:, -1] == np.timedelta64(1, "h")).all():
        raise ValueError("Each target must be exactly one hour after its final input")
    if len(np.unique(targets)) != len(targets):
        raise ValueError("Duplicate target hours")
    if "y" in values:
        y = values["y"]
        if y.shape != (len(x), 1, h, w) or not np.isin(y, [0, 1]).all():
            raise ValueError("Target must have shape N,1,H,W and binary values")
    elif require_target:
        raise ValueError("This operation requires binary targets y")
    return values


def normalize_weather(raw, mean, scale, *, missing_policy="error"):
    """Convert N,8,3,H,W physical weather values using supplied frozen statistics."""
    raw = np.asarray(raw, dtype=np.float32)
    mean, scale = np.asarray(mean, np.float32), np.asarray(scale, np.float32)
    if raw.ndim != 5 or raw.shape[1:3] != (8, 3) or mean.shape != (8,) or scale.shape != (8,):
        raise ValueError("Expected N,8,3,H,W weather and eight means/scales")
    if not np.isfinite(mean).all() or not np.isfinite(scale).all() or not (scale > 0).all():
        raise ValueError("Invalid normalization statistics")
    valid = np.isfinite(raw) & (np.abs(raw) < 1e25)
    if not valid.all():
        if missing_policy != "legacy_raw_zero":
            raise ValueError("Missing weather; use an explicitly documented missing-value policy")
        raw = np.where(valid, raw, np.float32(0))
    if missing_policy not in {"error", "legacy_raw_zero"}:
        raise ValueError("Unknown missing-value policy")
    return ((raw - mean[None, :, None, None, None]) / scale[None, :, None, None, None]).astype(np.float32)
