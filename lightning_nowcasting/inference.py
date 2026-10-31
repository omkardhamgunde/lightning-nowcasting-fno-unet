"""Strict checkpoint loading and batched CPU/CUDA prediction."""
import hashlib
from pathlib import Path
import numpy as np
import torch

from .model import FNO2DLightning


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_model(config):
    expected = len(config["input_variables"]) * config["history_hours"] * 2 + 2
    if config["projection_input_channels"] != expected or len(config["input_variables"]) != 9 or config["history_hours"] != 3:
        raise ValueError("Model configuration has inconsistent channel dimensions")
    return FNO2DLightning(in_channels=expected, width=config["width"], modes=config["modes"], unet_base=config["unet_base"])


def choose_device(name):
    if name == "auto": name = "cuda" if torch.cuda.is_available() else "cpu"
    if name not in {"cpu", "cuda"}: raise ValueError("Device must be cpu, cuda or auto")
    if name == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA requested but unavailable")
    return torch.device(name)


def load_model(config, checkpoint, device, *, demo=False):
    if demo:
        if not config["synthetic_demo"] or checkpoint is not None:
            raise ValueError("The untrained demo requires demo.json and no checkpoint")
        torch.manual_seed(20261031)
    elif checkpoint is None:
        raise ValueError("A trained checkpoint is required; use --demo only for the synthetic workflow")
    digest = sha256(checkpoint) if checkpoint else None
    if checkpoint and config.get("checkpoint_sha256") and digest != config["checkpoint_sha256"]:
        raise ValueError("Checkpoint SHA256 differs from the published model identity")
    model = make_model(config)
    if checkpoint:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if not isinstance(state, dict) or not state or not all(isinstance(v, torch.Tensor) for v in state.values()):
            raise ValueError("Expected a plain tensor state_dict checkpoint")
        model.load_state_dict(state, strict=True)
    return model.to(device).eval(), digest


@torch.inference_mode()
def predict(model, x, device, batch_size=1):
    if batch_size < 1: raise ValueError("Batch size must be positive")
    parts = []
    for start in range(0, len(x), batch_size):
        batch = torch.from_numpy(np.ascontiguousarray(x[start:start + batch_size])).to(device)
        probability = model(batch).sigmoid().cpu().numpy()
        if not np.isfinite(probability).all(): raise ValueError("Nonfinite model output")
        parts.append(probability.astype(np.float32))
    return np.concatenate(parts)
