#!/usr/bin/env python3
"""Train corrected FNO–U-Net with plain BCE on explicitly supplied train/validation bundles."""
import argparse
from copy import deepcopy
from pathlib import Path
import random
import socket
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from lightning_nowcasting.data import read_bundle, read_json, save_json
from lightning_nowcasting.inference import choose_device, make_model, sha256

ROOT = Path(__file__).resolve().parent


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--validation", type=Path, required=True)
    p.add_argument("--config", type=Path, default=ROOT / "configs/corrected_bce.json")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=20261013)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = p.parse_args(argv)
    host = socket.gethostname().split(".")[0]
    if host.startswith("iitm-ai-") and host not in {"iitm-ai-01", "iitm-ai-02"}:
        raise RuntimeError("This project's HPC training is restricted to nodes 01 and 02")
    if min(args.epochs, args.patience, args.batch_size, args.threads) < 1: raise ValueError("Settings must be positive")
    if args.output.exists(): raise FileExistsError("Output directory already exists")
    config = read_json(args.config)
    train = read_bundle(args.train, config, require_target=True)
    validation = read_bundle(args.validation, config, require_target=True)
    # New datasets must be partitioned without sharing any observation hour.
    train_hours = np.unique(np.concatenate([train["input_times"].ravel(), train["target_times"]]))
    val_hours = np.unique(np.concatenate([validation["input_times"].ravel(), validation["target_times"]]))
    if np.intersect1d(train_hours, val_hours).size:
        raise ValueError("Training and validation share input/target hours; supply separated chronological bundles")
    if train["target_times"].max() >= validation["target_times"].min():
        raise ValueError("Validation must follow training chronologically")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(args.threads)
    device = choose_device(args.device)
    model = make_model(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=.5, patience=2)
    loaders = [DataLoader(TensorDataset(torch.from_numpy(d["x"]), torch.from_numpy(d["y"].astype(np.float32))),
                          batch_size=args.batch_size, shuffle=False, num_workers=0) for d in [train, validation]]
    args.output.mkdir(parents=True, exist_ok=False)
    best_loss = float("inf"); best_epoch = 0; stalled = 0; history = []
    for epoch in range(1, args.epochs + 1):
        started = time.monotonic(); losses = []
        for training, loader in zip([True, False], loaders):
            model.train(training); total = 0.
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                with torch.set_grad_enabled(training):
                    logits = model(x)
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="mean")
                if not torch.isfinite(loss): raise ValueError("Nonfinite BCE loss")
                if training:
                    optimizer.zero_grad(set_to_none=True); loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                    optimizer.step()
                total += loss.item()
            losses.append(total / len(loader))
        scheduler.step(losses[1])
        if losses[1] < best_loss:
            best_loss = losses[1]; best_epoch = epoch; stalled = 0
            torch.save(model.state_dict(), args.output / "best.pt")
        else: stalled += 1
        history.append({"epoch": epoch, "train_BCE": losses[0], "validation_BCE": losses[1],
                        "best_epoch": best_epoch, "learning_rate": optimizer.param_groups[0]["lr"],
                        "minutes": (time.monotonic() - started) / 60})
        save_json(args.output / "history.json", history)
        print(f"Epoch {epoch}: train BCE {losses[0]:.6f}, validation BCE {losses[1]:.6f}", flush=True)
        if stalled >= args.patience: break
    inference_config = deepcopy(config)
    inference_config.update(name="new_bce_training_run", selected_epoch=best_epoch,
                            checkpoint_sha256=sha256(args.output / "best.pt"))
    save_json(args.output / "model_config.json", inference_config)
    save_json(args.output / "training_summary.json", {"completed_epochs": len(history), "selected_epoch": best_epoch,
              "selected_by": "minimum validation BCE", "seed": args.seed, "test_evaluated": False,
              "loss": "unweighted BCEWithLogitsLoss mean", "device": str(device),
              "torch": str(torch.__version__), "checkpoint_sha256": inference_config["checkpoint_sha256"],
              "sample_kind": str(train["sample_kind"].item()), "historical_training_reproduction": False})
    return inference_config


if __name__ == "__main__": main()
