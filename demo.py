#!/usr/bin/env python3
"""One-command demo with synthetic data and reduced, randomly initialized weights."""
from pathlib import Path
import sys
from predict import main

ROOT = Path(__file__).resolve().parent
if __name__ == "__main__":
    main(["--config", str(ROOT / "configs/demo.json"), "--input", str(ROOT / "samples/synthetic_sample.npz"),
          "--output", str(ROOT / "outputs/demo"), "--device", "cpu", "--demo", *sys.argv[1:]])
