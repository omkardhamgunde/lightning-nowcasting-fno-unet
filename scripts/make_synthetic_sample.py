"""Generate an openly distributable synthetic input; no observed data or weights."""
from pathlib import Path
import argparse
import numpy as np
from lightning_nowcasting.data import VARIABLES


def generate(path, start="2026-01-01T00", count=2):
    rng = np.random.default_rng(20261031)
    h, w = 17, 19
    yy, xx = np.meshgrid(np.linspace(-1, 1, h), np.linspace(-1, 1, w), indexing="ij")
    x = rng.normal(0, .3, (count, 9, 3, h, w)).astype(np.float32)
    y = np.zeros((count, 1, h, w), np.uint8)
    for n in range(count):
        for t in range(4):
            blob = np.exp(-((xx - (-.45 + .1 * (n + t))) ** 2 + (yy - .15) ** 2) / .06)
            if t < 3:
                x[n, :8, t] += blob.astype(np.float32)
                x[n, 8, t] = (blob > .5).astype(np.float32)
            else: y[n, 0] = (blob > .5).astype(np.uint8)
    targets = np.datetime64(start, "ns") + (np.arange(count) + 3) * np.timedelta64(1, "h")
    times = targets[:, None] - np.array([3, 2, 1]) * np.timedelta64(1, "h")
    path = Path(path)
    if path.exists(): raise FileExistsError("Sample already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, x=x, y=y, channel_names=np.array(VARIABLES), input_times=times,
                        target_times=targets, latitude=np.linspace(30, 26, h), longitude=np.linspace(75, 79, w),
                        normalization_id=np.array("synthetic-standardized-v1"), sample_kind=np.array("synthetic"))
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("samples/synthetic_sample.npz"))
    args = parser.parse_args()
    print(generate(args.output))
