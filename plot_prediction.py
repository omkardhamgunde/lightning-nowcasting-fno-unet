#!/usr/bin/env python3
"""Plot a probability colorbar and a separate binary-occurrence legend."""
import argparse
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True, help="prediction.npz produced by predict.py")
    p.add_argument("--output", type=Path, required=True, help="New PNG output path")
    p.add_argument("--index", type=int, default=0)
    args = p.parse_args()
    if args.output.exists(): raise FileExistsError("Output exists")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    with np.load(args.input, allow_pickle=False) as d: values = {k: d[k] for k in d.files}
    i = args.index
    if not 0 <= i < len(values["probability"]): raise ValueError("Invalid sample index")
    panels = 3 if "target" in values else 2
    fig, axes = plt.subplots(1, panels, figsize=(5 * panels, 5), constrained_layout=True)
    dx = abs(float(np.diff(values["longitude"])[0])); dy = abs(float(np.diff(values["latitude"])[0]))
    extent = [values["longitude"][0] - dx/2, values["longitude"][-1] + dx/2,
              values["latitude"][-1] - dy/2, values["latitude"][0] + dy/2]
    image = axes[0].imshow(values["probability"][i, 0], origin="upper", extent=extent, vmin=0, vmax=1, cmap="viridis")
    axes[0].set_title("Lightning occurrence probability")
    fig.colorbar(image, ax=axes[0], label="Probability", shrink=.8)
    binary_cmap = ListedColormap(["#ffffff", "#17324d"])
    axes[1].imshow(values["binary"][i, 0], origin="upper", extent=extent, vmin=0, vmax=1, cmap=binary_cmap)
    axes[1].set_title(f"Binary prediction > {float(values['threshold']):.3f}")
    if panels == 3:
        axes[2].imshow(values["target"][i, 0], origin="upper", extent=extent, vmin=0, vmax=1, cmap=binary_cmap)
        synthetic = str(values.get("sample_kind", np.array("observed")).item()) == "synthetic"
        axes[2].set_title("Synthetic target" if synthetic else "Recorded binary occurrence")
    for ax in axes:
        ax.set_xlabel("Longitude (°E)"); ax.set_ylabel("Latitude (°N)")
    axes[1].legend(handles=[Patch(facecolor="white", edgecolor="black", label="No occurrence (0)"),
                            Patch(facecolor="#17324d", label="Occurrence (1)")], loc="upper right", fontsize=8)
    title = np.datetime_as_string(values["target_times"][i], unit="h")
    if bool(values.get("untrained_demo", False)): title = "Untrained synthetic demo | " + title
    fig.suptitle(title)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180); plt.close(fig)
    print(args.output)


if __name__ == "__main__": main()
