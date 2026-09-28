"""Add area-normalized Intra plots to already-saved TARDIS cell/replica results."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_root", type=Path)
    args = parser.parse_args()
    for archive in args.output_root.rglob("*_result.npz"):
        if archive.name not in {"cell_result.npz", "replica_result.npz"}:
            continue
        destination = archive.parent / "intra_area_normalized.png"
        if destination.is_file():
            continue
        with np.load(archive) as arrays:
            if "intra_conditional_density_um_inv" not in arrays:
                continue
            x = arrays["bin_mids_m"] * 1e6
            edges = arrays["edges_m"] * 1e6
            density = arrays["intra_conditional_density_um_inv"]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True, constrained_layout=True)
        for axis in axes:
            axis.plot(x, density, color="#1baf7a", linewidth=3.2)
            axis.axhline(0, color="#777777", linewidth=0.8, linestyle=":")
            axis.grid(axis="y", color="#dddddd", linewidth=0.7)
            axis.spines[["top", "right"]].set_visible(False)
            axis.set_xlabel("Distance (μm)")
        axes[0].set_xlim(edges[0], edges[-1]); axes[0].set_title("Full range")
        axes[0].set_ylabel("Area-normalized Intra density (μm⁻¹)")
        axes[1].set_xlim(0, min(0.5, edges[-1])); axes[1].set_title("Zoom: 0–0.5 μm")
        level = "Replica" if archive.name.startswith("replica") else "Cell"
        fig.suptitle(f"{level} Intra · area normalized")
        fig.savefig(destination, dpi=180, facecolor="white", bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()
