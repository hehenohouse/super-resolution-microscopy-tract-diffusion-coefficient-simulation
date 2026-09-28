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
        full_destination = archive.parent / "intra_area_normalized_full_range.png"
        zoom_destination = archive.parent / "intra_area_normalized_zoom_0p5um.png"
        if full_destination.is_file() and zoom_destination.is_file():
            continue
        with np.load(archive) as arrays:
            if "intra_conditional_density_um_inv" not in arrays:
                continue
            x = arrays["bin_mids_m"] * 1e6
            edges = arrays["edges_m"] * 1e6
            density = arrays["intra_conditional_density_um_inv"]
        for destination, zoom in ((full_destination, False), (zoom_destination, True)):
            fig, axis = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
            axis.plot(x, density, color="#1baf7a", linewidth=3.2)
            axis.axhline(0, color="#777777", linewidth=0.8, linestyle=":")
            axis.grid(axis="y", color="#dddddd", linewidth=0.7)
            axis.spines[["top", "right"]].set_visible(False)
            axis.set_xlabel("Distance (μm)")
            axis.set_xlim(0, min(0.5, edges[-1])) if zoom else axis.set_xlim(edges[0], edges[-1])
            axis.set_title("Zoom: 0–0.5 μm" if zoom else "Full range")
            axis.set_ylabel("Area-normalized Intra density (μm⁻¹)")
            level = "Replica" if archive.name.startswith("replica") else "Cell"
            fig.suptitle(f"{level} Intra · area normalized")
            fig.savefig(destination, dpi=180, facecolor="white", bbox_inches="tight")
            plt.close(fig)


if __name__ == "__main__":
    main()
