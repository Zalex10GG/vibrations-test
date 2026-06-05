"""Mesh-convergence study for the modal analysis.

Reads the convergence table from ``data/convergence/convergence.csv``,
computes relative errors between consecutive refinements and against the
finest mesh, writes the augmented CSV, produces the three SVG/PNG plots
and copies the SVGs into ``latex/figures/`` for the report.

Run from the project root with::

    uv run python scripts/convergence.py
"""
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

DATA = Path("data/convergence")
RESULTS = Path("results/convergence")
LATEX_FIG = Path("latex/figures")

sns.set_theme(style="whitegrid")
PALETTE = sns.color_palette("viridis", n_colors=3)


def main() -> None:
    df = pd.read_csv(DATA / "convergence.csv")

    df["inv_mesh"] = 1.0 / df["mesh_size"]
    df = df.sort_values("inv_mesh")

    df["error_rel_measures"] = 100.0 * df["frequency_Hz"].diff().abs() / df["frequency_Hz"]
    df["error_rel_min"] = 100.0 * (df["frequency_Hz"] - df["frequency_Hz"].iloc[-1]).abs() / df["frequency_Hz"].iloc[-1]

    png_dir = RESULTS / "png"
    svg_dir = RESULTS / "svg"
    png_dir.mkdir(parents=True, exist_ok=True)
    svg_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(RESULTS / "convergence_results.csv", index=False)

    fig1, ax1 = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=df, x="inv_mesh", y="frequency_Hz", ax=ax1, color=PALETTE[0], s=80, zorder=5)
    sns.lineplot(data=df, x="inv_mesh", y="frequency_Hz", ax=ax1, color=PALETTE[0], alpha=0.5)
    ax1.set_xlabel("1 / Mesh size [1/m]")
    ax1.set_ylabel("Frequency [Hz]")
    ax1.set_title("Frequency vs 1/mesh size")
    fig1.savefig(png_dir / "frequency.png", dpi=150, bbox_inches="tight")
    fig1.savefig(svg_dir / "frequency.svg", bbox_inches="tight")
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=df, x="inv_mesh", y="error_rel_measures", ax=ax2, color=PALETTE[1], s=80, zorder=5)
    sns.lineplot(data=df, x="inv_mesh", y="error_rel_measures", ax=ax2, color=PALETTE[1], alpha=0.5)
    ax2.set_xlabel("1 / Mesh size [1/m]")
    ax2.set_ylabel("Relative error between measures [%]")
    ax2.set_title("Error between consecutive measures vs 1/mesh size")
    fig2.savefig(png_dir / "error_measures.png", dpi=150, bbox_inches="tight")
    fig2.savefig(svg_dir / "error_measures.svg", bbox_inches="tight")
    plt.close(fig2)

    fig3, ax3 = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=df, x="inv_mesh", y="error_rel_min", ax=ax3, color=PALETTE[2], s=80, zorder=5)
    sns.lineplot(data=df, x="inv_mesh", y="error_rel_min", ax=ax3, color=PALETTE[2], alpha=0.5)
    ax3.set_xlabel("1 / Mesh size [1/m]")
    ax3.set_ylabel("Total relative error [%]")
    ax3.set_title("Total relative error vs 1/mesh size")
    fig3.savefig(png_dir / "error_total.png", dpi=150, bbox_inches="tight")
    fig3.savefig(svg_dir / "error_total.svg", bbox_inches="tight")
    plt.close(fig3)

    LATEX_FIG.mkdir(parents=True, exist_ok=True)
    for svg_file in ["frequency.svg", "error_measures.svg", "error_total.svg"]:
        shutil.copy2(svg_dir / svg_file, LATEX_FIG / svg_file)

    print("Done. CSV and SVGs generated, SVGs copied to latex/figures/.")


if __name__ == "__main__":
    main()
