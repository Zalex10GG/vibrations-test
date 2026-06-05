"""Post-processing of the harmonic response CSV exports from ANSYS.

Reads the displacement/stress tables produced by the Harmonic Response
analysis (one per probe and per damping ratio) and generates the
comparison plots used in Section 8 of the report.

Expected files in ``data/harmonic/``::

    nose_deformation_1percent.txt   nose_deformation_5percent.txt   nose_deformation_10percent.txt
    fin_deformation_1percent.txt    fin_deformation_5percent.txt    fin_deformation_10percent.txt
    root_stress_1percent.txt        root_stress_5percent.txt        root_stress_10percent.txt

Run from the project root with::

    uv run python scripts/harmonic.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA = Path("data/harmonic")
OUT = Path("results/harmonic")
OUT.mkdir(parents=True, exist_ok=True)

DAMPING = [1, 5, 10]
DAMPING_LABELS = {1: r"$\zeta = 1\%$", 5: r"$\zeta = 5\%$", 10: r"$\zeta = 10\%$"}

PROBES = {
    "nose": {
        "prefix": "nose_deformation",
        "col": "disp_m",
        "ylabel": "Displacement at nose, $u_Y$ [m]",
        "title": "Nose tip displacement vs frequency",
        "tag": "nose_displacement",
        "peak_unit": "mm",
    },
    "fin": {
        "prefix": "fin_deformation",
        "col": "disp_m",
        "ylabel": "Displacement at fin tip, $u_Y$ [m]",
        "title": "Fin tip displacement vs frequency",
        "tag": "fin_displacement",
        "peak_unit": "um",
    },
    "stress": {
        "prefix": "root_stress",
        "col": "stress_Pa",
        "ylabel": "von Mises stress at fin root, $\\sigma_{vM}$ [Pa]",
        "title": "Fin root von Mises stress vs frequency",
        "tag": "fin_root_stress",
        "peak_unit": "MPa",
    },
}

NATURAL_FREQS = [97.43, 98.47, 222.58, 223.12, 401.91, 403.08,
                 565.06, 575.60, 758.44, 759.71, 837.33]


def load_probe(prefix: str, col_name: str, damping_pct: int) -> pd.DataFrame:
    """Read one ANSYS harmonic export and return a tidy DataFrame.

    ANSYS exports prepend a row-index column and append an empty column
    (both surrounded by tabs), use a comma as decimal separator, and mix
    plain decimals with scientific notation (e.g. ``3,3903e+005``).
    """
    path = DATA / f"{prefix}_{damping_pct}percent.txt"
    raw = pd.read_csv(path, sep="\t", header=0, engine="python")
    df = raw.iloc[:, 1:4]
    df.columns = ["Frequency_Hz", col_name, "Phase_deg"]
    for c in df.columns:
        df[c] = pd.to_numeric(
            df[c].astype(str).str.replace(",", ".", regex=False),
            errors="coerce",
        )
    return df


def find_peaks(df: pd.DataFrame, col: str, window: int = 3):
    """Return the list of (frequency_Hz, amplitude) of local maxima."""
    f = df["Frequency_Hz"].to_numpy()
    a = df[col].to_numpy()
    peaks = []
    for i in range(window, len(a) - window):
        local = a[i - window : i + window + 1]
        if a[i] == local.max() and a[i] > local.min():
            peaks.append((f[i], a[i]))
    return peaks


def plot_probe(probe_key: str) -> dict:
    """Build the comparison figure for one probe across the 3 damping runs."""
    info = PROBES[probe_key]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    summary_rows = []
    for pct in DAMPING:
        df = load_probe(info["prefix"], info["col"], pct)
        ax.loglog(df["Frequency_Hz"], df[info["col"]],
                  label=DAMPING_LABELS[pct], lw=1.2)
        peaks = sorted(find_peaks(df, info["col"]), key=lambda x: -x[1])
        for f_peak, a_peak in peaks[:2]:
            summary_rows.append({
                "probe": probe_key,
                "damping_pct": pct,
                "peak_freq_Hz": float(f_peak),
                "peak_value": float(a_peak),
            })

    for f in NATURAL_FREQS:
        ax.axvline(f, color="grey", ls=":", lw=0.5, alpha=0.7)

    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel(info["ylabel"])
    ax.set_title(info["title"])
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()

    tag = info["tag"]
    fig.savefig(OUT / f"{tag}.svg")
    fig.savefig(OUT / f"{tag}.png", dpi=160)
    plt.close(fig)
    return {"probe": probe_key, "figure": str(OUT / f"{tag}.svg"),
            "peaks": summary_rows}


# Two main resonance peaks of interest (from the modal analysis).
PRIMARY_PEAK_HZ = 97.43
SECONDARY_PEAK_HZ = 565.06


def peak_near(df: pd.DataFrame, col: str, target_hz: float, tol_hz: float = 6.0):
    """Return the maximum amplitude of ``col`` within ±tol_hz of target_hz."""
    mask = (df["Frequency_Hz"] >= target_hz - tol_hz) & (
        df["Frequency_Hz"] <= target_hz + tol_hz
    )
    if not mask.any():
        return float("nan")
    idx = df.loc[mask, col].idxmax()
    return float(df.loc[idx, col])


def plot_damping_reduction() -> str:
    """Peak amplitude vs damping ratio, with theoretical 1/zeta overlay.

    Three subplots (one per probe) on log-log axes. Two measured peaks per
    probe (97 Hz and 565 Hz) and a dashed 1/zeta reference normalised at
    zeta=1% show how closely the multi-DOF response follows the SDOF
    prediction A_peak = A_0 / zeta.
    """
    fig, axes = plt.subplots(3, 1, figsize=(7.5, 9), sharex=True)
    damping_pct = np.array(DAMPING, dtype=float)
    zeta = damping_pct / 100.0

    for ax, (key, info) in zip(axes, PROBES.items()):
        peaks_primary, peaks_secondary = [], []
        for pct in DAMPING:
            df = load_probe(info["prefix"], info["col"], pct)
            peaks_primary.append(peak_near(df, info["col"], PRIMARY_PEAK_HZ))
            peaks_secondary.append(peak_near(df, info["col"], SECONDARY_PEAK_HZ))

        peaks_primary = np.array(peaks_primary)
        peaks_secondary = np.array(peaks_secondary)

        ax.loglog(damping_pct, peaks_primary, "o-",
                  label=f"Mode 1 ($\\approx {PRIMARY_PEAK_HZ}\\ Hz$)", lw=1.4)
        ax.loglog(damping_pct, peaks_secondary, "s-",
                  label=f"Mode 7 ($\\approx {SECONDARY_PEAK_HZ}\\ Hz$)", lw=1.4)

        # Theoretical 1/zeta reference, normalised to the zeta=1% data point.
        zeta_fine = np.linspace(0.5, 20, 200)
        ref_primary = peaks_primary[0] * (zeta[0] * 100.0) / zeta_fine
        ref_secondary = peaks_secondary[0] * (zeta[0] * 100.0) / zeta_fine
        ax.loglog(zeta_fine, ref_primary, "--", color="C0", alpha=0.5,
                  label="$1/\\zeta$ ref")
        ax.loglog(zeta_fine, ref_secondary, "--", color="C1", alpha=0.5)

        ax.set_ylabel(info["ylabel"])
        ax.set_title(info["title"].replace(" vs frequency", ""))
        ax.legend(loc="upper right", fontsize=8, frameon=True)
        ax.grid(True, which="both", alpha=0.3)

    axes[-1].set_xlabel("Damping ratio $\\zeta$ [%]")
    fig.suptitle("Peak amplitude vs damping ratio (log-log)", y=1.0, fontsize=12)
    fig.tight_layout()

    out = OUT / "peak_vs_damping.svg"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(OUT / "peak_vs_damping.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return str(out)


def _format_peak(value: float, unit: str) -> str:
    """Compact engineering label for a peak value, e.g. '1.89 mm' or '13.7 MPa'."""
    if unit == "mm":
        return f"{value * 1e3:.3g} mm"
    if unit == "um":
        return f"{value * 1e6:.3g} $\\mu$m"
    if unit == "MPa":
        return f"{value / 1e6:.3g} MPa"
    if unit == "kPa":
        return f"{value / 1e3:.3g} kPa"
    return f"{value:.3g} {unit}"


def plot_peak_bars() -> str:
    """Grouped bar chart of peak amplitude per damping, per resonance mode.

    Three subplots (one per probe). X-axis: damping ratio. Two bars per
    damping value: Mode 1 (~97 Hz) and Mode 7 (~565 Hz). Log y-axis to
    keep both modes visible on the same plot. Numeric value annotated
    above each bar.
    """
    bar_colors = {"primary": "#1f77b4", "secondary": "#ff7f0e"}
    damping_labels = [f"{p}\\%" for p in DAMPING]

    fig, axes = plt.subplots(3, 1, figsize=(8.5, 10), sharex=True)
    x = np.arange(len(DAMPING))
    width = 0.36

    for ax, (key, info) in zip(axes, PROBES.items()):
        peaks_primary, peaks_secondary = [], []
        for pct in DAMPING:
            df = load_probe(info["prefix"], info["col"], pct)
            peaks_primary.append(peak_near(df, info["col"], PRIMARY_PEAK_HZ))
            peaks_secondary.append(peak_near(df, info["col"], SECONDARY_PEAK_HZ))

        peaks_primary = np.array(peaks_primary)
        peaks_secondary = np.array(peaks_secondary)

        unit = info["peak_unit"]
        b1 = ax.bar(x - width / 2, peaks_primary, width,
                    color=bar_colors["primary"],
                    label=f"Mode 1 ($f_1 \\approx {PRIMARY_PEAK_HZ}\\,$Hz)")
        b2 = ax.bar(x + width / 2, peaks_secondary, width,
                    color=bar_colors["secondary"],
                    label=f"Mode 7 ($f_7 \\approx {SECONDARY_PEAK_HZ}\\,$Hz)")

        for bars, vals in [(b1, peaks_primary), (b2, peaks_secondary)]:
            for bar, v in zip(bars, vals):
                if not np.isfinite(v) or v <= 0:
                    continue
                ax.text(bar.get_x() + bar.get_width() / 2,
                        v * 1.15,
                        _format_peak(v, unit),
                        ha="center", va="bottom", fontsize=8)

        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([f"$\\zeta = {l}$" for l in damping_labels])
        ax.set_ylabel(info["ylabel"])
        ax.set_title(info["title"].replace(" vs frequency", ""))
        ax.legend(loc="upper right", fontsize=8, frameon=True)
        ax.grid(True, which="both", axis="y", alpha=0.3)

    axes[-1].set_xlabel("Damping ratio")
    fig.suptitle("Resonance peak amplitude per damping ratio", y=1.0, fontsize=12)
    fig.tight_layout()

    out = OUT / "peak_bars.svg"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(OUT / "peak_bars.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return str(out)


def main() -> None:
    print(f"Output directory: {OUT.resolve()}")
    all_peaks = []
    for key in PROBES:
        result = plot_probe(key)
        print(f"  [ok] {result['figure']}")
        all_peaks.extend(result["peaks"])

    summary = pd.DataFrame(all_peaks)
    summary_path = OUT / "peaks_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\nPeak summary written to {summary_path}\n")
    print(summary.to_string(index=False))

    damping_fig = plot_damping_reduction()
    print(f"\n  [ok] {damping_fig}")

    bars_fig = plot_peak_bars()
    print(f"  [ok] {bars_fig}")


if __name__ == "__main__":
    main()
