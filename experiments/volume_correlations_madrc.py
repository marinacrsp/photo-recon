#!/usr/bin/env python3
"""
Volume correlation analysis across segmentation methods with per-subject reference
volumes, for a dataset with a SINGLE slab distance (no 4/8/12 mm variants).

Loads subjects from separate method and reference directories (glob-based):
  - ref_dir/subject/ * /seg_stats.txt                 (ground truth)
  - tricubic_dir/subject/seg_stats.txt                (Tricubic)
  - imputed_dir/subject/ * /seg_stats_unet.txt        (Imputed)
  - photo_recon_dir/subject/ * /seg_stats_photo_recon.txt   (Photo-recon)

Left and right hemisphere labels are collapsed to a single bilateral region by
averaging their volumes (per subject and method) before analysis.

The figure plots signed relative error (%) vs. log10(reference volume). Each point is
one (subject, bilateral label) observation; the Y-axis is shared across the three
method subplots to enable direct visual comparison.

Outputs:
    * volume_correlations_figure.pdf / .svg
    * volume_correlations_stats.csv
    * volume_correlations_stats.tex

Usage:
    python build_volume_correlations_singledist.py \\
        --ref-dir /path/to/reference \\
        --photo-recon-dir /path/to/Photo-recon \\
        --tricubic-dir /path/to/Tricubic \\
        --imputed-dir /path/to/Imputed \\
        --out-dir /path/to/output
"""

from __future__ import annotations

import os
import re
import glob
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

# =============================================================================
# CONFIGURATION
# =============================================================================
REF_DIR = None
PHOTO_RECON_DIR = None
TRICUBIC_DIR = None
IMPUTED_DIR = None
OUT_DIR = None

METHODS = ["Photo-recon", "Tricubic", "Imputed"]

# Labels to exclude from the analysis. Matching is case-insensitive against the
# label with separators/spaces removed, so "Brain-Stem", "Brain Stem" and
# "Brainstem" all match "brainstem", and any label containing "CSF" matches "csf".
EXCLUDE_LABEL_PATTERNS = ["csf", "brainstem"]

plt.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
})


# =============================================================================
# DATA LOADING
# =============================================================================
def read_segstats(stats_file: str) -> pd.DataFrame:
    """Read a segmentation statistics file (format: SegId NVoxels Volume_mm3 Label)."""
    rows = []
    try:
        with open(stats_file) as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.strip().split(maxsplit=4)
                if len(parts) != 5:
                    continue
                try:
                    rows.append({
                        "SegId":      int(parts[1]),
                        "NVoxels":    int(parts[2]),
                        "Volume_mm3": float(parts[3]),
                        "Label":      parts[4],
                    })
                except (ValueError, IndexError):
                    continue
    except FileNotFoundError:
        pass
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _first_glob(pattern: str):
    """Return the first matching path for a glob pattern, or None if none match."""
    matches = sorted(glob.glob(pattern))
    return matches[0] if matches else None


def load_all_subjects() -> tuple:
    """
    Load all subjects from separate method and reference directories.

    Returns:
        df     - Subject, Method, Label, Volume_mm3  (predictions)
        ref_df - Subject, Label, Volume_mm3          (per-subject ground truth)
    """
    records, ref_records = [], []

    if not os.path.isdir(REF_DIR):
        raise FileNotFoundError(f"Reference directory not found: {REF_DIR}")

    subjects = sorted(
        d for d in os.listdir(REF_DIR)
        if os.path.isdir(os.path.join(REF_DIR, d))
    )
    print(f"Found {len(subjects)} subject(s)")

    for subject in subjects:
        ref_file = _first_glob(os.path.join(REF_DIR, subject, "*", "seg_stats.txt"))
        if ref_file is None:
            print(f"  [ref] no seg_stats.txt for {subject}, skipping")
            continue
        for _, row in read_segstats(ref_file).iterrows():
            ref_records.append({
                "Subject":    subject,
                "Label":      row["Label"],
                "Volume_mm3": float(row["Volume_mm3"]),
            })

    method_dirs = {
        "Photo-recon": PHOTO_RECON_DIR,
        "Tricubic":    TRICUBIC_DIR,
        "Imputed":     IMPUTED_DIR,
    }
    for method, method_dir in method_dirs.items():
        if not os.path.isdir(method_dir):
            print(f"Warning: {method} directory not found: {method_dir}")
            continue

        for subject in subjects:
            subject_path = os.path.join(method_dir, subject)
            if not os.path.isdir(subject_path):
                continue

            if method == "Tricubic":
                pattern = os.path.join(subject_path, "seg_stats.txt")
            elif method == "Imputed":
                pattern = os.path.join(subject_path, "*", "seg_stats_unet.txt")
            else:  # Photo-recon
                pattern = os.path.join(subject_path, "*", "seg_stats_photo_recon.txt")

            stats_file = _first_glob(pattern)
            if stats_file is None:
                print(f"  [{method}] no stats for {subject}, skipping")
                continue

            for _, row in read_segstats(stats_file).iterrows():
                records.append({
                    "Subject":    subject,
                    "Method":     method,
                    "Label":      row["Label"],
                    "Volume_mm3": float(row["Volume_mm3"]),
                })

    df     = pd.DataFrame.from_records(records)
    ref_df = pd.DataFrame.from_records(ref_records)
    if df.empty:
        raise RuntimeError("No prediction records loaded.")
    if ref_df.empty:
        raise RuntimeError("No reference volumes loaded.")
    return df, ref_df


# =============================================================================
# HEMISPHERE COMBINING
# =============================================================================
def normalize_label(label: str) -> str:
    """
    Strip hemisphere markers to obtain a bilateral base label.
    Handles prefix/suffix/infix Left/Right/lh/rh in any separator convention.
    Midline structures (no marker) are returned unchanged.
    """
    s = label.strip()
    # infix: ctx-lh-X or wm-rh-X
    s = re.sub(r"(^|[-_ ])(lh|rh)([-_ ])", r"\1", s, flags=re.IGNORECASE)
    # prefix: Left-X, Right_X, lh-X, rh X ...
    s = re.sub(r"^(left|right|lh|rh)[-_ ]", "", s, flags=re.IGNORECASE)
    # suffix: X Left, X-Right, X_rh ...
    s = re.sub(r"[-_ ](left|right|lh|rh)$", "", s, flags=re.IGNORECASE)
    return s.strip("-_ ")


def combine_hemispheres(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse left/right labels by averaging volumes within each
    (Subject, Method). Groupby guarantees no duplicate labels.
    """
    if df.empty:
        return df
    out = df.copy()
    out["Label"] = out["Label"].map(normalize_label)
    return (
        out.groupby(["Subject", "Method", "Label"], as_index=False)
        ["Volume_mm3"].mean()
    )


def combine_reference_hemispheres(ref_df: pd.DataFrame) -> pd.DataFrame:
    """Collapse left/right reference labels by averaging within each Subject."""
    if ref_df.empty:
        return ref_df
    out = ref_df.copy()
    out["Label"] = out["Label"].map(normalize_label)
    return out.groupby(["Subject", "Label"], as_index=False)["Volume_mm3"].mean()


# =============================================================================
# LABEL EXCLUSION
# =============================================================================
def _squash_label(label: str) -> str:
    """Lowercase and remove separators/spaces for robust label matching."""
    return re.sub(r"[^a-z0-9]", "", str(label).lower())


def is_excluded_label(label: str) -> bool:
    s = _squash_label(label)
    return any(pat in s for pat in EXCLUDE_LABEL_PATTERNS)


def drop_excluded_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows whose (normalized) label matches EXCLUDE_LABEL_PATTERNS."""
    if df.empty:
        return df
    keep = ~df["Label"].map(is_excluded_label)
    return df[keep].copy()


# =============================================================================
# STATISTICS
# =============================================================================
def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return np.nan
    try:
        return float(np.corrcoef(x, y)[0, 1])
    except Exception:
        return np.nan


def label_mean_volume(df: pd.DataFrame, method: str, label: str) -> float:
    v = df[(df["Method"] == method) & (df["Label"] == label)]["Volume_mm3"]
    return float(v.mean()) if len(v) > 0 else np.nan


def label_std_volume(df: pd.DataFrame, method: str, label: str) -> float:
    v = df[(df["Method"] == method) & (df["Label"] == label)]["Volume_mm3"]
    return float(v.std()) if len(v) > 1 else np.nan


def label_n_subjects(df: pd.DataFrame, method: str, label: str) -> int:
    return int(df[(df["Method"] == method) &
                  (df["Label"] == label)]["Subject"].nunique())


# =============================================================================
# FIGURE: signed relative error vs. log10(reference volume)
# =============================================================================
def make_figure(df: pd.DataFrame, ref_df: pd.DataFrame) -> plt.Figure:
    """
    Three subplots (one per method), shared Y-axis.

    X-axis : log10(reference volume [mm^3])
    Y-axis : signed relative error (%) = (V_method - V_ref) / V_ref * 100

    Each point is one (subject, bilateral label) observation. A horizontal line
    at y = 0 marks perfect agreement. The inset reports mean bias (%) and the
    95 % limits of agreement (mean +/- 1.96 SD, Bland-Altman convention).
    """
    ref_long = ref_df.rename(columns={"Volume_mm3": "Ref_mm3"})[
        ["Subject", "Label", "Ref_mm3"]
    ]

    # First pass: collect all relative errors across methods to fix a shared Y range.
    all_errors = []
    merged_cache = {}
    for method in METHODS:
        m = df[df["Method"] == method].merge(ref_long, on=["Subject", "Label"], how="inner")
        m = m[(m["Ref_mm3"] > 0) & (m["Volume_mm3"] > 0)].copy()
        m["RelErr"] = (m["Volume_mm3"] - m["Ref_mm3"]) / m["Ref_mm3"] * 100.0
        merged_cache[method] = m
        all_errors.extend(m["RelErr"].tolist())

    if all_errors:
        lo_pct = np.percentile(all_errors, 2)
        hi_pct = np.percentile(all_errors, 98)
        margin = (hi_pct - lo_pct) * 0.10
        y_lo   = lo_pct - margin
        y_hi   = hi_pct + margin
    else:
        y_lo, y_hi = -50, 50

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)

    for ax, method in zip(axes, METHODS):
        m = merged_cache[method]

        # Single-distance: one scatter series per method subplot.
        if not m.empty:
            ax.scatter(
                np.log10(m["Ref_mm3"].to_numpy()),
                m["RelErr"].to_numpy(),
                s=18, alpha=0.55,
                color="#4477AA",
                edgecolors="none",
                label="Observations",
                zorder=2,
            )

        # Horizontal reference line at 0 % error.
        ax.axhline(0, color="black", linewidth=1.0, linestyle="--",
                   alpha=0.6, label="Zero error", zorder=1)

        # X-axis limits.
        if not m.empty:
            x_vals = np.log10(m["Ref_mm3"].to_numpy())
            x_margin = (x_vals.max() - x_vals.min()) * 0.04
            ax.set_xlim(x_vals.min() - x_margin, x_vals.max() + x_margin)

        # Inset statistics: mean bias and 95 % LoA.
        if not m.empty:
            bias_all = m["RelErr"].mean()
            sd_all   = m["RelErr"].std()
            loa_lo   = bias_all - 1.96 * sd_all
            loa_hi   = bias_all + 1.96 * sd_all
            stats_txt = (
                f"Bias = {bias_all:+.1f} %\n"
                f"LoA  [{loa_lo:+.1f}, {loa_hi:+.1f}] %\n"
                f"n = {len(m)}"
            )
            ax.text(
                0.03, 0.97, stats_txt,
                transform=ax.transAxes,
                fontsize=10, verticalalignment="top", family="monospace",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.85),
            )

        ax.set_ylim(y_lo, y_hi)
        ax.set_xlabel("Reference volume [mm^3, log10]", fontsize=12, fontweight="bold")
        ax.set_title(method, fontsize=13, fontweight="bold", pad=6)
        ax.grid(True, alpha=0.25, linestyle=":", which="both")
        # ax.legend(fontsize=10, loc="lower right")

    axes[0].set_ylabel("Relative volume error (%)", fontsize=12, fontweight="bold")

    fig.suptitle(
        "Signed relative volume error vs. reference (bilateral label averages)",
        fontsize=13, y=1.01,
    )
    plt.tight_layout()
    return fig


# =============================================================================
# AUDIT TABLE (CSV)
# =============================================================================
def write_audit(df: pd.DataFrame, ref_vols_by_label: dict) -> str:
    rows = []
    for label in sorted(df["Label"].unique()):
        ref_vol = ref_vols_by_label.get(label, np.nan)
        row = {"Label": label, "Reference_mm3": ref_vol}
        for method in METHODS:
            row[f"{method}_mean_mm3"] = label_mean_volume(df, method, label)
            row[f"{method}_std_mm3"]  = label_std_volume(df, method, label)
            row[f"{method}_n"]        = label_n_subjects(df, method, label)
        rows.append(row)
    path = os.path.join(OUT_DIR, "volume_correlations_stats.csv")
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    return path


# =============================================================================
# LATEX TABLE
# =============================================================================
def build_latex(df: pd.DataFrame, ref_vols_by_label: dict) -> str:
    L = [
        r"\begin{table*}[h!]",
        r"\centering",
        r"\caption{Mean volume measurements (mm$^3$) across segmentation methods. "
        r"Left/right regions are averaged into a single bilateral label. "
        r"Reference volumes are shown for comparison.}",
        r"\label{tab:volume_correlations}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"\textbf{Label} & \textbf{Reference} & "
        r"\textbf{Photo-recon} & \textbf{Tricubic} & \textbf{Imputed} \\",
        r"\midrule",
    ]
    for label in sorted(df["Label"].unique()):
        ref_vol = ref_vols_by_label.get(label, np.nan)
        ref_str = f"{ref_vol:.1f}" if not np.isnan(ref_vol) else "--"

        def fmt(method, lbl=label):
            m = label_mean_volume(df, method, lbl)
            return f"{m:.1f}" if not np.isnan(m) else "--"

        L.append(
            f"{label:20s} & {ref_str} & "
            f"{fmt('Photo-recon')} & {fmt('Tricubic')} & {fmt('Imputed')} \\\\"
        )
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(L)


# =============================================================================
# OUTPUTS
# =============================================================================
def save_outputs(df: pd.DataFrame, ref_df: pd.DataFrame,
                 ref_vols_by_label: dict) -> list:
    os.makedirs(OUT_DIR, exist_ok=True)

    fig = make_figure(df, ref_df)
    svg = os.path.join(OUT_DIR, "volume_correlations_figure.svg")
    pdf = os.path.join(OUT_DIR, "volume_correlations_figure.pdf")
    fig.savefig(svg, bbox_inches="tight", dpi=300)
    fig.savefig(pdf, bbox_inches="tight", dpi=300)
    plt.close(fig)

    tex = os.path.join(OUT_DIR, "volume_correlations_stats.tex")
    Path(tex).write_text(build_latex(df, ref_vols_by_label) + "\n", encoding="utf-8")

    audit = write_audit(df, ref_vols_by_label)
    return [pdf, svg, tex, audit]


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Volume correlation analysis (single distance): relative error vs. reference volume."
    )
    parser.add_argument("--ref-dir",         required=True)
    parser.add_argument("--photo-recon-dir", required=True)
    parser.add_argument("--tricubic-dir",    required=True)
    parser.add_argument("--imputed-dir",     required=True)
    parser.add_argument("--out-dir",         required=True)
    args = parser.parse_args()

    global REF_DIR, PHOTO_RECON_DIR, TRICUBIC_DIR, IMPUTED_DIR, OUT_DIR
    REF_DIR          = args.ref_dir
    PHOTO_RECON_DIR  = args.photo_recon_dir
    TRICUBIC_DIR     = args.tricubic_dir
    IMPUTED_DIR      = args.imputed_dir
    OUT_DIR          = args.out_dir

    print("[Loading data...]")
    df, ref_df = load_all_subjects()

    print("[Combining left/right hemispheres...]")
    df     = combine_hemispheres(df)
    ref_df = combine_reference_hemispheres(ref_df)

    print(f"[Excluding labels matching {EXCLUDE_LABEL_PATTERNS} ...]")
    df     = drop_excluded_labels(df)
    ref_df = drop_excluded_labels(ref_df)

    ref_vols_by_label = ref_df.groupby("Label")["Volume_mm3"].mean().to_dict()

    print(f"  Prediction rows : {len(df):,}")
    print(f"  Reference rows  : {len(ref_df):,}")
    print(f"  Subjects        : {df['Subject'].nunique()}")
    print(f"  Methods         : {sorted(df['Method'].unique())}")
    print(f"  Bilateral labels: {df['Label'].nunique()}")

    print("[Generating outputs...]")
    outputs = save_outputs(df, ref_df, ref_vols_by_label)
    print("Wrote:")
    for f in outputs:
        print(f"  {f}")


if __name__ == "__main__":
    main()