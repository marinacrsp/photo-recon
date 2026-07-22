#!/usr/bin/env python3
"""
Volume correlation analysis across segmentation methods with per-subject reference volumes.

Loads subjects from separate method and reference directories:
  - ref_dir/subject/seg_stats.txt                           (ground truth, no distance variants)
  - photo_recon_dir/subject/seg_stats_{4mm,8mm,12mm}.txt
  - tricubic_dir/subject/seg_stats_{4mm,8mm,12mm}.txt
  - imputed_dir/subject/seg_stats_{4mm,8mm,12mm}.txt

Left and right hemisphere labels are collapsed to a single bilateral region by
averaging their volumes (per subject, method, and distance) before analysis.

The figure plots signed relative error (%) vs. log10(reference volume). Each point is
one (subject, bilateral label, distance) observation; the Y-axis is shared across the
three method subplots to enable direct visual comparison.

Outputs:
    * volume_correlations_figure.pdf / .svg
    * volume_correlations_stats.csv
    * volume_correlations_stats.tex

Usage:
    python build_volume_correlations.py \\
        --ref-dir /path/to/reference \\
        --photo-recon-dir /path/to/Photo-recon \\
        --tricubic-dir /path/to/Tricubic \\
        --imputed-dir /path/to/Imputed \\
        --out-dir /path/to/output
"""

from __future__ import annotations

import os
import re
import argparse
from pathlib import Path
from matplotlib.patches import Patch
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
METHOD_COLORS = {
    "Photo-recon": "#BCA8A2",
    "Tricubic":    "#7079CF",
    "Imputed":     "#C8624C",
}
DISTANCES = ["4mm", "8mm", "12mm"]
EXCLUDE_LABEL_PATTERNS = ["csf", "brainstem", "cerebellum", "ventraldc", "accumbens", "inf", "3rd"]

DISTANCE_COLORS = {
    "4mm":  "#1f77b4",   # blue
    "8mm":  "#ff7f0e",   # orange
    "12mm": "#2ca02c",   # green
}

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


def load_all_subjects() -> tuple:
    """
    Load all subjects from separate method and reference directories.

    Returns:
        df     – Subject, Method, Distance, Label, Volume_mm3  (predictions)
        ref_df – Subject, Label, Volume_mm3                    (per-subject ground truth)
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
        ref_file = os.path.join(REF_DIR, subject, f"seg_stats_{subject}.txt")
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
            for distance in DISTANCES:
                stats_file = os.path.join(subject_path, f"seg_stats_{distance}.txt")
                for _, row in read_segstats(stats_file).iterrows():
                    records.append({
                        "Subject":    subject,
                        "Method":     method,
                        "Distance":   distance,
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
    # prefix: Left-X, Right_X, lh-X, rh X …
    s = re.sub(r"^(left|right|lh|rh)[-_ ]", "", s, flags=re.IGNORECASE)
    # suffix: X Left, X-Right, X_rh …
    s = re.sub(r"[-_ ](left|right|lh|rh)$", "", s, flags=re.IGNORECASE)
    return s.strip("-_ ")


def combine_hemispheres(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse left/right labels by averaging volumes within each
    (Subject, Method, Distance). Groupby guarantees no duplicate labels.
    """
    if df.empty:
        return df
    out = df.copy()
    out["Label"] = out["Label"].map(normalize_label)
    return (
        out.groupby(["Subject", "Method", "Distance", "Label"], as_index=False)
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
# STATISTICS
# =============================================================================
def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return np.nan
    try:
        return float(np.corrcoef(x, y)[0, 1])
    except Exception:
        return np.nan


def label_mean_volume(df: pd.DataFrame, method: str,
                      distance: str, label: str) -> float:
    v = df[(df["Method"]   == method)   &
           (df["Distance"] == distance) &
           (df["Label"]    == label)]["Volume_mm3"]
    return float(v.mean()) if len(v) > 0 else np.nan


def label_std_volume(df: pd.DataFrame, method: str,
                     distance: str, label: str) -> float:
    v = df[(df["Method"]   == method)   &
           (df["Distance"] == distance) &
           (df["Label"]    == label)]["Volume_mm3"]
    return float(v.std()) if len(v) > 1 else np.nan


def label_n_subjects(df: pd.DataFrame, method: str,
                     distance: str, label: str) -> int:
    return int(df[(df["Method"]   == method)   &
                  (df["Distance"] == distance) &
                  (df["Label"]    == label)]["Subject"].nunique())


# =============================================================================
# FIGURE  –  Option 1: signed relative error vs. log10(reference volume)
# =============================================================================
def make_figure(df: pd.DataFrame, ref_df: pd.DataFrame) -> plt.Figure:
    """
    Three subplots (one per method), shared Y-axis.

    X-axis : log10(reference volume [mm³])
    Y-axis : signed relative error (%) = (V_method - V_ref) / V_ref * 100

    Each point is one (subject, bilateral label, distance) observation.
    Points are colored by slab distance. A horizontal line at y = 0 marks
    perfect agreement. The inset reports mean bias (%) and the 95 % limits
    of agreement (mean ± 1.96 SD, Bland-Altman convention).
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

    # Shared Y limits: clip to 5th–95th percentile of the pooled error distribution
    # then widen by 10 % to give breathing room.
    if all_errors:
        lo_pct = np.percentile(all_errors, 2)
        hi_pct = np.percentile(all_errors, 98)
        margin  = (hi_pct - lo_pct) * 0.10
        y_lo    = lo_pct - margin
        y_hi    = hi_pct + margin
    else:
        y_lo, y_hi = -50, 50

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)

    for ax, method in zip(axes, METHODS):
        m = merged_cache[method]

        # Scatter one series per distance.
        for distance in DISTANCES:
            d = m[m["Distance"] == distance]
            if d.empty:
                continue
            ax.scatter(
                np.log10(d["Ref_mm3"].to_numpy()),
                d["RelErr"].to_numpy(),
                s=18, alpha=0.55,
                color=DISTANCE_COLORS.get(distance, "#7f7f7f"),
                edgecolors="none",
                label=distance,
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

        # Per-distance mean bias lines (thin horizontal dashes, same color).
        for distance in DISTANCES:
            d = m[m["Distance"] == distance]
            if len(d) < 2:
                continue
            bias = d["RelErr"].mean()
            ax.axhline(bias, color=DISTANCE_COLORS[distance],
                       linewidth=1.2, linestyle=":", alpha=0.8, zorder=1)

        # Inset statistics: pooled mean bias and 95 % LoA.
        if not m.empty:
            bias_all = m["RelErr"].mean()
            sd_all   = m["RelErr"].std()
            loa_lo   = bias_all - 1.96 * sd_all
            loa_hi   = bias_all + 1.96 * sd_all
            stats_txt = (
                f"Bias = {bias_all:+.1f} %\n"
                f"LoA  [{loa_lo:+.1f}, {loa_hi:+.1f}] %\n"
                # f"n = {len(m)}"
            )
            ax.text(
                0.03, 0.97, stats_txt,
                transform=ax.transAxes,
                fontsize=10, verticalalignment="top", family="monospace",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.85),
            )

        ax.set_ylim(y_lo, y_hi)
        ax.set_xlabel("Reference volume [mm³, log₁₀]", fontsize=12, fontweight="bold")
        ax.set_title(method, fontsize=13, fontweight="bold", pad=6)
        ax.grid(True, alpha=0.25, linestyle=":", which="both")
        ax.legend(fontsize=10, loc="lower right",
                  title="Distance", title_fontsize=10)

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
        for distance in DISTANCES:
            row = {"Label": label, "Distance": distance, "Reference_mm3": ref_vol}
            for method in METHODS:
                row[f"{method}_mean_mm3"] = label_mean_volume(df, method, distance, label)
                row[f"{method}_std_mm3"]  = label_std_volume(df, method, distance, label)
                row[f"{method}_n"]        = label_n_subjects(df, method, distance, label)
            rows.append(row)
    path = os.path.join(OUT_DIR, "volume_correlations_stats.csv")
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    return path

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
# LATEX TABLE
# =============================================================================
def build_latex(df: pd.DataFrame, ref_vols_by_label: dict) -> str:
    L = [
        r"\begin{table*}[h!]",
        r"\centering",
        r"\caption{Mean volume measurements (mm$^3$) across segmentation methods and "
        r"slab distances. Left/right regions are averaged into a single bilateral label. "
        r"Reference volumes are shown for comparison.}",
        r"\label{tab:volume_correlations}",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"\textbf{Label} & \textbf{Distance} & \textbf{Reference} & "
        r"\textbf{Photo-recon} & \textbf{Tricubic} & \textbf{Imputed} \\",
        r"\midrule",
    ]
    for label in sorted(df["Label"].unique()):
        ref_vol = ref_vols_by_label.get(label, np.nan)
        ref_str = f"{ref_vol:.1f}" if not np.isnan(ref_vol) else "--"
        for distance in DISTANCES:
            def fmt(method, lbl=label, dist=distance):
                m = label_mean_volume(df, method, dist, lbl)
                return f"{m:.1f}" if not np.isnan(m) else "--"
            L.append(
                f"{label:20s} & {distance} & {ref_str} & "
                f"{fmt('Photo-recon')} & {fmt('Tricubic')} & {fmt('Imputed')} \\\\"
            )
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(L)

# =============================================================================
# SHARED MERGE / HELPERS FOR THE FIGURES
# =============================================================================
def merge_with_ref(df, ref_df):
    """
    Merge predictions with per-subject reference volumes and derive:
        Diff    = V_method - V_ref                (mm^3, signed)
        MeanVol = (V_method + V_ref) / 2          (mm^3, Bland-Altman x)
        RelErr  = Diff / V_ref  * 100             (%, vs reference)
        RelBA   = Diff / MeanVol * 100            (%, Bland-Altman)
    """
    ref_long = ref_df.rename(columns={"Volume_mm3": "Ref_mm3"})[
        ["Subject", "Label", "Ref_mm3"]
    ]
    m = df.merge(ref_long, on=["Subject", "Label"], how="inner")
    m = m[(m["Ref_mm3"] > 0) & (m["Volume_mm3"] > 0)].copy()
    m["Diff"]    = m["Volume_mm3"] - m["Ref_mm3"]
    m["MeanVol"] = 0.5 * (m["Volume_mm3"] + m["Ref_mm3"])
    m["RelErr"]  = m["Diff"] / m["Ref_mm3"] * 100.0
    m["RelBA"]   = m["Diff"] / m["MeanVol"] * 100.0
    return m


def ordered_labels(m, ref_vols_by_label):
    """Labels sorted by descending reference volume (largest structures first)."""
    labels = list(m["Label"].unique())
    return sorted(labels, key=lambda lab: ref_vols_by_label.get(lab, 0.0),
                  reverse=True)


def _shared_ylim(values, pad_frac=0.10, fallback=(-50.0, 50.0)):
    """Robust shared y-limits from the 1st-99th percentile, padded."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return fallback
    lo, hi = np.percentile(v, [1, 99])
    pad = (hi - lo) * pad_frac
    return lo - pad, hi + pad


# =============================================================================
# FIGURE 1: per-structure boxplots of signed relative error, method-grouped,
#           one panel per slab distance
# =============================================================================
def fig_boxplot_relerr(df, ref_df, ref_vols_by_label):
    m_all = merge_with_ref(df, ref_df)
    labels = ordered_labels(m_all, ref_vols_by_label)
    n_lab = len(labels)
    y_lo, y_hi = _shared_ylim(m_all["RelErr"].to_numpy())

    w = 0.8 / max(len(METHODS), 1)
    fig, axes = plt.subplots(
        1, len(DISTANCES),
        figsize=(max(14.0, 0.9 * n_lab * len(DISTANCES)), 6.0),
        sharey=True, squeeze=False,
    )
    axes = axes[0]

    for ax, distance in zip(axes, DISTANCES):
        md = m_all[m_all["Distance"] == distance]
        for j, method in enumerate(METHODS):
            offset = (j - (len(METHODS) - 1) / 2.0) * w
            data, pos = [], []
            for i, lab in enumerate(labels):
                arr = md[(md["Method"] == method) &
                         (md["Label"] == lab)]["RelErr"].to_numpy()
                arr = arr[np.isfinite(arr)]
                if arr.size:
                    data.append(arr)
                    pos.append(i + offset)
            if not data:
                continue
            bp = ax.boxplot(data, positions=pos, widths=w * 0.9,
                            patch_artist=True, showfliers=False,
                            medianprops=dict(color="black", linewidth=1.2))
            for box in bp["boxes"]:
                box.set(facecolor=METHOD_COLORS[method], alpha=0.85,
                        edgecolor="black", linewidth=0.6)
            for part in ("whiskers", "caps"):
                for artist in bp[part]:
                    artist.set(linewidth=0.8)

        ax.axhline(0, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
        ax.set_xticks(range(n_lab))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_xlim(-0.6, n_lab - 0.4)
        ax.set_ylim(y_lo, y_hi)
        ax.set_title(f"Slab distance: {distance}", fontsize=12, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.25, linestyle=":")

    axes[0].set_ylabel("Signed relative volume error (%)",
                       fontsize=12, fontweight="bold")
    handles = [Patch(facecolor=METHOD_COLORS[m], edgecolor="black", label=m)
               for m in METHODS]
    axes[-1].legend(handles=handles, loc="upper right", fontsize=10,
                    title="Method", title_fontsize=10)
    fig.suptitle(
        "Signed relative volume error by structure and method, per slab distance",
        fontsize=13, y=1.02,
    )
    plt.tight_layout()
    return fig


# =============================================================================
# FIGURE 2: relative Bland-Altman per method
# =============================================================================
def fig_bland_altman(df, ref_df):
    m_all = merge_with_ref(df, ref_df)
    y_lo, y_hi = _shared_ylim(m_all["RelBA"].to_numpy(), pad_frac=0.15)

    fig, axes = plt.subplots(
        1, len(METHODS), figsize=(6.0 * len(METHODS), 5.5),
        sharey=True, sharex=True, squeeze=False,
    )
    axes = axes[0]

    for ax, method in zip(axes, METHODS):
        mm = m_all[m_all["Method"] == method]
        for distance in DISTANCES:
            d = mm[mm["Distance"] == distance]
            if d.empty:
                continue
            ax.scatter(np.log10(d["MeanVol"].to_numpy()), d["RelBA"].to_numpy(),
                       s=16, alpha=0.5,
                       color=DISTANCE_COLORS.get(distance, "#7f7f7f"),
                       edgecolors="none", label=distance, zorder=2)
        for distance in DISTANCES:
            d = mm[mm["Distance"] == distance]
            if len(d) < 2:
                continue
            bias, sd = d["RelBA"].mean(), d["RelBA"].std()
            c = DISTANCE_COLORS.get(distance, "#7f7f7f")
            ax.axhline(bias, color=c, linestyle="-", linewidth=1.3, alpha=0.9, zorder=1)
            ax.axhline(bias + 1.96 * sd, color=c, linestyle=":", linewidth=1.0, alpha=0.7, zorder=1)
            ax.axhline(bias - 1.96 * sd, color=c, linestyle=":", linewidth=1.0, alpha=0.7, zorder=1)

        ax.axhline(0, color="black", linestyle="--", linewidth=1.0, alpha=0.5)
        ax.set_ylim(y_lo, y_hi)
        ax.set_xlabel("Mean of method and reference [mm^3, log10]",
                      fontsize=12, fontweight="bold")
        ax.set_title(method, fontsize=13, fontweight="bold", pad=6)
        ax.grid(True, alpha=0.25, linestyle=":")
        ax.legend(fontsize=9, loc="lower right", title="Distance", title_fontsize=9)

    axes[0].set_ylabel("Relative difference (%): (method - reference) / mean",
                       fontsize=12, fontweight="bold")
    fig.suptitle(
        "Relative Bland-Altman agreement per method (bias and 95% LoA per distance)",
        fontsize=13, y=1.02,
    )
    plt.tight_layout()
    return fig


# =============================================================================
# FIGURE 3: small multiples, error vs structure size (method x distance)
# =============================================================================
def fig_small_multiples(df, ref_df):
    m_all = merge_with_ref(df, ref_df)
    n_rows, n_cols = len(METHODS), len(DISTANCES)
    y_lo, y_hi = _shared_ylim(m_all["RelErr"].to_numpy())

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5.0 * n_cols, 3.8 * n_rows),
        sharex=True, sharey=True, squeeze=False,
    )
    for r, method in enumerate(METHODS):
        for c, distance in enumerate(DISTANCES):
            ax = axes[r][c]
            d = m_all[(m_all["Method"] == method) & (m_all["Distance"] == distance)]
            if not d.empty:
                ax.scatter(np.log10(d["Ref_mm3"].to_numpy()), d["RelErr"].to_numpy(),
                           s=14, alpha=0.5,
                           color=DISTANCE_COLORS.get(distance, "#7f7f7f"),
                           edgecolors="none", zorder=2)
                ax.axhline(d["RelErr"].mean(), color="black",
                           linestyle=":", linewidth=1.0, alpha=0.7, zorder=1)
            ax.axhline(0, color="black", linestyle="--", linewidth=0.9, alpha=0.5)
            ax.set_ylim(y_lo, y_hi)
            ax.grid(True, alpha=0.25, linestyle=":")
            if r == 0:
                ax.set_title(distance, fontsize=12, fontweight="bold")
            if c == 0:
                ax.set_ylabel(f"{method}\nrel. error (%)", fontsize=11, fontweight="bold")
            if r == n_rows - 1:
                ax.set_xlabel("Ref vol [mm^3, log10]", fontsize=11)

    fig.suptitle(
        "Signed relative error vs structure size (rows: method, columns: slab distance)",
        fontsize=13, y=1.01,
    )
    plt.tight_layout()
    return fig


# =============================================================================
# FIGURE 4: pooled mean relative error vs slab distance, per method
# =============================================================================
def fig_bias_vs_distance(df, ref_df):
    m_all = merge_with_ref(df, ref_df)
    x = np.arange(len(DISTANCES), dtype=float)
    offsets = np.linspace(-0.12, 0.12, len(METHODS)) if len(METHODS) > 1 else [0.0]

    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    for j, method in enumerate(METHODS):
        means, cis = [], []
        for distance in DISTANCES:
            d = m_all[(m_all["Method"] == method) &
                      (m_all["Distance"] == distance)]["RelErr"].to_numpy()
            d = d[np.isfinite(d)]
            if d.size >= 2:
                means.append(float(d.mean()))
                cis.append(1.96 * float(d.std(ddof=1)) / np.sqrt(d.size))
            else:
                means.append(np.nan)
                cis.append(np.nan)
        ax.errorbar(x + offsets[j], means, yerr=cis, marker="o", capsize=4,
                    linewidth=1.6, color=METHOD_COLORS[method], label=method)

    ax.axhline(0, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(DISTANCES)
    ax.set_xlabel("Slab distance", fontsize=12, fontweight="bold")
    ax.set_ylabel("Mean signed relative error (%), 95% CI of the mean",
                  fontsize=12, fontweight="bold")
    ax.set_title("Volume error vs slab distance (pooled across structures)",
                 fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.25, linestyle=":")
    ax.legend(title="Method", title_fontsize=10)
    plt.tight_layout()
    return fig

# =============================================================================
# OUTPUTS
# =============================================================================
def save_outputs(df, ref_df, ref_vols_by_label):
    os.makedirs(OUT_DIR, exist_ok=True)
    outputs = []

    def _save(fig, stem):
        svg = os.path.join(OUT_DIR, stem + ".svg")
        pdf = os.path.join(OUT_DIR, stem + ".pdf")
        fig.savefig(svg, bbox_inches="tight", dpi=300)
        fig.savefig(pdf, bbox_inches="tight", dpi=300)
        plt.close(fig)
        outputs.extend([pdf, svg])

    _save(make_figure(df, ref_df),                           "volume_correlations_figure")
    _save(fig_boxplot_relerr(df, ref_df, ref_vols_by_label), "volume_boxplot_relerr")
    _save(fig_bland_altman(df, ref_df),                      "volume_bland_altman")
    _save(fig_small_multiples(df, ref_df),                   "volume_smallmultiples")
    _save(fig_bias_vs_distance(df, ref_df),                  "volume_bias_vs_distance")

    tex = os.path.join(OUT_DIR, "volume_correlations_stats.tex")
    Path(tex).write_text(build_latex(df, ref_vols_by_label) + "\n", encoding="utf-8")
    outputs.append(tex)

    audit = write_audit(df, ref_vols_by_label)
    outputs.append(audit)
    return outputs


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Volume correlation analysis: relative error vs. reference volume."
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
    print("[Excluding extra labels...]")
    df     = drop_excluded_labels(df)
    ref_df = drop_excluded_labels(ref_df)
    ref_vols_by_label = ref_df.groupby("Label")["Volume_mm3"].mean().to_dict()

    print(f"  Prediction rows : {len(df):,}")
    print(f"  Reference rows  : {len(ref_df):,}")
    print(f"  Subjects        : {df['Subject'].nunique()}")
    print(f"  Methods         : {sorted(df['Method'].unique())}")
    print(f"  Distances       : {sorted(df['Distance'].unique())}")
    print(f"  Bilateral labels: {df['Label'].nunique()}")

    print("[Generating outputs...]")
    outputs = save_outputs(df, ref_df, ref_vols_by_label)
    print("Wrote:")
    for f in outputs:
        print(f"  {f}")


if __name__ == "__main__":
    main()