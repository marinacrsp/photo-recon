#!/usr/bin/env python3
"""
Joint volume-correlation analysis over BOTH cohorts:

  * UW    - three slab distances (4/8/12 mm), flat layout:
                <dir>/<subject>/seg_stats_{distance}.txt
                reference: <ref_dir>/<subject>/seg_stats_{subject}.txt
  * MADRC - one reconstruction per sample, glob layout:
                ref:   <ref_dir>/<subject>/ * /seg_stats.txt
                tri:   <tri_dir>/<subject>/seg_stats.txt
                imp:   <imp_dir>/<subject>/ * /seg_stats_unet.txt
                photo: <photo_dir>/<subject>/ * /seg_stats_photo_recon.txt

Labels are selected by their original SegId numbering (allowlist LABEL_NAMES);
left/right IDs sharing a name collapse to one bilateral region.

Produces exactly two deliverables (the other diagnostic figures are omitted):

  1. volume_correlations_joint.pdf / .svg
     A 2 x 3 grid: rows = cohort (UW, MADRC), columns = method. Each panel is a
     scatter of signed relative error (%) vs log10(reference volume), shared axes.

  2. volume_error_table_joint.tex  (+ volume_error_stats_joint.csv)
     One table*, sections MADRC / UW-4mm / UW-8mm / UW-12mm; rows = regions;
     columns = per-method normalized error (mean over subjects, in [0,1]) plus
     pairwise Wilcoxon signed-rank p-values (paired, on per-subject absolute
     errors, computed per section and region).

Usage:
    python build_volume_correlations_joint.py \\
        --uw-ref-dir ...    --uw-photo-recon-dir ... \\
        --uw-tricubic-dir ...  --uw-imputed-dir ... \\
        --madrc-ref-dir ... --madrc-photo-recon-dir ... \\
        --madrc-tricubic-dir ...  --madrc-imputed-dir ... \\
        --out-dir /path/to/output
"""

from __future__ import annotations

import os
import re
import glob
import argparse
import itertools
from pathlib import Path
from utils.summary_tables import build_summary_outputs

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from scipy.stats import linregress
from utils.combine_hemispheres import process #as process_madrc
# =============================================================================
# CONFIGURATION
# =============================================================================
OUT_DIR = None

Q_LEVEL=0.05

METHODS = ["Photo-recon", "Tricubic", "Imputed"]
METHOD_DISPLAY = {
    "Photo-recon": "3D reconstruction \n of slab photographs",
    "Tricubic":    "Cubic",
    "Imputed":     "Imputed",
}
METHOD_ABBR = {"Photo-recon": "PR", "Tricubic": "Cubic", "Imputed": "UNet"}

DISTANCES   = ["12mm", "8mm", "4mm"]          # UW slab distances

MADRC_LABEL = "MADRC"                          # Distance tag for the MADRC cohort
SECTION_ORDER = [MADRC_LABEL] + DISTANCES      # table section order (MADRC first)
SECTION_ORDER_TABLE = [MADRC_LABEL] + ["4mm", "8mm", "12mm"] 
SECTION_HEADER = {
    "MADRC": "MADRC", "4mm": "UW -- 4 mm", "8mm": "UW -- 8 mm", "12mm": "UW -- 12 mm",
}
APPLY_BH = False
PANEL_LETTERS = ["(a)", "(b)", "(c)"]

# Normalization scope: per (Distance/section, Label). MADRC is a single section,
# so it is effectively per region there.
NORMALIZE_BY = ["Label"]

# Pairwise comparisons for the p-value columns. Default: all three pairs. To
# report only the imputation-vs-baseline comparisons, set:
#   PVALUE_PAIRS = [("Photo-recon","Imputed"), ("Tricubic","Imputed")]
PVALUE_PAIRS = list(itertools.combinations(METHODS, 2))

# Statistical test: "wilcoxon" (paired signed-rank) or "ranksum" (Mann-Whitney).
TEST = "wilcoxon"
SEG_FAILURE_RATIO = 0.0
DISTANCE_COLORS = {"4mm": "#d2691e" , "8mm": "#e9967a", "12mm": "#ffcba4"}
MADRC_COLOR = "#A894EE"

POINT_ORDER = ["4mm", "8mm", "12mm"]   # exact label strings, in display order
LINE_ORDER  = ["y = x", "LS Fit"]
# --- LABEL SELECTION BY ORIGINAL SegId NUMBERING -----------------------------
LABEL_NAMES = {
    2: "WM", 3: "Cortex", 4: "Ventricle", 10: "Thalamus", 11: "Caudate",
    12: "Putamen", 13: "Pallidum", 17: "Hippocampus", 18: "Amygdala",
    41: "WM", 42: "Cortex", 43: "Ventricle", 49: "Thalamus", 50: "Caudate",
    51: "Putamen", 52: "Pallidum", 53: "Hippocampus", 54: "Amygdala",
}
ALLOWED_SEGIDS = set(LABEL_NAMES)
RENAME_TO_CANONICAL = True

plt.rcParams.update({
    "font.size": 20, "axes.labelsize": 20,
    "xtick.labelsize": 20, "ytick.labelsize": 20, "legend.fontsize": 20,
})

ERROR_CAPTION = (
    r"Region-specific normalized volume error of automated segmentations of 3D "
    r"reconstructions of photographs, for the MADRC and UW datasets."
    r"Gold-standard volumes are obtained from MRI scans."
)

PVALUE_CAPTION = (
    r"Pairwise statistical comparisons of the per-subject absolute volume errors "
    r"between reconstruction methods, for the MADRC and UW datasets, by region and "
    r"slab distance. P-values are from " +
    ("Wilcoxon signed-rank (paired)" if TEST == "wilcoxon"
     else "Wilcoxon rank-sum (Mann-Whitney)") +
    r" tests. Dashes indicate comparisons without paired samples. "
    r"Abbreviations: PR = Photo-recon, Cub = cubic interpolation, "
    r"UNet = U-Net imputation."
)
# =============================================================================
# DATA LOADING
# =============================================================================
def read_segstats(stats_file: str) -> pd.DataFrame:
    """Read a segmentation statistics file (columns: * SegId NVoxels Volume_mm3 Label)."""
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
                    rows.append({"SegId": int(parts[1]), "NVoxels": int(parts[2]),
                                 "Volume_mm3": float(parts[3]), "Label": parts[4]})
                except (ValueError, IndexError):
                    continue
    except FileNotFoundError:
        pass
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _first_glob(pattern: str):
    matches = sorted(glob.glob(pattern))
    return matches[0] if matches else None


def load_uw_subjects(ref_dir, photo_dir, tri_dir, imp_dir) -> tuple:
    """UW: flat layout, three slab distances per subject per method."""
    records, ref_records = [], []
    if not os.path.isdir(ref_dir):
        raise FileNotFoundError(f"UW reference directory not found: {ref_dir}")
    subjects = sorted(d for d in os.listdir(ref_dir)
                      if os.path.isdir(os.path.join(ref_dir, d)))
    print(f"[UW] Found {len(subjects)} subject(s)")

    for subject in subjects:
        ref_file = os.path.join(ref_dir, subject, f"seg_stats_{subject}.txt")
        for _, r in read_segstats(ref_file).iterrows():
            ref_records.append({"Subject": subject, "SegId": int(r["SegId"]),
                                "Label": r["Label"], "Volume_mm3": float(r["Volume_mm3"])})

    method_dirs = {"Photo-recon": photo_dir, "Tricubic": tri_dir, "Imputed": imp_dir}
    for method, mdir in method_dirs.items():
        if not os.path.isdir(mdir):
            print(f"[UW] Warning: {method} directory not found: {mdir}")
            continue
        for subject in subjects:
            spath = os.path.join(mdir, subject)
            if not os.path.isdir(spath):
                continue
            for distance in DISTANCES:
                sf = os.path.join(spath, f"seg_stats_{distance}.txt")
                for _, r in read_segstats(sf).iterrows():
                    records.append({"Subject": subject, "Method": method,
                                    "Distance": distance, "SegId": int(r["SegId"]),
                                    "Label": r["Label"], "Volume_mm3": float(r["Volume_mm3"])})

    df, ref_df = pd.DataFrame.from_records(records), pd.DataFrame.from_records(ref_records)
    if df.empty or ref_df.empty:
        raise RuntimeError("UW: no records loaded; check the UW paths.")
    return df, ref_df


def load_madrc_subjects(ref_dir, photo_dir, tri_dir, imp_dir) -> tuple:
    """MADRC: glob layout, one file per subject per method, no slab distances."""
    records, ref_records = [], []
    if not os.path.isdir(ref_dir):
        raise FileNotFoundError(f"MADRC reference directory not found: {ref_dir}")
    subjects = sorted(d for d in os.listdir(ref_dir)
                      if os.path.isdir(os.path.join(ref_dir, d)))
    print(f"[MADRC] Found {len(subjects)} subject(s)")

    for subject in subjects:
        ref_file = _first_glob(os.path.join(ref_dir, subject, "*", "seg_stats.txt"))
        if ref_file is None:
            print(f"[MADRC] [ref] no seg_stats.txt for {subject}, skipping")
            continue
        for _, r in read_segstats(ref_file).iterrows():
            ref_records.append({"Subject": subject, "SegId": int(r["SegId"]),
                                "Label": r["Label"], "Volume_mm3": float(r["Volume_mm3"])})

    method_dirs = {"Photo-recon": photo_dir, "Tricubic": tri_dir, "Imputed": imp_dir}
    for method, mdir in method_dirs.items():
        if not os.path.isdir(mdir):
            print(f"[MADRC] Warning: {method} directory not found: {mdir}")
            continue
        for subject in subjects:
            spath = os.path.join(mdir, subject)
            if not os.path.isdir(spath):
                continue
            if method == "Tricubic":
                pattern = os.path.join(spath, "seg_stats_reconany_presurf.txt")
            elif method == "Imputed":
                pattern = os.path.join(spath, "seg_stats_imputation_reconany_presurf.txt")
            else:  # Photo-recon
                pattern = os.path.join(spath, "seg_stats_photo_reconany_presurf.txt")
            sf = _first_glob(pattern)
            if sf is None:
                print(f"[MADRC] [{method}] no stats for {subject}, skipping")
                continue
            for _, r in read_segstats(sf).iterrows():
                records.append({"Subject": subject, "Method": method,
                                "Distance": MADRC_LABEL, "SegId": int(r["SegId"]),
                                "Label": r["Label"], "Volume_mm3": float(r["Volume_mm3"])})

    df, ref_df = pd.DataFrame.from_records(records), pd.DataFrame.from_records(ref_records)
    if df.empty or ref_df.empty:
        raise RuntimeError("MADRC: no records loaded; check the MADRC paths.")
    return df, ref_df


# =============================================================================
# FILTERING / HEMISPHERE COMBINING / MERGE
# =============================================================================
def drop_segmentation_failures(m, thresh=SEG_FAILURE_RATIO):
    ratio = m["Volume_mm3"] / m["Ref_mm3"]
    keep = ratio >= thresh
    dropped = (~keep).sum()
    if dropped:
        print(f"[seg-failure filter] dropped {dropped} of {len(m)} rows "
              f"(ratio < {thresh}):")
        print(m.loc[~keep, ["Subject", "Method", "Distance", "Label",
                            "Volume_mm3", "Ref_mm3"]].to_string(index=False))
    return m[keep].copy()

def apply_label_whitelist(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if "SegId" not in df.columns:
        raise KeyError("apply_label_whitelist requires a 'SegId' column.")
    out = df[df["SegId"].isin(ALLOWED_SEGIDS)].copy()
    if RENAME_TO_CANONICAL:
        out["Label"] = out["SegId"].map(LABEL_NAMES)
    return out


def normalize_label(label: str) -> str:
    s = label.strip()
    s = re.sub(r"(^|[-_ ])(lh|rh)([-_ ])", r"\1", s, flags=re.IGNORECASE)
    s = re.sub(r"^(left|right|lh|rh)[-_ ]", "", s, flags=re.IGNORECASE)
    s = re.sub(r"[-_ ](left|right|lh|rh)$", "", s, flags=re.IGNORECASE)
    return s.strip("-_ ")


def combine_hemispheres(df: pd.DataFrame) -> pd.DataFrame:
    """Average left/right within (Subject, Method, Distance)."""
    if df.empty:
        return df
    out = df.copy()
    out["Label"] = out["Label"].map(normalize_label)
    return (out.groupby(["Subject", "Method", "Distance", "Label"], as_index=False)
              ["Volume_mm3"].mean())


def combine_reference_hemispheres(ref_df: pd.DataFrame) -> pd.DataFrame:
    if ref_df.empty:
        return ref_df
    out = ref_df.copy()
    out["Label"] = out["Label"].map(normalize_label)
    return out.groupby(["Subject", "Label"], as_index=False)["Volume_mm3"].mean()


def merge_with_ref(df: pd.DataFrame, ref_df: pd.DataFrame) -> pd.DataFrame:
    """Per-observation frame with Distance, Diff = V_method - V_ref, and Ref_mm3."""
    ref_long = ref_df.rename(columns={"Volume_mm3": "Ref_mm3"})[["Subject", "Label", "Ref_mm3"]]
    m = df.merge(ref_long, on=["Subject", "Label"], how="inner")
    m = m[(m["Ref_mm3"] > 0) & (m["Volume_mm3"] > 0)].copy()
    m["Diff"] = m["Volume_mm3"] - m["Ref_mm3"]
    return m


# def process(df: pd.DataFrame, ref_df: pd.DataFrame, tag: str) -> pd.DataFrame:
#     df = apply_label_whitelist(df)
#     ref_df = apply_label_whitelist(ref_df)
#     if df.empty or ref_df.empty:
#         raise RuntimeError(f"All {tag} rows removed by the SegId allowlist.")
#     df = combine_hemispheres(df)
#     ref_df = combine_reference_hemispheres(ref_df)
#     return merge_with_ref(df, ref_df)

def _pearson(x, y):
    """Pearson correlation coefficient."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    fit = linregress(x, y)

    a = fit.intercept
    b = fit.slope
    r = fit.rvalue
    p = fit.pvalue
    return a, b, r, p

def make_concordance_figure(m_uw: pd.DataFrame, m_mad: pd.DataFrame) -> plt.Figure:
    """
    Per-cohort, per-method concordance of measured vs reference volume on log-log
    axes, with identity line, LS fit, and +/-1.96*residual-SD prediction band.

    For UW, r and SD_r are computed and displayed separately for each distance.
    For MADRC, r and SD_r are computed over the complete cohort.
    """
    rows = [("UW", m_uw, True), ("MADRC", m_mad, False)]

    allv = []
    for _, mm, _ in rows:
        if not mm.empty:
            allv += np.log10(mm["Ref_mm3"]).tolist()
            allv += np.log10(mm["Volume_mm3"]).tolist()

    lo, hi = (min(allv), max(allv)) if allv else (2.0, 5.5)
    pad = (hi - lo) * 0.05
    lo -= pad
    hi += pad

    fig, axes = plt.subplots(
        2, 3,
        figsize=(18, 12),
        sharex=True,
        sharey=True
    )
    known = POINT_ORDER + LINE_ORDER
    seen = {}
    for r, (coh, mm, by_dist) in enumerate(rows):

        for c, method in enumerate(METHODS):
            ax = axes[r, c]

            # ---------------------------------------------------------
            # Identity line
            # ---------------------------------------------------------
            ax.plot(
                [lo, hi], [lo, hi],
                ls="--",
                color="black",
                lw=1.4,
                alpha=0.6,
                zorder=1,
                label="y = x"
            )

            sub = mm[mm["Method"] == method]

            if not sub.empty:

                # =====================================================
                # UW: calculate metrics separately for each distance
                # =====================================================
                if by_dist:

                    for dist in DISTANCES:

                        dd = sub[sub["Distance"] == dist]

                        if dd.empty:
                            continue

                        x = np.log10(dd["Ref_mm3"].to_numpy())
                        y = np.log10(dd["Volume_mm3"].to_numpy())

                        # Scatter points
                        ax.scatter(
                            x,
                            y,
                            s=36,
                            alpha=0.55,
                            color=DISTANCE_COLORS.get(
                                dist,
                                "#7f7f7f"
                            ),
                            edgecolors="none",
                            label=dist,
                            zorder=3
                        )

                        # -------------------------------------------------
                        # Per-distance Pearson correlation
                        # -------------------------------------------------
                        a, b, r_val, p = _pearson(x, y)

                        if np.isfinite(b):

                            xg = np.linspace(lo, hi, 100)
                            yg = a + b * x

                            # Residuals for THIS distance
                            resid = y - (a + b * x)

                            rsd = (
                                np.std(resid, ddof=2)
                                if len(resid) > 2
                                else np.nan
                            )

                            # LS fit for THIS distance
                            ax.plot(
                                xg,
                                a + b * xg,
                                color=DISTANCE_COLORS.get(
                                    dist,
                                    "#7f7f7f"
                                ),
                                lw=1.6,
                                alpha=0.8,
                                zorder=4
                            )

                            # Prediction band for THIS distance
                            if np.isfinite(rsd):
                                ax.fill_between(
                                    xg,
                                    a + b * xg - 1.96 * rsd,
                                    a + b * xg + 1.96 * rsd,
                                    color=DISTANCE_COLORS.get(
                                        dist,
                                        "#7f7f7f"
                                    ),
                                    alpha=0.08,
                                    zorder=2
                                )

                            # -------------------------------------------------
                            # Per-distance metric annotation
                            # -------------------------------------------------
                            ax.text(
                                0.05,
                                0.95 - 0.06 * DISTANCES.index(dist),
                                f"{dist}: r = {r_val:.2f}, "
                                f"SD$_r$ = {rsd:.2f}",
                                transform=ax.transAxes,
                                fontsize=16,
                                va="top",
                                color= "#333333"
                            )

                # =====================================================
                # MADRC: calculate metrics over the whole cohort
                # =====================================================
                else:

                    x = np.log10(sub["Ref_mm3"].to_numpy())
                    y = np.log10(sub["Volume_mm3"].to_numpy())

                    # Scatter
                    ax.scatter(
                        x,
                        y,
                        s=36,
                        alpha=0.55,
                        color=MADRC_COLOR,
                        edgecolors="none",
                        zorder=3
                    )

                    a, b, r_val, p = _pearson(x, y)

                    if np.isfinite(b):

                        xg = np.linspace(lo, hi, 100)
                        yg = a + b * xg

                        resid = y - (a + b * x)
                        rsd = (
                            np.std(resid, ddof=2)
                            if len(resid) > 2
                            else np.nan
                        )

                        # LS fit
                        ax.plot(
                            xg,
                            yg,
                            color="#463F61",
                            lw=1.6,
                            zorder=4,
                            label="LS Fit"
                        )

                        # Prediction band
                        if np.isfinite(rsd):
                            ax.fill_between(
                                xg,
                                yg - 1.96 * rsd,
                                yg + 1.96 * rsd,
                                color="#dbd7d2",
                                alpha=0.2,
                                zorder=2
                            )

                        # MADRC metric annotation
                        ax.text(
                            0.05,
                            0.95 - 0.06 * DISTANCES.index(dist),
                            f"r = {r_val:.2f}, SD$_r$ = {rsd:.2f}",
                            transform=ax.transAxes,
                            fontsize=16,
                            va="top"
                        )

            # ---------------------------------------------------------
            # Axis formatting
            # ---------------------------------------------------------
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, alpha=0.25, ls=":", which="both")

            if r == 0 :
                ax.set_title(
                    f"{PANEL_LETTERS[c]} {METHOD_DISPLAY[method]}",
                    fontsize=20,
                    pad=8
                )

            if c == 0:
                ax.set_ylabel(
                    f"{coh}\nPred. volume [mm$^3$ log10]",
                    fontsize=20
                )

            if r == 1:
                ax.set_xlabel(
                    "Ref. volume [mm$^3$ log10]",
                    fontsize=20
                )

            handles, labels = axes[0, -1].get_legend_handles_labels()

            for h, l in zip(handles, labels):
                seen.setdefault(l, h)

            ordered_labels = [l for l in known if l in seen]
            ordered_labels += [l for l in seen if l not in known]

            ordered_handles = [seen[l] for l in ordered_labels]

    # -------------------------------------------------------------
    # Legend
    # -------------------------------------------------------------
    axes[0, -1].legend(
        ordered_handles,
        ordered_labels,
        fontsize=17,
        loc="lower right",
        title_fontsize=17,
        handlelength=1.2,
        handletextpad=0.5
    )

    fig.subplots_adjust(
        wspace=0.02,
        hspace=0.08,
        left=0.11
    )

    return fig

# =============================================================================
# NORMALIZED ERROR + P-VALUES
# =============================================================================
def valid_m(pvals) -> int:
    """Tests actually performed (non-NaN) in a family."""
    p = np.asarray(pvals, dtype=float)
    return int(np.count_nonzero(~np.isnan(p)))


def benjamini_hochberg(pvals) -> np.ndarray:
    """BH-adjusted q-values; preserves NaNs and input order."""
    p = np.asarray(pvals, dtype=float)
    q = np.full_like(p, np.nan)
    idx = np.where(~np.isnan(p))[0]
    m = idx.size
    if m == 0:
        return q
    order = idx[np.argsort(p[idx])]
    adj = p[order] * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]           # monotone from the top
    q[order] = np.clip(adj, 0.0, 1.0)
    return q


def collect_pvalue_family(m, labels):
    """Collect raw p over the whole table (sections x regions x pairs) in a
    fixed order, then BH-adjust once. Shared by the table and the CSV so both
    report identical numbers."""
    present = [s for s in SECTION_ORDER_TABLE if s in set(m["Distance"])]
    keys, raw = [], []
    for sec in present:
        for lab in labels:
            for a, b in PVALUE_PAIRS:
                keys.append((sec, lab, (a, b)))
                raw.append(pvalue(m, sec, lab, a, b))
    m_valid = valid_m(raw)
    q = benjamini_hochberg(raw) if APPLY_BH else raw
    return present, keys, dict(zip(keys, raw)), dict(zip(keys, q)), m_valid

def aggregate(m: pd.DataFrame) -> pd.DataFrame:
    return 


def pvalue(m: pd.DataFrame, distance: str, label: str,
           method_a: str, method_b: str) -> float:
    """Paired test on per-subject absolute errors for one (section, region).
    Returns the RAW p-value; BH correction is applied once per table."""
    sel = (m["Distance"] == distance) & (m["Label"] == label)
    a = m[sel & (m["Method"] == method_a)][["Subject", "AbsErr"]]
    b = m[sel & (m["Method"] == method_b)][["Subject", "AbsErr"]]
    merged = a.merge(b, on="Subject", suffixes=("_a", "_b"))
    if len(merged) < 1:
        return np.nan
    try:
        if TEST == "ranksum":
            from scipy.stats import ranksums
            _, p = ranksums(merged["AbsErr_a"], merged["AbsErr_b"])
        else:
            _, p = wilcoxon(merged["AbsErr_a"], merged["AbsErr_b"])
        return p
    except ValueError:
        return np.nan


def _fmt_p(p: float) -> str:
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "--"
    return f"{p:.3f}"


def ordered_labels(agg: pd.DataFrame) -> list:
    return sorted(set(agg["Label"]), key=str.lower)


# =============================================================================
# JOINT TABLE
# =============================================================================
JOINT_CAPTION = (
    r"Region-specific normalized volume error of automated segmentations of 3D "
    r"reconstructions of photographs, for the MADRC and UW cohorts. For each "
    r"region, the error $|V_{\mathrm{method}} - V_{\mathrm{ref}}|$ is divided by "
    r"the maximum absolute error for that region across all methods, subjects, "
    r"cohorts and slab distances, so values lie in $[0, 1]$; each cell is the "
    r"mean over subjects. Because a single per-region denominator is shared "
    r"across slab distances, the normalized error increases with slab thickness. "
    r"P-values are from Wilcoxon signed-rank tests (paired) on the per-subject "
    r"absolute errors. Abbreviations: PR = Photo-recon, Cub = cubic "
    r"interpolation, UNet = U-Net imputation. Gold-standard volumes are obtained "
    r"from MRI scans."
)


def build_error_latex(agg, labels):
    """Table 1: per-method normalized error only."""
    present = [s for s in SECTION_ORDER_TABLE if s in set(agg["Distance"])]
    ncol = 1 + len(METHODS)
    header = (r"\textbf{Region}"
              + "".join(r" & \textbf{%s}" % METHOD_DISPLAY[m_] for m_ in METHODS)
              + r" \\")
    L = [r"\begin{table*}[h!]", r"\centering", r"\caption{%s}" % ERROR_CAPTION,
         r"\label{app:volume_error_joint}",
         r"\begin{tabular}{l%s}" % ("c" * len(METHODS)), r"\toprule"]
    for si, sec in enumerate(present):
        L.append(r"\multicolumn{%d}{c}{\textbf{%s}} \\" % (ncol, SECTION_HEADER[sec]))
        L.append(r"\midrule")
        if si == 0:
            L.append(header)
        d = agg[agg["Distance"] == sec]
        for lab in labels:
            cells = []
            for method in METHODS:
                v = d[(d["Label"] == lab) & (d["Method"] == method)]["norm_err_mean"]
                cells.append(f"{float(v.iloc[0]):.3f}" if len(v) else "--")
            L.append("%-16s & %s \\\\" % (lab, " & ".join(cells)))
        L.append(r"\bottomrule" if si == len(present) - 1 else r"\midrule")
    L += [r"\end{tabular}", r"\end{table*}"]
    return "\n".join(L)

def build_pvalue_latex(m, labels):
    """Pairwise comparisons, Benjamini-Hochberg (FDR) corrected across the whole table."""
    present, keys, pmap, qmap, m_valid = collect_pvalue_family(m, labels)
    print(f"[volume] BH-FDR over m = {m_valid} valid tests (bold at q < {Q_LEVEL})")

    def cell(sec, lab, pair):
        qi = qmap.get((sec, lab, pair), np.nan)
        if qi is None or (isinstance(qi, float) and np.isnan(qi)):
            return "--"
        disp = (r"$< 0.0001$") if qi < 1e-4 else (r"%.3f" % qi)
        return disp

    caption = (
        r"Pairwise statistical comparisons of the per-subject absolute volume "
        r"errors between reconstruction methods, for the MADRC and UW datasets, "
        r"by region and slab distance (" +
        ("Wilcoxon signed-rank, paired" if TEST == "wilcoxon"
         else "Wilcoxon rank-sum, Mann-Whitney") +
        r"). Reported values are Benjamini-Hochberg adjusted $q$-values "
        r"controlling the false discovery rate across the $m=%d$ comparisons in "
        r"this table; entries with $q<0.05$ are shown in bold. Dashes indicate "
        r"comparisons without paired samples. Abbreviations: PR = Photo-recon, "
        r"Cub = cubic interpolation, UNet = U-Net imputation." % m_valid
    )
    ncol = 1 + len(PVALUE_PAIRS)
    header = (r"\textbf{Region}"
              + "".join(r" & \textbf{q (%s vs %s)}" % (METHOD_ABBR[a], METHOD_ABBR[b])
                        for a, b in PVALUE_PAIRS) + r" \\")
    L = [r"\begin{table*}[h!]", r"\centering", r"\caption{%s}" % caption,
         r"\label{app:volume_pvalues_joint}",
         r"\begin{tabular}{l%s}" % ("c" * len(PVALUE_PAIRS)), r"\toprule"]
    for si, sec in enumerate(present):
        L.append(r"\multicolumn{%d}{c}{\textbf{%s}} \\" % (ncol, SECTION_HEADER[sec]))
        L.append(r"\midrule")
        if si == 0:
            L.append(header)
        for lab in labels:
            cells = [cell(sec, lab, (a, b)) for a, b in PVALUE_PAIRS]
            L.append("%-16s & %s \\\\" % (lab, " & ".join(cells)))
        L.append(r"\bottomrule" if si == len(present) - 1 else r"\midrule")
    L += [r"\end{tabular}", r"\end{table*}"]
    return "\n".join(L)

def write_joint_csv(agg: pd.DataFrame, m: pd.DataFrame, labels: list) -> str:
    present = [s for s in SECTION_ORDER if s in set(agg["Distance"])]
    rows = []
    for sec in present:
        d = agg[agg["Distance"] == sec]
        for lab in labels:
            row = {"Section": SECTION_HEADER[sec], "Distance": sec, "Region": lab}
            for method in METHODS:
                r = d[(d["Label"] == lab) & (d["Method"] == method)]
                row[f"{method}_norm_err"] = float(r["norm_err_mean"].iloc[0]) if len(r) else np.nan
                row[f"{method}_norm_err_std"] = float(r["norm_err_std"].iloc[0]) if len(r) else np.nan
                row[f"{method}_n"] = int(r["n"].iloc[0]) if len(r) else 0
            for a, b in PVALUE_PAIRS:
                p = pvalue(m, sec, lab, a, b)
                if p < 1e-3:
                    write_p =  (r"$< 0.001$") 
                else:
                    write_p = (r"%.3f" % p)
                row[f"p_{METHOD_ABBR[a]}_vs_{METHOD_ABBR[b]}"] = write_p
            rows.append(row)
    path = os.path.join(OUT_DIR, "volume_error_stats_joint.csv")
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    return path

# =============================================================================
# JOINT FIGURE (2 rows: cohort; 3 columns: method)
# =============================================================================
def make_joint_figure(m_uw: pd.DataFrame, m_mad: pd.DataFrame) -> plt.Figure:
    rows = [("UW", m_uw, True), ("MADRC", m_mad, False)]

    def rel(mm):
        return (mm["Diff"] / mm["Ref_mm3"] * 100.0).to_numpy()

    pooled_rel, pooled_x = [], []
    for _, mm, _ in rows:
        if not mm.empty:
            pooled_rel.extend(rel(mm).tolist())
            pooled_x.extend(np.log10(mm["Ref_mm3"].to_numpy()).tolist())
    if pooled_rel:
        lo, hi = np.percentile(pooled_rel, [2, 98]); mrg = (hi - lo) * 0.10
        y_lo, y_hi = lo - mrg, hi + mrg
    else:
        y_lo, y_hi = -50.0, 50.0
    if pooled_x:
        x_lo, x_hi = min(pooled_x), max(pooled_x); xm = (x_hi - x_lo) * 0.04
        x_lo, x_hi = x_lo - xm, x_hi + xm
    else:
        x_lo, x_hi = 0.0, 1.0

    fig, axes = plt.subplots(2, 3, figsize=(18, 9.5), sharey=True, sharex=True)
    for r, (dlabel, mm, by_dist) in enumerate(rows):
        for c, method in enumerate(METHODS):
            ax = axes[r][c]
            sub = mm[mm["Method"] == method].copy()
            if not sub.empty:
                sub["RelErr"] = sub["Diff"] / sub["Ref_mm3"] * 100.0
                if by_dist:
                    for dist in DISTANCES:
                        dd = sub[sub["Distance"] == dist]
                        if dd.empty:
                            continue
                        ax.scatter(np.log10(dd["Ref_mm3"].to_numpy()),
                                   dd["RelErr"].to_numpy(), s=36, alpha=0.55,
                                   color=DISTANCE_COLORS.get(dist, "#7f7f7f"),
                                   edgecolors="none", label=dist, zorder=2)
                else:
                    ax.scatter(np.log10(sub["Ref_mm3"].to_numpy()),
                               sub["RelErr"].to_numpy(), s=36, alpha=0.55,
                               color=MADRC_COLOR, edgecolors="none", zorder=2)
                bias, sd = sub["RelErr"].mean(), sub["RelErr"].std()
                txt = (f"Bias = {bias:+.1f} %\n"
                       f"LoA  [{bias - 1.96 * sd:+.1f}, {bias + 1.96 * sd:+.1f}] %")
                ax.text(0.03, 0.97, txt, transform=ax.transAxes, fontsize=20,
                        verticalalignment="top", family="sans-serif",
                        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.55))

            ax.axhline(0, color="black", linewidth=1.0, linestyle="--", alpha=0.6, zorder=1)
            ax.set_ylim(y_lo, y_hi); ax.set_xlim(x_lo, x_hi)
            ax.grid(True, alpha=0.25, linestyle=":", which="both")
            if r == 0:
                ax.set_title(METHOD_DISPLAY[method], fontsize=20, fontweight="bold", pad=6)
            if c == 0:
                ax.set_ylabel(f"{dlabel}\nRelative volume error (%)",
                              fontsize=20, fontweight="bold")
            if r == len(rows) - 1:
                ax.set_xlabel("Reference volume [mm$^3$, log10]",
                              fontsize=20, fontweight="bold")
        if by_dist:
            axes[r][-1].legend(fontsize=20, loc="lower right",
                               title="Distance", title_fontsize=9)
    plt.tight_layout()
    return fig


# =============================================================================
# MAIN
# =============================================================================
def main():
    p = argparse.ArgumentParser(description="Joint UW+MADRC volume-correlation outputs.")
    p.add_argument("--uw-ref-dir",         required=True)
    p.add_argument("--uw-photo-recon-dir", required=True)
    p.add_argument("--uw-tricubic-dir",    required=True)
    p.add_argument("--uw-imputed-dir",     required=True)
    p.add_argument("--madrc-ref-dir",         required=True)
    p.add_argument("--madrc-photo-recon-dir", required=True)
    p.add_argument("--madrc-tricubic-dir",    required=True)
    p.add_argument("--madrc-imputed-dir",     required=True)
    p.add_argument("--out-dir",            required=True)
    args = p.parse_args()

    global OUT_DIR
    OUT_DIR = args.out_dir
    os.makedirs(OUT_DIR, exist_ok=True)

    print("[Loading UW...]")
    uw_df, uw_ref = load_uw_subjects(ref_dir=args.uw_ref_dir, photo_dir=args.uw_photo_recon_dir,
                                     tri_dir=args.uw_tricubic_dir, imp_dir=args.uw_imputed_dir)
    m_uw = process(uw_df, uw_ref, "UW")

    print("[Loading MADRC...]")
    mad_df, mad_ref = load_madrc_subjects(ref_dir=args.madrc_ref_dir, photo_dir=args.madrc_photo_recon_dir,
                                          tri_dir=args.madrc_tricubic_dir, imp_dir=args.madrc_imputed_dir)
    m_mad = process(mad_df, mad_ref, "MADRC")
    m_all = pd.concat([m_uw, m_mad], ignore_index=True)
    agg = (m_all.groupby(["Label", "Method", "Distance"])
              .agg(norm_err_mean=("AbsErr_rel", "mean"),
                   norm_err_std=("AbsErr_rel", "std"),
                   n=("Subject", "nunique"))
              .reset_index())
    labels = ordered_labels(agg)
    # print(f"  Sections : {[s for s in SECTION_ORDER if s in set(agg['Distance'])]}")
    print(f"  Regions  : {labels}")

    outputs = []
    outputs += build_summary_outputs(m_all, OUT_DIR)

    fig = make_joint_figure(m_uw, m_mad)
    svg = os.path.join(OUT_DIR, "task_2.1_uwmadrc_volume_correlations.svg")
    pdf = os.path.join(OUT_DIR, "task_2.1_uwmadrc_volume_correlations.pdf")
    fig.savefig(svg, bbox_inches="tight", dpi=300)
    fig.savefig(pdf, bbox_inches="tight", dpi=300)
    plt.close(fig)

    outputs += [pdf, svg]
    figc = make_concordance_figure(m_uw, m_mad)
    svg_c = os.path.join(OUT_DIR, "task_2.1_uwmadrc_volume_concordance.svg")
    pdf_c = os.path.join(OUT_DIR, "task_2.1_uwmadrc_volume_concordance.pdf")
    figc.savefig(svg_c, bbox_inches="tight", dpi=300)
    figc.savefig(pdf_c, bbox_inches="tight", dpi=300)
    plt.close(figc)
    outputs += [pdf_c, svg_c]

    tex_err = os.path.join(OUT_DIR, "volume_error_table_joint.tex")
    Path(tex_err).write_text(build_error_latex(agg, labels) + "\n", encoding="utf-8")
    outputs.append(tex_err)

    tex_p = os.path.join(OUT_DIR, "volume_pvalue_table_joint.tex")
    Path(tex_p).write_text(build_pvalue_latex(m_all, labels) + "\n", encoding="utf-8")
    outputs.append(tex_p)

    outputs.append(write_joint_csv(agg, m_all, labels))

    print("Wrote:")
    for f in outputs:
        print(f"  {f}")


if __name__ == "__main__":
    main()