#!/usr/bin/env python3
"""
Task 1 (surface / thickness reconstruction error) - unified builder.

Reproduces, from the FreeSurfer reconstructions, the pial-surface, white-matter
surface and cortical-thickness error table and the 3-panel box-plot figure,
now comparing THREE methods against the deformed gold-standard MRI:
  * Photo-recon  (photo_recon.orig            / photo_recon_resampled_{d})
  * Imputed      (photo_recon.machine_learning / imputed_unet_resampled_{d})
  * Tricubic     (photo_recon_tricubic_gray    / <UW template, see note>)
across UW-4/8/12 mm and MADRC.

The pipeline has two stages:
  1. compute : walk the FreeSurfer tree, compute per-case, per-hemisphere,
               per-label errors, and cache them to a long CSV. Expensive
               (cKDTree nearest-neighbour over every vertex); run once.
  2. report  : read the cache, aggregate, print subject counts, and write the
               LaTeX table, the figure and audit CSVs. Fast; re-run freely.

MADRC layout note (important):
  Photo-recon and Imputed are CO-LOCATED inside
      best_recon_ss_qc/<case>/<inner>/<madrc_subdir>
  Tricubic lives in a SEPARATE tree,
      tricubic-recon-any/<base_subject>/photo_recon_tricubic_gray
  where <base_subject> is the qc case name with the _both/_left/_right suffix
  removed (e.g. sub-2604_both -> sub-2604). Hemispheres for Tricubic are taken
  from the qc case naming, so the three methods stay aligned per subject.

UW layout note:
  Only 10_/11_/12_ were provided, so the UW Tricubic folder is unknown. Set
  UW_TRICUBIC["uw_parent"] / ["uw_subdir"] below if a UW Tricubic tree exists.
  If Tricubic was run on MADRC only, leave the placeholders: those UW cases
  simply fail to load and are skipped, so the UW Tricubic cells stay blank and
  the UW panels keep two boxes while MADRC shows three.

Reproducibility note:
  In your version compute_madrc() was commented out, so any cached CSV is
  UW-only and has no Tricubic. Run once with --recompute after these changes.

Usage:
    python build_task1.py --uw-dir ... --madrc-dir ... --out-dir ...            # report (compute if cache absent)
    python build_task1.py --uw-dir ... --madrc-dir ... --out-dir ... --recompute
    python build_task1.py ... --push                                            # also copy to OVERLEAF_REPO
"""

from __future__ import annotations
from matplotlib.transforms import ScaledTranslation
import os
import re
import argparse
import itertools
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                      # headless / one-click safe; remove for inline
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
from scipy.stats import wilcoxon

# =============================================================================
# CONFIGURATION
# =============================================================================
# These are initialized from the command line.
UW_DIR = None
MADRC_DIR = None
OUT_DIR = None
CACHE = None
OVERLEAF_REPO = None
APPLY_BH = False
# Parent under MADRC_DIR that holds the gold standard and the co-located methods.
MADRC_QC_PARENT = "best_recon_ss_qc"

# Gold-standard (deformed MRI) location.
GT = {"uw_parent": "12_recon_any_original_mri_deformed", "uw_subdir": "mri",
      "madrc_subdir": "mri.deformed"}
Q_LEVEL=0.05
# --- UW Tricubic placeholder (edit if a UW tricubic tree exists) -------------
# Set to the real UW parent folder and per-distance sub-folder template. Leave
# as-is if tricubic is MADRC-only; missing UW cases are skipped gracefully.
UW_TRICUBIC = {
    "uw_parent": "13_recon_any_tricubic",             # TODO: real folder or leave (skipped)
    "uw_subdir": "photo_recon_{d}_tricubic_gray",  # TODO: verify template
}

# Methods. UW: uw_parent + uw_subdir template. MADRC: madrc_parent (co-located
# when == MADRC_QC_PARENT, else a separate tree keyed by base subject id) +
# madrc_subdir.
METHODS = {
    "Photo-recon": {
        "uw_parent":    "10_recon_any_photo",
        "uw_subdir":    "photo_recon_resampled_{d}",
        "madrc_parent": MADRC_QC_PARENT,               # co-located in qc case folder
        "madrc_subdir": "photo_recon.orig",
    },
    "Cubic": {
        "uw_parent":    UW_TRICUBIC["uw_parent"],
        "uw_subdir":    UW_TRICUBIC["uw_subdir"],
        "madrc_parent": "tricubic-recon-any",          # SEPARATE tree, base subject id
        "madrc_subdir": "photo_recon_tricubic_gray",
    },
    "Imputed": {
        "uw_parent":    "11_recon_any_imputations_unet",
        "uw_subdir":    "imputed_unet_resampled_{d}",
        "madrc_parent": MADRC_QC_PARENT,               # co-located in qc case folder
        "madrc_subdir": "photo_recon.machine_learning",
    },
}

METHOD_ORDER = ["Photo-recon","Cubic","Imputed"]
REFERENCE    = "Photo-recon"                    # baseline for the p-value row
COMPARISON   = "Imputed"                        # method tested against REFERENCE

# The p-value table auto-generates every pairwise comparison among the methods
# present, in METHOD_ORDER order, i.e. for three methods:
#   Photo-recon vs Imputed, Photo-recon vs Tricubic, Imputed vs Tricubic.
# REFERENCE / COMPARISON above remain only as the default args of pvalue().

# Statistical test for the p-value row. The manuscript caption says
# "Wilcoxon Rank Sum" (unpaired), but the notebook uses scipy.stats.wilcoxon
# (paired signed-rank) on the per-case means, which is what produced the
# reference numbers. "wilcoxon" reproduces them; "ranksum" matches the caption.
TEST = "wilcoxon"                               # "wilcoxon" | "ranksum"

# Notebook cell 5 averages hemispheres as (lh+rh)/2 for pial/wm but, by an
# apparent typo, (rh+rh)/2 for cortical thickness (RH only). False uses the
# correct (lh+rh)/2 for all three; True reproduces the notebook's cortical bug.
CORTICAL_RH_ONLY = False

DISTANCES  = ["4mm", "8mm", "12mm"]
CONDITIONS = ["UW-4mm", "UW-8mm", "UW-12mm", "MADRC"]

SECTIONS = [("pial", "Pial Surface Error"),
            ("wm",   "White Matter Surface Error"),
            ("cortical", "Cortical Thickness Error")]
PANEL_TITLES = ["(a) Pial error", "(b) White matter error",
                "(c) Cortical thickness"]

COND_HEADER = {"UW-4mm": "UW -- 4 mm", "UW-8mm": "UW -- 8 mm",
               "UW-12mm": "UW -- 12 mm", "MADRC": "MADRC"}

PALETTE       = {"Photo-recon": "#FFFBFF", "Imputed": "#DBE4EE", "Cubic": "#A7A6F8"}
LEGEND_LABEL  = {"Photo-recon": "3D reconstruction\nof slab photographs",
                 "Imputed": "Imputed", "Cubic": "Cubic"}

# Human-readable name of the test actually used, for the p-value caption.
TEST_NAME = {"wilcoxon": "Wilcoxon signed-rank",
             "ranksum": "Wilcoxon rank-sum (Mann-Whitney)"}

CAPTION_SCORES = (
    "Surface and thickness errors (in mm) for Recon-Any of 3D photo "
    "reconstructions, computed against gold-standard MRI references, reported "
    "for each reconstruction method and evaluation condition."
)

CAPTION_PVALUES = (
    "Pairwise statistical comparisons of the surface and thickness errors "
    "between reconstruction methods. P-values from " + TEST_NAME[TEST] +
    " tests are reported for all evaluations."
)
plt.rcParams.update({"font.size": 16, "axes.labelsize": 16,
                     "xtick.labelsize": 16, "ytick.labelsize": 16,
                     "legend.fontsize": 16})


# =============================================================================
# HELPERS
# =============================================================================
def _base_subject(name: str) -> str:
    """Strip the MADRC hemisphere suffix: sub-2604_both -> sub-2604."""
    return re.sub(r"_(both|left|right)$", "", name)


def _hemis_from_name(name: str):
    """Hemisphere set implied by a MADRC qc case name."""
    if "both" in name:
        return ("lh", "rh")
    return ("lh",) if "left" in name else ("rh",)
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

# =============================================================================
# STAGE 1 - SURFACE COMPUTATION (needs nibabel + the FreeSurfer tree)
# =============================================================================
def get_surface_data_hemis(fname_surf: str, fname_label: str):
    """Read one hemisphere: per-label mean thickness, white/pial vertices, labels."""
    import nibabel as nib
    surf_wm, _ = nib.freesurfer.read_geometry(f"{fname_surf}.white")
    surf_pial, _ = nib.freesurfer.read_geometry(f"{fname_surf}.pial")
    cort_thickness = np.linalg.norm(surf_wm - surf_pial, axis=1)
    labels, _, _ = nib.freesurfer.read_annot(f"{fname_label}.aparc.annot")
    cthick = pd.DataFrame({"Thickness": cort_thickness, "Label": labels.astype(int)})
    cthick = cthick.groupby("Label", as_index=False)["Thickness"].mean()
    return {"cthick": cthick, "wm": surf_wm, "pial": surf_pial, "labels": labels}


def _symmetric_label_distance(gt_vertices, m_vertices, gt_labels, m_labels):
    """Per-label mean of the symmetric surface-to-surface nearest distance."""
    from scipy.spatial import cKDTree
    d_m2g, _ = cKDTree(gt_vertices).query(m_vertices, k=1, workers=-1)   # method -> GT
    d_g2m, _ = cKDTree(m_vertices).query(gt_vertices, k=1, workers=-1)   # GT -> method
    mean_m2g = pd.Series(d_m2g).groupby(m_labels.astype(int)).mean()
    mean_g2m = pd.Series(d_g2m).groupby(gt_labels.astype(int)).mean()
    return (mean_m2g + mean_g2m) / 2.0                                   # index = label


def per_label_errors(gt: dict, meth: dict) -> pd.DataFrame:
    """Per-label pial, white and thickness errors for one method vs GT (one hemi)."""
    wm = _symmetric_label_distance(gt["wm"], meth["wm"], gt["labels"], meth["labels"])
    pial = _symmetric_label_distance(gt["pial"], meth["pial"], gt["labels"], meth["labels"])
    ct = meth["cthick"].merge(gt["cthick"], on="Label", suffixes=("_m", "_g"))
    ct_err = (ct["Thickness_m"] - ct["Thickness_g"]).abs()
    ct_err.index = ct["Label"].astype(int)
    out = pd.concat({"err_wm": wm, "err_pial": pial, "err_cortical": ct_err}, axis=1)
    out.index.name = "label"
    return out.reset_index()


def _emit(records, file, cond, hemi, method, err_df):
    for _, r in err_df.iterrows():
        records.append({"file": file, "distance": cond, "hemi": hemi,
                        "label": int(r["label"]), "method": method,
                        "err_wm": r["err_wm"], "err_pial": r["err_pial"],
                        "err_cortical": r["err_cortical"]})


def _load(surf_prefix, label_prefix):
    return get_surface_data_hemis(surf_prefix, label_prefix)


def compute_uw(records: list) -> None:
    case_root = os.path.join(UW_DIR, "00_photo_recon")
    # 00_photo_recon may not exist; fall back to the GT parent for the case list.
    if not os.path.isdir(case_root):
        case_root = os.path.join(UW_DIR, GT["uw_parent"])
    if not os.path.isdir(case_root):
        print(f"[UW] missing case root {case_root}, skipping UW cohort")
        return
    cases = sorted(d for d in os.listdir(case_root)
                   if os.path.isdir(os.path.join(case_root, d)))
    for file in cases:
        try:
            gt = {h: _load(os.path.join(UW_DIR, GT["uw_parent"], file, GT["uw_subdir"], "surf", h),
                           os.path.join(UW_DIR, GT["uw_parent"], file, GT["uw_subdir"], "label", h))
                  for h in ("lh", "rh")}
        except Exception as exc:                              # noqa: BLE001
            print(f"[UW] skip {file}: GT load failed ({exc})")
            continue
        for d in DISTANCES:
            cond = f"UW-{d}"
            for method, spec in METHODS.items():
                sub = spec["uw_subdir"].format(d=d)
                for h in ("lh", "rh"):
                    try:
                        ms = _load(os.path.join(UW_DIR, spec["uw_parent"], file, sub, "surf", h),
                                   os.path.join(UW_DIR, spec["uw_parent"], file, sub, "label", h))
                        _emit(records, file, cond, h, method, per_label_errors(gt[h], ms))
                    except Exception as exc:                  # noqa: BLE001
                        print(f"[UW] skip {file}/{d}/{method}/{h} ({exc})")


def _madrc_method_dir(spec: dict, colocated_folder: str, base_subject: str) -> str:
    """Resolve a method's MADRC directory: co-located in the qc case folder, or
    in a separate parent tree keyed by the base subject id (e.g. Tricubic)."""
    parent = spec.get("madrc_parent", MADRC_QC_PARENT)
    if parent == MADRC_QC_PARENT:
        return os.path.join(colocated_folder, spec["madrc_subdir"])
    return os.path.join(MADRC_DIR, parent, base_subject, spec["madrc_subdir"])


def compute_madrc(records: list) -> None:
    qc = os.path.join(MADRC_DIR, MADRC_QC_PARENT)
    if not os.path.isdir(qc):
        print(f"[MADRC] missing {qc}, skipping MADRC cohort")
        return
    for file in sorted(os.listdir(qc)):
        case_path = os.path.join(qc, file)
        if not os.path.isdir(case_path):
            continue
        inner = sorted(d for d in os.listdir(case_path)
                       if os.path.isdir(os.path.join(case_path, d)))
        if not inner:
            continue
        folder = os.path.join(case_path, inner[0])       # the *_synthsr_recon_any subfolder
        base = _base_subject(file)                       # sub-2604_both -> sub-2604
        hemis = _hemis_from_name(file)
        try:
            gt = {h: _load(os.path.join(folder, GT["madrc_subdir"], "surf", h),
                           os.path.join(folder, GT["madrc_subdir"], "label", h))
                  for h in hemis}
        except Exception as exc:                          # noqa: BLE001
            print(f"[MADRC] skip {file}: GT load failed ({exc})")
            continue
        for method, spec in METHODS.items():
            mdir = _madrc_method_dir(spec, folder, base)
            for h in hemis:
                try:
                    ms = _load(os.path.join(mdir, "surf", h),
                               os.path.join(mdir, "label", h))
                    # Emit under the qc case name so the three methods align
                    # per subject for the paired p-value merge.
                    _emit(records, file, "MADRC", h, method, per_label_errors(gt[h], ms))
                except Exception as exc:                  # noqa: BLE001
                    print(f"[MADRC] skip {file}/{method}/{h} ({exc})")


def compute_and_cache() -> pd.DataFrame:
    records: list = []
    compute_uw(records)
    compute_madrc(records)                                # ENABLED (was commented out)
    df = pd.DataFrame.from_records(records)
    if df.empty:
        raise RuntimeError("No surface records computed. Check ROOT / paths.")
    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_csv(CACHE, index=False)
    print(f"Cached {len(df)} rows to {CACHE}")
    return df


def load_long(recompute: bool = False) -> pd.DataFrame:
    if (not recompute) and os.path.exists(CACHE):
        print(f"Loading cached surface errors from {CACHE}")
        return pd.read_csv(CACHE)
    return compute_and_cache()


# =============================================================================
# SUBJECT COUNTS
# =============================================================================
def report_counts(long: pd.DataFrame) -> pd.DataFrame:
    """Count subjects that actually contributed measurements, per dataset and
    per method. MADRC subjects are de-duplicated by base id (hemisphere splits
    of one subject count once). 'complete_all_methods' = subjects present for
    every method that appears in that dataset. Writes a per-condition CSV."""
    df = long.copy()
    df["dataset"] = np.where(df["distance"].astype(str).str.startswith("UW"),
                             "UW", "MADRC")
    df["subject"] = df.apply(
        lambda r: _base_subject(r["file"]) if r["dataset"] == "MADRC" else r["file"],
        axis=1,
    )

    rows = []
    for ds, g in df.groupby("dataset"):
        methods_here = [m for m in METHOD_ORDER if m in set(g["method"])]
        per_method = {m: g[g.method == m]["subject"].nunique() for m in METHOD_ORDER}
        sets = [set(g[g.method == m]["subject"]) for m in methods_here]
        complete = len(set.intersection(*sets)) if sets else 0
        rows.append({"dataset": ds,
                     "subjects_total": g["subject"].nunique(),
                     "complete_all_methods": complete,
                     **{f"n_{m}": per_method[m] for m in METHOD_ORDER}})
    summary = pd.DataFrame(rows).set_index("dataset")

    # Per-condition x method breakdown (rigorous, for the audit trail).
    per_cond = (df.groupby(["distance", "method"])["subject"].nunique()
                  .rename("n_subjects").reset_index())

    os.makedirs(OUT_DIR, exist_ok=True)
    summary.to_csv(os.path.join(OUT_DIR, "task1_subject_counts.csv"))
    per_cond.to_csv(os.path.join(OUT_DIR, "task1_subject_counts_by_condition.csv"),
                    index=False)

    print("\n================ SUBJECT COUNTS (subjects actually used) ================")
    print(summary.to_string())
    print("\nPer condition x method:")
    print(per_cond.to_string(index=False))
    print("=========================================================================\n")
    return summary


# =============================================================================
# STAGE 2 - AGGREGATION, TABLE, FIGURE
# =============================================================================
def per_case_means(long: pd.DataFrame) -> pd.DataFrame:
    """Hemisphere-average per label, then per-case mean over labels."""
    metric_map = {"err_pial": "pial", "err_wm": "wm", "err_cortical": "cortical"}
    m = long.melt(id_vars=["file", "distance", "hemi", "label", "method"],
                  value_vars=list(metric_map), var_name="metric", value_name="error")
    m["metric"] = m["metric"].map(metric_map)
    m["error"] = pd.to_numeric(m["error"], errors="coerce")
    m = m.dropna(subset=["error"])

    if CORTICAL_RH_ONLY:
        m = m[~((m.metric == "cortical") & (m.hemi != "rh"))]

    hemi_avg = m.groupby(["file", "distance", "label", "method", "metric"],
                         as_index=False)["error"].mean()
    return hemi_avg.groupby(["file", "distance", "method", "metric"],
                            as_index=False)["error"].mean()


def _methods_present(pc: pd.DataFrame) -> list:
    """METHOD_ORDER filtered to methods that produced any data (avoids empty
    all-'--' rows and seaborn hue_order errors when a method is absent)."""
    have = set(pc["method"])
    return [m for m in METHOD_ORDER if m in have]


def table_value(pc: pd.DataFrame, metric: str, method: str, cond: str) -> float:
    v = pc[(pc.metric == metric) & (pc.method == method) & (pc.distance == cond)]["error"]
    return float(v.mean()) if len(v) else np.nan


def pvalue(pc: pd.DataFrame, metric: str, cond: str,
           method_a: str = COMPARISON, method_b: str = REFERENCE, n_hypotheses: int = 12) -> float:
    sel = (pc.metric == metric) & (pc.distance == cond)
    a = pc[sel & (pc.method == method_a)][["file", "error"]]
    b = pc[sel & (pc.method == method_b)][["file", "error"]]
    merged = a.merge(b, on="file", suffixes=("_a", "_b"))
    if len(merged) < 1:
        return np.nan
    try:
        if TEST == "ranksum":
            from scipy.stats import ranksums
            _, p = ranksums(merged["error_a"], merged["error_b"])
        else:
            _, p = wilcoxon(merged["error_a"], merged["error_b"])
        return float(p)
    except ValueError:
        return np.nan


def _fmt_p(p: float) -> str:
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "--"
    if p < 0.001:
        return r"$<$0.001"
    return f"{p:.3f}"

def valid_m(pvals) -> int:
    """Tests actually performed (non-NaN) in a family."""
    p = np.asarray(pvals, dtype=float)
    return int(np.count_nonzero(~np.isnan(p)))

def _pairwise(methods: list):
    """All method pairs in METHOD_ORDER order (A,B),(A,C),(B,C) for 3 methods."""
    return list(itertools.combinations(methods, 2))


def build_latex_scores(pc: pd.DataFrame) -> str:
    """Table 1: mean errors per method and condition (no p-values)."""
    methods = _methods_present(pc)
    ncol = 1 + len(CONDITIONS)
    L = [r"\begin{table}[h!]", r"\centering", r"\caption{", CAPTION_SCORES, r"}",
         r"\label{tab:task1_surface}",
         r"\begin{tabular}{l%s}" % ("c" * len(CONDITIONS)), r"\toprule"]

    for si, (metric, title) in enumerate(SECTIONS):
        L.append(r"\multicolumn{%d}{c}{\textbf{%s}} \\" % (ncol, title))
        L.append(r"\midrule")
        L.append(r"\textbf{Dataset}"
                 + "".join(r" & \textbf{%s}" % COND_HEADER[c] for c in CONDITIONS)
                 + r" \\")
        for method in methods:
            cells = []
            for c in CONDITIONS:
                v = table_value(pc, metric, method, c)
                cells.append("--" if np.isnan(v) else f"{v:.3f}")
            L.append("%-11s & %s \\\\" % (method, " & ".join(cells)))
        L.append(r"\bottomrule" if si == len(SECTIONS) - 1 else r"\midrule")

    L += [r"\end{tabular}", r"\end{table}"]
    return "\n".join(L)


def build_latex_pvalues(pc: pd.DataFrame) -> str:
    """Pairwise comparisons, Benjamini-Hochberg (FDR) corrected across the whole table."""
    pairs = _pairwise(_methods_present(pc))

    keys, raw = [], []
    for metric, _t in SECTIONS:
        for ma, mb in pairs:
            for c in CONDITIONS:
                keys.append((metric, (ma, mb), c))
                raw.append(pvalue(pc, metric, c, ma, mb))

    m = valid_m(raw)
    q = benjamini_hochberg(raw) if APPLY_BH else raw
    qmap = dict(zip(keys, q))
    print(f"[surface] BH-FDR over m = {m} valid tests (bold at q < {Q_LEVEL})")

    def cell(metric, pair, c):
        qi = qmap[(metric, pair, c)]
        if qi is None or (isinstance(qi, float) and np.isnan(qi)):
            return "--"
        disp = (r"$< 0.0001$") if qi < 1e-4 else (r"%.3f" % qi)
        return disp

    caption = (
        r"Pairwise statistical comparisons of the surface and thickness errors "
        r"between reconstruction methods (" + TEST_NAME[TEST] + r", paired). "
        r"Reported values are Benjamini-Hochberg adjusted $q$-values controlling "
        r"the false discovery rate across the $m=%d$ comparisons in this table; "
        r"entries with $q<0.05$ are shown in bold. Dashes indicate comparisons "
        r"without paired samples." % m
    )

    ncol = 1 + len(CONDITIONS)
    L = [r"\begin{table}[h!]", r"\centering", r"\caption{", caption, r"}",
         r"\label{tab:task1_pvalues}",
         r"\begin{tabular}{l%s}" % ("c" * len(CONDITIONS)), r"\toprule"]
    for si, (metric, title) in enumerate(SECTIONS):
        L.append(r"\multicolumn{%d}{c}{\textbf{%s}} \\" % (ncol, title))
        L.append(r"\midrule")
        L.append(r"\textbf{Comparison}"
                 + "".join(r" & \textbf{%s}" % COND_HEADER[c] for c in CONDITIONS)
                 + r" \\")
        for ma, mb in pairs:
            row = " & ".join(cell(metric, (ma, mb), c) for c in CONDITIONS)
            L.append(r"%s vs.\ %s & %s \\" % (ma, mb, row))
        L.append(r"\bottomrule" if si == len(SECTIONS) - 1 else r"\midrule")
    L += [r"\end{tabular}", r"\end{table}"]
    return "\n".join(L)


def build_figure(pc: pd.DataFrame) -> plt.Figure:
    methods = _methods_present(pc)
    palette = {m: PALETTE[m] for m in methods}
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    for i, (metric, _title) in enumerate(SECTIONS):
        ax = axes[i]
        d = pc[pc.metric == metric]
        sns.boxplot(
            data=d, x="distance", y="error", hue="method",
            order=CONDITIONS, hue_order=methods, palette=palette,
            dodge=True, width=0.4, ax=ax,
            boxprops=dict(edgecolor="black", linewidth=1.1),
            whiskerprops=dict(color="black", linewidth=1.1),
            capprops=dict(color="black", linewidth=1.1),
            medianprops=dict(color="black", linewidth=1.1),
            flierprops=dict(marker="o", markerfacecolor="white",
                            markeredgecolor="black", markersize=3),
        )
        # ax.set_axisbelow(True)                      # draw grid behind the data, not on top
        ax.grid(True, axis="y", linestyle="--", alpha=0.5, linewidth=1)
        
        ax.set_title(PANEL_TITLES[i], fontsize=16)
        ax.set_xlabel("")
        if i != 1:
            ax.set_ylabel("Error (mm)", fontsize=16)
        else:
            ax.set_ylabel("")
        ax.tick_params(axis="y", which="both", left=True, length=8, width=1.0, color="black")
        ax.tick_params(axis="x", which="both", bottom=True, length=4, width=1.0, color="black")
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right")

        if metric in ("pial", "wm") and len(d):
            # Data-driven upper bound so Tricubic is not clipped; floor at 2.15
            # to preserve the original look when tricubic errors are small.
            ymax = max(2.15, float(d["error"].quantile(0.99)) * 1.05)
            ax.set_ylim(0.5, ymax)

        leg = ax.get_legend()
        if i == 0:
            handles = [Patch(facecolor=PALETTE[m], edgecolor="black",
                             label=LEGEND_LABEL[m]) for m in methods]
            ax.legend(handles=handles, loc="upper left", frameon=True, fontsize=12)
        elif leg is not None:
            leg.remove()

    fig.tight_layout()
    return fig


# =============================================================================
# OUTPUTS
# =============================================================================
def save_outputs(pc: pd.DataFrame) -> list:
    os.makedirs(OUT_DIR, exist_ok=True)

    fig = build_figure(pc)
    svg = os.path.join(OUT_DIR, "task_1_uwmadrc_surface_errors.svg")
    pdf = os.path.join(OUT_DIR, "task_1_uwmadrc_surface_errors.pdf")
    fig.savefig(svg, bbox_inches="tight", dpi=300)
    fig.savefig(pdf, bbox_inches="tight", dpi=300)
    plt.close(fig)

    tex_scores = os.path.join(OUT_DIR, "task1_surface_errors.tex")
    Path(tex_scores).write_text(build_latex_scores(pc) + "\n", encoding="utf-8")

    tex_pvals = os.path.join(OUT_DIR, "task1_surface_pvalues.tex")
    Path(tex_pvals).write_text(build_latex_pvalues(pc) + "\n", encoding="utf-8")

    audit = os.path.join(OUT_DIR, "task1_surface_errors_audit.csv")
    pc.to_csv(audit, index=False, encoding="utf-8-sig")
    return [tex_scores, tex_pvals, pdf, svg, audit]


def push_to_overleaf(files: list, repo=None,
                     message="Update Task 1 table and figure") -> None:
    repo = repo or os.environ.get("OVERLEAF_REPO")
    if not repo:
        raise SystemExit("Set OVERLEAF_REPO (env var or edit here) before --push.")
    repo = os.path.abspath(repo)
    for f in files:
        shutil.copy(f, os.path.join(repo, os.path.basename(f)))
    subprocess.run(["git", "-C", repo, "add", "--all"], check=True)
    subprocess.run(["git", "-C", repo, "commit", "-m", message], check=True)
    subprocess.run(["git", "-C", repo, "push"], check=True)
    print(f"Pushed {len(files)} file(s) to {repo}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Task 1 surface-error table + figure (3 methods)."
    )
    parser.add_argument("--uw-dir", type=str, required=True, help="Path to the UW dataset.")
    parser.add_argument("--madrc-dir", type=str, required=True, help="Path to the MADRC dataset.")
    parser.add_argument("--out-dir", type=str, required=True, help="Directory for outputs.")
    parser.add_argument("--overleaf-repo", type=str, default=None, help="Optional local Overleaf repo.")
    parser.add_argument("--recompute", action="store_true", help="Force recomputation.")
    parser.add_argument("--push", action="store_true", help="Copy outputs to Overleaf and git push.")
    args = parser.parse_args()

    global UW_DIR, MADRC_DIR, OUT_DIR, CACHE, OVERLEAF_REPO
    UW_DIR = args.uw_dir
    MADRC_DIR = args.madrc_dir
    OUT_DIR = args.out_dir
    CACHE = os.path.join(OUT_DIR, "task1_surface_errors_long.csv")
    OVERLEAF_REPO = args.overleaf_repo

    long = load_long(recompute=args.recompute)
    report_counts(long)
    pc = per_case_means(long)
    outputs = save_outputs(pc)

    print("Wrote:")
    for f in outputs:
        print(f"  {f}")

    if args.push:
        push_to_overleaf(outputs, repo=OVERLEAF_REPO)


if __name__ == "__main__":
    main()