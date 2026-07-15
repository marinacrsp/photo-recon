#!/usr/bin/env python3
"""
Task 1 (surface / thickness reconstruction error) - unified builder.

Reproduces, from the FreeSurfer reconstructions, the pial-surface, white-matter
surface and cortical-thickness error table and the 3-panel box-plot figure,
comparing Photo-recon (photo_recon.orig) and Imputed (photo_recon.machine_learning)
against the deformed gold-standard MRI, across UW-4/8/12 mm and MADRC.

The pipeline has two stages:

  1. compute  : walk the FreeSurfer tree, compute per-case, per-hemisphere,
                per-label errors, and cache them to a long CSV. Expensive
                (cKDTree nearest-neighbour over every vertex); run once.
  2. report   : read the cache, aggregate, and write the LaTeX table, the
                figure, and an audit CSV. Fast; re-run freely for tweaks.

A third method (e.g. Tricubic) is added by filling one entry in METHODS and
adding its name to METHOD_ORDER / PALETTE / LEGEND_LABEL, then re-running with
--recompute. Nothing else changes.

Usage:
    python build_task1.py                # report from cache (compute if absent)
    python build_task1.py --recompute    # force the surface computation
    python build_task1.py --push         # also copy outputs into OVERLEAF_REPO
"""

from __future__ import annotations

import os
import argparse
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
ROOT      = "/home/marina/ms_thesis"
UW_DIR    = os.path.join(ROOT, "photo_recon_uw")
MADRC_DIR = os.path.join(ROOT, "photo_recon_madrc")
OUT_DIR   = os.path.join(ROOT, "evaluation_results", "task_1_surface_reconstruction")
CACHE     = os.path.join(OUT_DIR, "task1_surface_errors_long.csv")

# Gold-standard (deformed MRI) location.
GT = {"uw_parent": "12_recon_any_original_mri_deformed", "uw_subdir": "mri",
      "madrc_subdir": "mri.deformed"}

# Methods. Each needs its UW parent folder + per-distance sub-folder template,
# and its MADRC sub-folder. To add Tricubic later, uncomment and fill in the
# real folder names, then add "Tricubic" to METHOD_ORDER / PALETTE / LEGEND_LABEL.
METHODS = {
    "Photo-recon": {
        "uw_parent":    "10_recon_any_photo",
        "uw_subdir":    "photo_recon_resampled_{d}",
        "madrc_subdir": "photo_recon.orig",
    },
    "Imputed": {
        "uw_parent":    "11_recon_any_imputations_unet",
        "uw_subdir":    "imputed_unet_resampled_{d}",
        "madrc_subdir": "photo_recon.machine_learning",
    },
    # "Tricubic": {
    #     "uw_parent":    "13_recon_any_imputations_tricubic",
    #     "uw_subdir":    "imputed_tricubic_resampled_{d}",
    #     "madrc_subdir": "photo_recon.tricubic",
    # },
}

METHOD_ORDER = ["Photo-recon", "Imputed"]      # add "Tricubic" here later
REFERENCE    = "Photo-recon"                    # baseline for the p-value row
COMPARISON   = "Imputed"                        # method tested against REFERENCE

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
PANEL_TITLES = ["(a) Pial matter error", "(b) White matter error",
                "(c) Cortical thickness"]

COND_HEADER = {"UW-4mm": "UW -- 4 mm", "UW-8mm": "UW -- 8 mm",
               "UW-12mm": "UW -- 12 mm", "MADRC": "MADRC"}

PALETTE       = {"Photo-recon": "#FFFBFF", "Imputed": "#DBE4EE", "Tricubic": "#A7A6F8"}
LEGEND_LABEL  = {"Photo-recon": "3D reconstruction\nof slab photographs",
                 "Imputed": "Imputed", "Tricubic": "Tricubic"}

CAPTION = (
    "Surface and thickness errors (in mm) for Recon-Any of 3D photo "
    "reconstructions, computed against gold-standard MRI references. P-values "
    "from Wilcoxon Rank Sum statistical tests comparing both methods are "
    "reported for all evaluations."
)


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
    if not os.path.isdir(case_root):
        print(f"[UW] missing {case_root}, skipping UW cohort")
        return
    for file in sorted(os.listdir(case_root)):
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
                try:
                    for h in ("lh", "rh"):
                        ms = _load(os.path.join(UW_DIR, spec["uw_parent"], file, sub, "surf", h),
                                   os.path.join(UW_DIR, spec["uw_parent"], file, sub, "label", h))
                        _emit(records, file, cond, h, method, per_label_errors(gt[h], ms))
                except Exception as exc:                      # noqa: BLE001
                    print(f"[UW] skip {file}/{d}/{method} ({exc})")


def compute_madrc(records: list) -> None:
    qc = os.path.join(MADRC_DIR, "best_recon_ss_qc")
    if not os.path.isdir(qc):
        print(f"[MADRC] missing {qc}, skipping MADRC cohort")
        return
    for file in sorted(os.listdir(qc)):
        inner = os.listdir(os.path.join(qc, file))
        if not inner:
            continue
        folder = os.path.join(qc, file, inner[0])
        hemis = ("lh", "rh") if "both" in file else (("lh",) if "left" in file else ("rh",))
        try:
            gt = {h: _load(os.path.join(folder, GT["madrc_subdir"], "surf", h),
                           os.path.join(folder, GT["madrc_subdir"], "label", h))
                  for h in hemis}
        except Exception as exc:                              # noqa: BLE001
            print(f"[MADRC] skip {file}: GT load failed ({exc})")
            continue
        for method, spec in METHODS.items():
            try:
                for h in hemis:
                    ms = _load(os.path.join(folder, spec["madrc_subdir"], "surf", h),
                               os.path.join(folder, spec["madrc_subdir"], "label", h))
                    _emit(records, file, "MADRC", h, method, per_label_errors(gt[h], ms))
            except Exception as exc:                          # noqa: BLE001
                print(f"[MADRC] skip {file}/{method} ({exc})")


def compute_and_cache() -> pd.DataFrame:
    records: list = []
    compute_uw(records)
    compute_madrc(records)
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


def table_value(pc: pd.DataFrame, metric: str, method: str, cond: str) -> float:
    v = pc[(pc.metric == metric) & (pc.method == method) & (pc.distance == cond)]["error"]
    return float(v.mean()) if len(v) else np.nan


def pvalue(pc: pd.DataFrame, metric: str, cond: str,
           method_a: str = COMPARISON, method_b: str = REFERENCE) -> float:
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


def build_latex(pc: pd.DataFrame) -> str:
    ncol = 1 + len(CONDITIONS)
    L = [r"\begin{table}[h!]", r"\centering", r"\caption{", CAPTION, r"}",
         r"\label{tab:task1_surface}",
         r"\begin{tabular}{l%s}" % ("c" * len(CONDITIONS)), r"\toprule"]

    for si, (metric, title) in enumerate(SECTIONS):
        L.append(r"\multicolumn{%d}{c}{\textbf{%s}} \\" % (ncol, title))
        L.append(r"\midrule")
        L.append(r"\textbf{Dataset}"
                 + "".join(r" & \textbf{%s}" % COND_HEADER[c] for c in CONDITIONS)
                 + r" \\")
        for method in METHOD_ORDER:
            cells = []
            for c in CONDITIONS:
                v = table_value(pc, metric, method, c)
                cells.append("--" if np.isnan(v) else f"{v:.3f}")
            L.append("%-11s & %s \\\\" % (method, " & ".join(cells)))
        prow = " & ".join(_fmt_p(pvalue(pc, metric, c)) for c in CONDITIONS)
        L.append("%-11s & %s \\\\" % ("p-Value", prow))
        L.append(r"\bottomrule" if si == len(SECTIONS) - 1 else r"\midrule")

    L += [r"\end{tabular}", r"\end{table}"]
    return "\n".join(L)


def build_figure(pc: pd.DataFrame) -> plt.Figure:
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7))
    for i, (metric, _title) in enumerate(SECTIONS):
        ax = axes[i]
        d = pc[pc.metric == metric]
        sns.boxplot(
            data=d, x="distance", y="error", hue="method",
            order=CONDITIONS, hue_order=METHOD_ORDER, palette=PALETTE,
            dodge=True, width=0.55, ax=ax,
            boxprops=dict(edgecolor="black", linewidth=1.1),
            whiskerprops=dict(color="black", linewidth=1.1),
            capprops=dict(color="black", linewidth=1.1),
            medianprops=dict(color="black", linewidth=1.1),
            flierprops=dict(marker="o", markerfacecolor="white",
                            markeredgecolor="black", markersize=3),
        )
        ax.set_title(PANEL_TITLES[i], fontsize=13)
        ax.set_xlabel("")
        ax.set_ylabel("Error (mm)", fontsize=12)
        ax.tick_params(axis="both", labelsize=11)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
        if metric in ("pial", "wm"):
            ax.set_ylim(0.5, 2.15)

        leg = ax.get_legend()
        if i == 0:
            handles = [Patch(facecolor=PALETTE[m], edgecolor="black",
                             label=LEGEND_LABEL[m]) for m in METHOD_ORDER]
            ax.legend(handles=handles, loc="upper left", frameon=True, fontsize=9)
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
    svg = os.path.join(OUT_DIR, "task1_surface_errors.svg")
    pdf = os.path.join(OUT_DIR, "task1_surface_errors.pdf")
    fig.savefig(svg, bbox_inches="tight", dpi=300)
    fig.savefig(pdf, bbox_inches="tight", dpi=300)
    plt.close(fig)

    tex = os.path.join(OUT_DIR, "task1_surface_errors.tex")
    Path(tex).write_text(build_latex(pc) + "\n", encoding="utf-8")

    audit = os.path.join(OUT_DIR, "task1_surface_errors_audit.csv")
    pc.to_csv(audit, index=False, encoding="utf-8-sig")
    return [tex, pdf, svg, audit]


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
    ap = argparse.ArgumentParser(description="Build Task 1 surface-error table + figure.")
    ap.add_argument("--recompute", action="store_true",
                    help="force the surface computation instead of using the cache")
    ap.add_argument("--push", action="store_true",
                    help="copy outputs into OVERLEAF_REPO, git commit and push")
    args = ap.parse_args()

    pc = per_case_means(load_long(recompute=args.recompute))
    outputs = save_outputs(pc)
    print("Wrote:")
    for f in outputs:
        print("  ", f)
    if args.push:
        push_to_overleaf(outputs)


if __name__ == "__main__":
    main()