#!/usr/bin/env python3
"""
Task 2 (SynthSeg segmentation) - unified builder.

Loads the UW cohort (4 / 8 / 12 mm) and the MADRC cohort for the Photo-recon,
Tricubic and Imputed methods, merges them into a single long dataframe, and
regenerates in one run:

    * task2_combined_boxplot.svg / .pdf   (single figure, all conditions)
    * task2_scores.tex                     (Overleaf-ready table)
    * task2_scores_audit.csv               (means + p-values, machine readable)

Significance is a Wilcoxon signed-rank test of each method against Photo-recon,
paired on (Case, Label) within each Condition x Region.

Usage:
    python build_task2.py            # regenerate figure + table + audit
    python build_task2.py --push     # also copy into OVERLEAF_REPO and git push
"""

from __future__ import annotations

import os
import glob
import shutil
import argparse
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
UW_DIR    = "/home/marina/ms_thesis/photo_recon_uw"
MADRC_DIR = "/home/marina/ms_thesis/photo_recon_madrc"
OUT_DIR   = "/home/marina/ms_thesis/evaluation_results/task_2_synthseg_segmentation"

# Methods plotted per condition. Default includes Tricubic (your explicit request).
# To reproduce image 2 exactly (Photo-recon + Imputed only) use:
# METHODS = ["Photo-recon", "Imputed"]
METHODS = ["Photo-recon", "Tricubic", "Imputed"]

# The reference table reports MADRC Tricubic as "--". Keep False to reproduce it.
# Setting True computes it, but case IDs under 04_bicubic_synthseg may not align
# with the best_recon_* cases, so the paired p-values can be undefined.
INCLUDE_TRICUBIC_MADRC = True

# Which method(s) get significance stars ON THE FIGURE (table always reports both).
# Image-2 style annotates only Imputed vs Photo-recon.
ANNOTATE_METHODS_IN_FIGURE = ["Imputed"]

# Local clone of the Overleaf Git remote (see notes at bottom of this file).
OVERLEAF_REPO = None   # e.g. "/home/marina/overleaf/neuropath-paper"

# -----------------------------------------------------------------------------
REFERENCE = "Photo-recon"
DISTANCES = ["4mm", "8mm", "12mm"]
CONDITIONS = ["4mm", "8mm", "12mm", "MADRC"]          # figure grouping order

COND_TABLE_NAME = {"MADRC": "MADRC", "4mm": "UW -- 4 mm",
                   "8mm": "UW -- 8 mm", "12mm": "UW -- 12 mm"}
TABLE_COND_ORDER = ["MADRC", "4mm", "8mm", "12mm"]     # table block order

FIGURE_REGION_ORDER = ["WM", "Cortex", "Ventricle", "Thalamus", "Caudate",
                       "Putamen", "Pallidum", "Hippocampus", "Amygdala"]
CORE_REGIONS = sorted(FIGURE_REGION_ORDER)             # alphabetical, for the table

LABEL_NAMES = {
    2: "WM", 3: "Cortex", 4: "Ventricle", 10: "Thalamus", 11: "Caudate",
    12: "Putamen", 13: "Pallidum", 17: "Hippocampus", 18: "Amygdala",
    77: "WM hypo", 819: "HypoThal-noMB", 821: "Fornix", 843: "MammillaryBody",
    865: "BasalForebrain", 869: "SeptalNuc",
    41: "WM", 42: "Cortex", 43: "Ventricle", 49: "Thalamus", 50: "Caudate",
    51: "Putamen", 52: "Pallidum", 53: "Hippocampus", 54: "Amygdala",
    820: "HypoThal-noMB", 822: "Fornix", 844: "MammillaryBody",
    866: "BasalForebrain", 870: "SeptalNuc",
}

PALETTE = {
    "Photo-recon | 4mm":  "#FCFAFC", "Photo-recon | 8mm":  "#E6E6E6",
    "Photo-recon | 12mm": "#CECECE", "Photo-recon | MADRC": "#BCA8A2",
    "Imputed | 4mm":  "#F6DACC", "Imputed | 8mm":  "#CC9686",
    "Imputed | 12mm": "#9A6856", "Imputed | MADRC": "#C8624C",
    "Tricubic | 4mm":  "#E6DBF3", "Tricubic | 8mm":   "#A29ECA",
    "Tricubic | 12mm": "#776E99", "Tricubic | MADRC": "#7079CF",
}

plt.rcParams.update({"font.size": 18, "axes.labelsize": 22,
                     "xtick.labelsize": 20, "ytick.labelsize": 20,
                     "legend.fontsize": 16})


# =============================================================================
# DATA LOADING
# =============================================================================
def _read_overlap(path: str) -> pd.DataFrame:
    """Read a SynthSeg overlap file (TSV: Label, VE, Dice, Jaccard)."""
    df = pd.read_csv(path, sep="\t", header=None)
    df.columns = ["Label", "VE", "Dice", "Jaccard"]
    return df[df["Label"] != 0]


def _emit(records: list, case: str, method: str, condition: str,
          df: pd.DataFrame) -> None:
    """Append one record per (core-region) label present in df."""
    for label, region in LABEL_NAMES.items():
        if region not in CORE_REGIONS:
            continue
        hit = df[df["Label"] == label]
        if hit.empty:
            continue
        records.append({
            "Condition": condition, "Case": case, "Method": method,
            "Label": int(label), "Region": region,
            "Dice": float(hit["Dice"].iloc[0]) / 100.0,
        })


def load_uw(records: list) -> None:
    """UW cohort: three parallel subfolders sharing case-folder names."""
    sub_method = {
        "04_photo_recon_synthseg": "Photo-recon",
        "04_bicubic_synthseg": "Tricubic",
        "04_unet_synthseg": "Imputed",
    }
    
    ref_sub = "04_photo_recon_synthseg"
    ref_path = os.path.join(UW_DIR, ref_sub)
    if not os.path.isdir(ref_path):
        print(f"[UW] missing {ref_path}, skipping UW cohort")
        return
    cases = sorted(os.listdir(ref_path))
    for d in DISTANCES:
        for case in cases:
            for sub, method in sub_method.items():
                if method not in METHODS:
                    continue
                elif method == 'Photo-recon':
                    hits = glob.glob(os.path.join(UW_DIR, sub, case, f"synthseg_photo_recon*{d}.json"))
                elif method == 'Imputed':
                    hits = glob.glob(os.path.join(UW_DIR, sub, case, f"synthseg_imputed_unet*{d}.json"))
                else:
                    hits = glob.glob(os.path.join(UW_DIR, sub, case, f"dice_*{d}.txt"))
                if not hits or not os.path.exists(hits[0]):
                    continue
                _emit(records, case, method, d, _read_overlap(hits[0]))


def load_madrc(records: list) -> None:
    """MADRC cohort: PR/Imputed from overlap txt, Tricubic from dice_*.json."""
    patterns = {
        "Photo-recon": "best_recon_ss_qc_compute_overlap/*/*/photo_recon.orig.json",
        "Imputed":     "best_recon_ss_qc_compute_overlap/*/*/photo_recon.machine_learning.json",
    }
    for method, pat in patterns.items():
        if method not in METHODS:
            continue
        for f in glob.glob(os.path.join(MADRC_DIR, pat)):
            case = os.path.basename(os.path.dirname(f))
            _emit(records, case, method, "MADRC", _read_overlap(f))

    if "Tricubic" in METHODS and INCLUDE_TRICUBIC_MADRC:
        for folder in glob.glob(os.path.join(MADRC_DIR, "04_bicubic_synthseg", "*")):
            case = os.path.basename(folder)
            hits = glob.glob(os.path.join(folder, "dice_*.json"))
            if not hits:
                continue
            try:
                _emit(records, case, "Tricubic", "MADRC", _read_overlap(hits[0]))
            except Exception as exc:                       # noqa: BLE001
                print(f"[MADRC/Tricubic] skip {case}: {exc}")


def build_dataframe() -> pd.DataFrame:
    records: list = []
    load_uw(records)
    load_madrc(records)
    df = pd.DataFrame.from_records(records)
    if df.empty:
        raise RuntimeError("No records loaded. Check UW_DIR / MADRC_DIR.")
    df["Hue"] = df["Method"] + " | " + df["Condition"]
    return df


# =============================================================================
# STATISTICS
# =============================================================================
def paired_pvalue(df: pd.DataFrame, condition: str, region: str,
                  method: str, reference: str = REFERENCE) -> float:
    """Wilcoxon signed-rank of `method` vs `reference`, paired on (Case, Label)."""
    sel = (df.Condition == condition) & (df.Region == region)
    a = df[sel & (df.Method == method)][["Case", "Label", "Dice"]]
    b = df[sel & (df.Method == reference)][["Case", "Label", "Dice"]]
    if a.empty or b.empty:
        return np.nan
    merged = a.merge(b, on=["Case", "Label"], suffixes=("_m", "_ref"))
    if len(merged) < 1:
        return np.nan
    diff = merged["Dice_m"].to_numpy() - merged["Dice_ref"].to_numpy()
    if np.allclose(diff, 0.0):                              # wilcoxon undefined
        return np.nan
    try:
        _, p = wilcoxon(merged["Dice_m"], merged["Dice_ref"])
        return float(p)
    except ValueError:
        return np.nan


def stars(p: float) -> str:
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def region_mean(df: pd.DataFrame, condition: str, region: str,
                method: str) -> float:
    v = df[(df.Condition == condition) & (df.Region == region)
           & (df.Method == method)]["Dice"]
    return float(v.mean()) if len(v) else np.nan


# =============================================================================
# LATEX TABLE
# =============================================================================
def _cell(mean: float, p: float, reference: bool) -> str:
    if mean is None or (isinstance(mean, float) and np.isnan(mean)):
        return "--"
    val = f"{mean:.3f}"
    if reference:
        return val
    s = stars(p)
    return val + (r"\sym{%s}" % s if s else r"\blank")


def build_latex(df: pd.DataFrame) -> str:
    L = []
    L.append(r"\begin{table*}[h!]")
    L.append(r"\centering")
    L.append(r"\caption{")
    L.append(r"Region-specific Dice scores of automated segmentations of 3D "
             r"reconstructions of photographs before and after imputation. "
             r"Tricubic interpolation results are also shown. Gold-standard "
             r"segmentations were obtained from MRI scans. Superscripts indicate "
             r"statistically significant differences with respect to the "
             r"Photo-recon method ($^{*}p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$; "
             r"Wilcoxon signed-rank test).}")
    L.append(r"\label{tab:task2_scores}")
    L.append(r"\begin{tabular}{lccc}")
    L.append(r"\toprule")

    for ci, cond in enumerate(TABLE_COND_ORDER):
        L.append(r"\multicolumn{4}{c}{\textbf{%s}} \\" % COND_TABLE_NAME[cond])
        L.append(r"\midrule")
        if ci == 0:
            L.append(r"\textbf{Region} & \textbf{Photo-recon} & "
                     r"\textbf{Tricubic} & \textbf{Imputed} \\")
        for region in CORE_REGIONS:
            pr = _cell(region_mean(df, cond, region, "Photo-recon"),
                       None, reference=True)
            tri = _cell(region_mean(df, cond, region, "Tricubic"),
                        paired_pvalue(df, cond, region, "Tricubic"),
                        reference=False)
            imp = _cell(region_mean(df, cond, region, "Imputed"),
                        paired_pvalue(df, cond, region, "Imputed"),
                        reference=False)
            L.append("%-11s & %s & %s & %s \\\\" % (region, pr, tri, imp))
        L.append(r"\bottomrule" if ci == len(TABLE_COND_ORDER) - 1
                 else r"\midrule")

    L.append(r"\end{tabular}")
    L.append(r"\end{table*}")
    return "\n".join(L)


# =============================================================================
# FIGURE
# =============================================================================
def _upper_whisker(values: np.ndarray) -> float:
    if len(values) == 0:
        return np.nan
    q1, q3 = np.percentile(values, [25, 75])
    hi = q3 + 1.5 * (q3 - q1)
    within = values[values <= hi]
    return float(within.max()) if len(within) else float(values.max())


def _sig_bracket(ax, x1: float, x2: float, y: float, text: str,
                 h: float = 0.012) -> None:
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y],
            lw=1.0, ls=":", c="k", clip_on=False)
    ax.text((x1 + x2) / 2.0, y + h, text, ha="center", va="bottom",
            fontsize=15, clip_on=False)


def _add_legends(ax) -> None:
    def patches(method):
        out = []
        for cond, lab in (("4mm", "UW-4mm"), ("8mm", "UW-8mm"),
                          ("12mm", "UW-12mm"), ("MADRC", "MADRC")):
            key = f"{method} | {cond}"
            if key in PALETTE:
                out.append(Patch(facecolor=PALETTE[key], edgecolor="k", label=lab))
        return out

    legends = []
    specs = [("3D Reconstruction\nof slab photographs", "Photo-recon", 0.02),
             ("Tricubic", "Tricubic", 0.24),
             ("Imputed", "Imputed", 0.40)]
    for title, method, x in specs:
        if method not in METHODS:
            continue
        leg = ax.legend(handles=patches(method), title=title, loc="lower left",
                        bbox_to_anchor=(x, 0.02), title_fontsize=13,
                        fontsize=12, frameon=True)
        legends.append(leg)
    for leg in legends:
        ax.add_artist(leg)


def make_figure(df: pd.DataFrame) -> plt.Figure:
    # only hue levels that actually carry data, in condition->method order
    present = set(df["Hue"])
    hue_order = [f"{m} | {c}" for c in CONDITIONS for m in METHODS
                 if f"{m} | {c}" in present]
    n = len(hue_order)
    width = 0.65

    fig, ax = plt.subplots(figsize=(18, 8))
    sns.boxplot(data=df, x="Region", y="Dice", order=FIGURE_REGION_ORDER,
                hue="Hue", hue_order=hue_order, palette=PALETTE,
                whiskerprops=dict(linewidth=1), capprops=dict(linewidth=1),
                medianprops=dict(linewidth=1.3), dodge=True, showfliers=False,
                width=width, ax=ax)
    # ax.set_ylabel("Synthseg", fontsize=22)
    ax.set_xlabel("")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=20)
    ax.tick_params(axis="y", labelsize=20)
    ax.margins(x=0.01)

    # def box_x(region_idx: int, hue: str) -> float:
    #     i = hue_order.index(hue)
    #     return region_idx + width * ((i + 0.5) / n - 0.5)

    # for ri, region in enumerate(FIGURE_REGION_ORDER):
    #     for cond in CONDITIONS:
    #         ref_hue = f"{REFERENCE} | {cond}"
    #         if ref_hue not in hue_order:
    #             continue
    #         ref_vals = df[(df.Region == region)
    #                       & (df.Hue == ref_hue)]["Dice"].to_numpy()
    #         for method in ANNOTATE_METHODS_IN_FIGURE:
    #             hue = f"{method} | {cond}"
    #             if hue not in hue_order:
    #                 continue
    #             s = stars(paired_pvalue(df, cond, region, method))
    #             if not s:
    #                 continue
    #             m_vals = df[(df.Region == region)
    #                         & (df.Hue == hue)]["Dice"].to_numpy()
    #             y = np.nanmax([_upper_whisker(ref_vals),
    #                            _upper_whisker(m_vals)]) + 0.02
    #             _sig_bracket(ax, box_x(ri, ref_hue), box_x(ri, hue), y, s)

    _add_legends(ax)
    return fig


# =============================================================================
# OUTPUTS
# =============================================================================
def _write_audit(df: pd.DataFrame) -> str:
    rows = []
    for cond in TABLE_COND_ORDER:
        for region in CORE_REGIONS:
            rows.append({
                "Condition": cond, "Region": region,
                "Photo-recon": region_mean(df, cond, region, "Photo-recon"),
                "Tricubic": region_mean(df, cond, region, "Tricubic"),
                "Imputed": region_mean(df, cond, region, "Imputed"),
                "p_Tricubic_vs_PR": paired_pvalue(df, cond, region, "Tricubic"),
                "p_Imputed_vs_PR": paired_pvalue(df, cond, region, "Imputed"),
            })
    path = os.path.join(OUT_DIR, "task2_scores_audit.csv")
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    return path


def save_outputs(df: pd.DataFrame) -> list:
    os.makedirs(OUT_DIR, exist_ok=True)

    fig = make_figure(df)
    svg = os.path.join(OUT_DIR, "task2_combined_boxplot.svg")
    pdf = os.path.join(OUT_DIR, "task2_combined_boxplot.pdf")
    fig.savefig(svg, bbox_inches="tight", dpi=300)
    fig.savefig(pdf, bbox_inches="tight", dpi=300)
    plt.close(fig)

    tex = os.path.join(OUT_DIR, "task2_scores.tex")
    Path(tex).write_text(build_latex(df) + "\n", encoding="utf-8")

    audit = _write_audit(df)
    return [tex, pdf, svg, audit]


def push_to_overleaf(files: list, repo=OVERLEAF_REPO,
                     message="Update Task 2 table and figure") -> None:
    if not repo:
        raise SystemExit("OVERLEAF_REPO is not set; cannot push.")
    repo = os.path.abspath(repo)
    for f in files:
        shutil.copy(f, os.path.join(repo, os.path.basename(f)))
    subprocess.run(["git", "-C", repo, "add", "--all"], check=True)
    subprocess.run(["git", "-C", repo, "commit", "-m", message], check=True)
    subprocess.run(["git", "-C", repo, "push"], check=True)
    print(f"Pushed {len(files)} file(s) to {repo}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build Task 2 figure + LaTeX table.")
    ap.add_argument("--push", action="store_true",
                    help="copy outputs into OVERLEAF_REPO, git commit and push")
    args = ap.parse_args()

    df = build_dataframe()
    outputs = save_outputs(df)
    print("Wrote:")
    for f in outputs:
        print("  ", f)
    if args.push:
        push_to_overleaf(outputs)


if __name__ == "__main__":
    main()