"""
=============================================================================
PATCH BLOCK -- per-region concordance figures + per-region Pearson tables
=============================================================================

Insert this block into `build_volume_correlations_joint.py`, replacing the
existing `make_concordance_figure` and `_pearson` definitions. It requires the
module-level configuration already present in that script (METHODS,
METHOD_DISPLAY, METHOD_ABBR, DISTANCES, MADRC_LABEL, SECTION_ORDER_TABLE,
SECTION_HEADER, DISTANCE_COLORS, MADRC_COLOR, POINT_ORDER, LINE_ORDER,
PANEL_LETTERS, OUT_DIR) and the imports numpy/pandas/matplotlib/scipy.
=============================================================================
"""

from __future__ import annotations

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress



# --- placeholders so this file compiles standalone; delete on paste ----------
try:
    METHODS
except NameError:  # pragma: no cover - host script supplies these
    METHODS = ["Photo-recon", "Tricubic", "Imputed"]
    METHOD_DISPLAY = {"Photo-recon": "3D reconstruction \n of slab photographs",
                      "Tricubic": "Cubic", "Imputed": "Imputed"}
    METHOD_ABBR = {"Photo-recon": "PR", "Tricubic": "Cubic", "Imputed": "UNet"}
    DISTANCES = ["12mm", "8mm", "4mm"]
    MADRC_LABEL = "MADRC"
    SECTION_ORDER_TABLE = [MADRC_LABEL, "4mm", "8mm", "12mm"]
    SECTION_HEADER = {"MADRC": "MADRC", "4mm": "UW -- 4 mm",
                      "8mm": "UW -- 8 mm", "12mm": "UW -- 12 mm"}
    DISTANCE_COLORS = {"4mm": "#d2691e", "8mm": "#e9967a", "12mm": "#ffcba4"}
    MADRC_COLOR = "#A894EE"
    POINT_ORDER = ["4mm", "8mm", "12mm"]
    LINE_ORDER = ["y = x", "LS Fit"]
    PANEL_LETTERS = ["(a)", "(b)", "(c)"]
    OUT_DIR = None
# ----------------------------------------------------------------------------


# =============================================================================
# CONFIGURATION
# =============================================================================
# Space in which the concordance is plotted AND in which r is computed.
CORR_SPACE = "raw"                   # "raw" | "log10"

# Multiple-comparison control. Family = all valid (section x region x method)
# tests of H0: rho = 0 in the table. Set to "none" to report raw p.
CORR_CORRECTION = "bonferroni"       # "bonferroni" | "none"
CORR_ALPHA = 0.05

# Mark cells whose adjusted p falls below CORR_ALPHA. Keep False if the table
# reports r with confidence intervals only and no significance claim is made.
BOLD_SIGNIFICANT = False

# Summary row appended under the per-region rows of each section:
#   "none"        -> no summary row
#   "pooled"      -> r recomputed over all regions at once (size-range inflated)
#   "fisher_mean" -> Fisher-z average of the per-region r, weighted by (n - 3)
SUMMARY_ROW_MODE = "fisher_mean"     # "none" | "pooled" | "fisher_mean"

# Report the number of subjects in the LaTeX table. The counts are always
# retained in the CSV regardless of this setting.
SHOW_N_COLUMN = False

# Per-region figures: axis limits from that region alone (False = shared range).
REGION_SHARED_AXES = False
REGION_FIG_SUBDIR = "concordance_by_region"

ANNOT_FONTSIZE = 18

SUMMARY_ROW_LABEL = {"pooled": r"\textit{All regions (pooled)}",
                     "fisher_mean": r"\textit{Mean (Fisher $z$)}"}


# =============================================================================
# TRANSFORM / FIT HELPERS
# =============================================================================
def _tx(v) -> np.ndarray:
    """Map volumes into the correlation/plot space defined by CORR_SPACE."""
    v = np.asarray(v, dtype=float)
    return np.log10(v) if CORR_SPACE == "log10" else v


def _vol_axis_label(kind: str) -> str:
    unit = r"mm$^3$, log$_{10}$" if CORR_SPACE == "log10" else r"mm$^3$"
    return ("Pred. volume [%s]" if kind == "y" else "Ref. volume [%s]") % unit


def _fmt_sd(v: float) -> str:
    """Residual SD for the panel annotation, scaled to the current space."""
    if not np.isfinite(v):
        return "n/a"
    if CORR_SPACE == "log10":
        return f"{v:.2f}"
    a = abs(v)
    if a >= 1e4:
        return f"{v:.3g}"
    if a >= 100:
        return f"{v:.0f}"
    return f"{v:.1f}"


def fit_stats(x, y) -> dict:
    """
    OLS fit and Pearson correlation with Fisher-z 95% CI and residual SD.

    Returns keys: n, a (intercept), b (slope), r, p, rsd, ci_lo, ci_hi.
    NaN-safe: degenerate inputs (n < 3, zero variance) return NaNs rather than
    raising, so empty region/method cells propagate as "--" downstream.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = int(x.size)
    out = dict(n=n, a=np.nan, b=np.nan, r=np.nan, p=np.nan,
               rsd=np.nan, ci_lo=np.nan, ci_hi=np.nan)
    if n < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return out
    fit = linregress(x, y)
    resid = y - (fit.intercept + fit.slope * x)
    out.update(a=float(fit.intercept), b=float(fit.slope), r=float(fit.rvalue),
               p=float(fit.pvalue),
               rsd=float(np.std(resid, ddof=2)) if n > 2 else np.nan)
    if n > 3 and abs(fit.rvalue) < 1.0:
        z = np.arctanh(fit.rvalue)
        se = 1.0 / np.sqrt(n - 3.0)
        out["ci_lo"] = float(np.tanh(z - 1.96 * se))
        out["ci_hi"] = float(np.tanh(z + 1.96 * se))
    return out


# Backwards-compatible shim for the original helper.
def _pearson(x, y):
    st = fit_stats(x, y)
    return st["a"], st["b"], st["r"], st["p"]


def fisher_mean_r(rs, ns) -> dict:
    """
    Weighted Fisher-z average of independent correlations.

    z_bar = sum w_i z_i / sum w_i with w_i = n_i - 3, back-transformed by tanh.
    The interval uses SE = 1 / sqrt(sum w_i). Returns NaNs if no valid input.
    """
    rs = np.asarray(rs, dtype=float)
    ns = np.asarray(ns, dtype=float)
    ok = np.isfinite(rs) & np.isfinite(ns) & (np.abs(rs) < 1.0) & (ns > 3)
    out = dict(n=int(np.nansum(ns[ok])) if ok.any() else 0, a=np.nan, b=np.nan,
               r=np.nan, p=np.nan, rsd=np.nan, ci_lo=np.nan, ci_hi=np.nan,
               k=int(ok.sum()))
    if not ok.any():
        return out
    w = ns[ok] - 3.0
    z = np.arctanh(rs[ok])
    z_bar = float(np.sum(w * z) / np.sum(w))
    se = 1.0 / np.sqrt(np.sum(w))
    out.update(r=float(np.tanh(z_bar)),
               ci_lo=float(np.tanh(z_bar - 1.96 * se)),
               ci_hi=float(np.tanh(z_bar + 1.96 * se)))
    return out


def bonferroni(pvals) -> np.ndarray:
    """Bonferroni-adjusted p-values; preserves NaNs and input order."""
    p = np.asarray(pvals, dtype=float)
    q = np.full_like(p, np.nan)
    idx = np.where(~np.isnan(p))[0]
    m = idx.size
    if m == 0:
        return q
    q[idx] = np.clip(p[idx] * m, 0.0, 1.0)
    return q


# =============================================================================
# FIGURE PRIMITIVES
# =============================================================================
def concordance_limits(frames, pad_frac: float = 0.05):
    """Square limits covering reference and predicted volumes."""
    vals = []
    for mm in frames:
        if mm is not None and not mm.empty:
            vals.append(_tx(mm["Ref_mm3"].to_numpy()))
            vals.append(_tx(mm["Volume_mm3"].to_numpy()))
    if not vals:
        return (2.0, 5.5) if CORR_SPACE == "log10" else (0.0, 1.0)
    allv = np.concatenate(vals)
    allv = allv[np.isfinite(allv)]
    if allv.size == 0:
        return (2.0, 5.5) if CORR_SPACE == "log10" else (0.0, 1.0)
    lo, hi = float(allv.min()), float(allv.max())
    if hi <= lo:
        hi = lo + 1.0
    pad = (hi - lo) * pad_frac
    lo, hi = lo - pad, hi + pad
    if CORR_SPACE == "raw" and lo < 0.0:
        lo = 0.0
    return lo, hi


def _draw_group(ax, x, y, color, lo, hi, point_label=None, line_label=None,
                line_color=None, band=True):
    """Scatter + LS fit + 1.96*residual-SD prediction band for one subgroup."""
    st = fit_stats(x, y)
    ax.scatter(x, y, s=36, alpha=0.55, color=color, edgecolors="none",
               label=point_label, zorder=3)
    if np.isfinite(st["b"]):
        xg = np.linspace(lo, hi, 100)
        yg = st["a"] + st["b"] * xg
        ax.plot(xg, yg, color=line_color or color, lw=1.6, alpha=0.85,
                zorder=4, label=line_label)
        if band and np.isfinite(st["rsd"]):
            ax.fill_between(xg, yg - 1.96 * st["rsd"], yg + 1.96 * st["rsd"],
                            color=line_color or color, alpha=0.10, zorder=2)
    return st


def _annotate(ax, lines):
    for i, txt in enumerate(lines):
        ax.text(0.04, 0.96 - 0.065 * i, txt, transform=ax.transAxes,
                fontsize=ANNOT_FONTSIZE, va="top", color="#333333")


def _ordered_legend(axes):
    seen = {}
    for ax in np.ravel(axes):
        handles, labels = ax.get_legend_handles_labels()
        for h, l in zip(handles, labels):
            if l and not l.startswith("_"):
                seen.setdefault(l, h)
    known = POINT_ORDER + LINE_ORDER
    labels = [l for l in known if l in seen] + [l for l in seen if l not in known]
    return [seen[l] for l in labels], labels


# =============================================================================
# CONCORDANCE FIGURE (pooled or per-region)
# =============================================================================
def make_concordance_figure(m_uw: pd.DataFrame, m_mad: pd.DataFrame,
                            region: str | None = None,
                            lims: tuple | None = None,
                            suptitle: str | None = None):
    """
    Per-cohort, per-method concordance of measured vs reference volume, with
    identity line, LS fit and +/-1.96*residual-SD prediction band.

    region=None  -> pooled over all regions (the original summary figure).
    region="WM"  -> restricted to that region; r is then a purely between-
                    subject quantity and is not inflated by region size range.

    UW metrics are computed per slab distance; MADRC over the whole cohort.
    Returns (figure, stats_dataframe).
    """
    rows = [("UW", m_uw, True), ("MADRC", m_mad, False)]
    if region is not None:
        rows = [(coh, mm[mm["Label"] == region].copy() if not mm.empty else mm,
                 by_dist) for coh, mm, by_dist in rows]

    lo, hi = lims if lims is not None else concordance_limits([mm for _, mm, _ in rows])

    fig, axes = plt.subplots(2, 3, figsize=(18, 12), sharex=True, sharey=True)
    records = []

    for r, (coh, mm, by_dist) in enumerate(rows):
        for c, method in enumerate(METHODS):
            ax = axes[r, c]
            ax.plot([lo, hi], [lo, hi], ls="--", color="black", lw=1.4,
                    alpha=0.6, zorder=1, label="y = x")

            sub = mm[mm["Method"] == method] if not mm.empty else mm
            if by_dist:
                groups = [(d, sub[sub["Distance"] == d] if not sub.empty else sub,
                           DISTANCE_COLORS.get(d, "#7f7f7f"), d, None)
                          for d in DISTANCES]
            else:
                groups = [(MADRC_LABEL, sub, MADRC_COLOR, None, "LS Fit")]

            annot = []
            for gname, dd, color, plabel, llabel in groups:
                if dd is None or dd.empty:
                    continue
                st = _draw_group(ax, _tx(dd["Ref_mm3"]), _tx(dd["Volume_mm3"]),
                                 color, lo, hi, point_label=plabel,
                                 line_label=llabel,
                                 line_color=None if by_dist else "#463F61")
                prefix = f"{gname}: " if by_dist else ""
                if np.isfinite(st["r"]):
                    annot.append(f"{prefix}r = {st['r']:.2f}, "
                                 f"SD$_r$ = {_fmt_sd(st['rsd'])}")
                else:
                    annot.append(f"{prefix}r undefined")
                records.append(dict(Cohort=coh, Section=gname, Method=method,
                                    Region=region if region else "Pooled", **st))
            # _annotate(ax, annot)

            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, alpha=0.25, ls=":", which="both")
            if r == 0:
                ax.set_title(f"{PANEL_LETTERS[c]} {METHOD_DISPLAY[method]}",
                             fontsize=20, pad=8)
            if c == 0:
                ax.set_ylabel(f"{coh}\n{_vol_axis_label('y')}", fontsize=20)
            if r == 1:
                ax.set_xlabel(_vol_axis_label("x"), fontsize=20)

    handles, labels = _ordered_legend(axes)
    if handles:
        axes[0, -1].legend(handles, labels, fontsize=20, loc="lower right",
                           handlelength=1.2, handletextpad=0.5)

    if suptitle:
        fig.suptitle(suptitle, fontsize=22, y=0.98)
    fig.subplots_adjust(wspace=0.02, hspace=0.08, left=0.11,
                        top=0.93 if suptitle else 0.96)
    return fig, pd.DataFrame.from_records(records)


def _slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(name)).strip("_").lower()


def make_region_concordance_figures(m_uw: pd.DataFrame, m_mad: pd.DataFrame,
                                    labels, out_dir: str) -> tuple:
    """One concordance figure per region. Returns (paths, stats_dataframe)."""
    sub_dir = os.path.join(out_dir, REGION_FIG_SUBDIR)
    os.makedirs(sub_dir, exist_ok=True)
    shared = concordance_limits([m_uw, m_mad]) if REGION_SHARED_AXES else None

    paths, frames = [], []
    for lab in labels:
        fig, st = make_concordance_figure(m_uw, m_mad, region=lab, lims=shared,
                                          suptitle=lab)
        if st.empty:
            plt.close(fig)
            print(f"[concordance] no data for region {lab}, skipped")
            continue
        stem = os.path.join(sub_dir, f"concordance_{_slug(lab)}")
        for ext in ("pdf", "svg"):
            path = f"{stem}.{ext}"
            fig.savefig(path, bbox_inches="tight", dpi=300)
            paths.append(path)
        plt.close(fig)
        frames.append(st)
    stats = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return paths, stats


# =============================================================================
# PER-REGION PEARSON CORRELATION TABLE
# =============================================================================
def compute_pearson_table(m_all: pd.DataFrame, labels) -> pd.DataFrame:
    """
    Long-format per (section, region, method) Pearson statistics.

    Sections are MADRC and the three UW slab distances, so each row is a
    within-cohort, within-region, within-distance correlation across subjects.
    p-values test H0: rho = 0 and are corrected across the per-region rows only;
    the summary row is derived from those rows and is excluded from the family.
    """
    present = [s for s in SECTION_ORDER_TABLE if s in set(m_all["Distance"])]
    region_list = list(labels)

    rows = []
    for sec in present:
        d_sec = m_all[m_all["Distance"] == sec]
        for lab in region_list:
            d_lab = d_sec[d_sec["Label"] == lab]
            for method in METHODS:
                d = d_lab[d_lab["Method"] == method]
                st = fit_stats(_tx(d["Ref_mm3"]), _tx(d["Volume_mm3"]))
                rows.append(dict(
                    Section=sec, SectionLabel=SECTION_HEADER.get(sec, sec),
                    Region=lab, Method=method, RowType="region",
                    n_obs=st["n"],
                    n_subjects=int(d["Subject"].nunique()) if len(d) else 0,
                    **{k: st[k] for k in
                       ("r", "ci_lo", "ci_hi", "p", "b", "a", "rsd")}))

    df = pd.DataFrame(rows)

    if SUMMARY_ROW_MODE != "none":
        summary = []
        for sec in present:
            d_sec = m_all[m_all["Distance"] == sec]
            for method in METHODS:
                if SUMMARY_ROW_MODE == "pooled":
                    d = d_sec[d_sec["Method"] == method]
                    st = fit_stats(_tx(d["Ref_mm3"]), _tx(d["Volume_mm3"]))
                    n_subj = int(d["Subject"].nunique()) if len(d) else 0
                else:  # fisher_mean
                    sel = df[(df["Section"] == sec) & (df["Method"] == method)]
                    st = fisher_mean_r(sel["r"], sel["n_obs"])
                    st = {k: st.get(k, np.nan) for k in
                          ("n", "a", "b", "r", "p", "rsd", "ci_lo", "ci_hi")}
                    n_subj = int(sel["n_subjects"].max()) if len(sel) else 0
                summary.append(dict(
                    Section=sec, SectionLabel=SECTION_HEADER.get(sec, sec),
                    Region="Summary", Method=method, RowType=SUMMARY_ROW_MODE,
                    n_obs=st["n"], n_subjects=n_subj,
                    **{k: st[k] for k in
                       ("r", "ci_lo", "ci_hi", "p", "b", "a", "rsd")}))
        df = pd.concat([df, pd.DataFrame(summary)], ignore_index=True)

    mask_family = df["RowType"] == "region"
    p_family = df["p"].where(mask_family, np.nan).to_numpy()
    df["p_adj"] = bonferroni(p_family) if CORR_CORRECTION == "bonferroni" else p_family
    df["m_tests"] = int(np.count_nonzero(~np.isnan(p_family)))
    return df


def _fmt_r(row) -> str:
    """One table cell: r with its Fisher 95% interval."""
    if not np.isfinite(row["r"]):
        return "--"
    cell = f"{row['r']:.2f}"
    if np.isfinite(row["ci_lo"]) and np.isfinite(row["ci_hi"]):
        cell += r"\,[%.2f, %.2f]" % (row["ci_lo"], row["ci_hi"])
    if BOLD_SIGNIFICANT:
        q = row.get("p_adj", np.nan)
        if np.isfinite(q) and q < CORR_ALPHA:
            cell = r"\textbf{%s}" % cell
    return cell


def _pearson_caption(m_tests: int) -> str:
    space_txt = (r"$\log_{10}$-transformed volumes" if CORR_SPACE == "log10"
                 else r"untransformed volumes")

    if CORR_CORRECTION == "bonferroni":
        test_txt = (r" Tests of $H_0\!:\rho=0$ are Bonferroni-adjusted across "
                    r"the $m=%d$ region-level comparisons in this table" % m_tests)
        test_txt += (r"; entries with $p_{\mathrm{adj}}<%.2f$ are shown in bold."
                     % CORR_ALPHA) if BOLD_SIGNIFICANT else \
                    (r" and are reported in the accompanying CSV.")
    else:
        test_txt = r" Uncorrected $p$-values are reported in the accompanying CSV."

    if SUMMARY_ROW_MODE == "fisher_mean":
        summary_txt = (r" The final row of each section is the Fisher $z$ "
                       r"average of the region-level coefficients above it, "
                       r"weighted by $n-3$, and back-transformed; it is not an "
                       r"arithmetic mean of $r$.")
    elif SUMMARY_ROW_MODE == "pooled":
        summary_txt = (r" The final row of each section recomputes $r$ over all "
                       r"regions at once. It is upwardly biased by the range of "
                       r"structure sizes and is not comparable with the "
                       r"region-level values above it.")
    else:
        summary_txt = ""

    abbr = ", ".join("%s = %s" % (METHOD_ABBR[m_], m_) for m_ in METHODS)

    return (
        r"Region-specific Pearson correlation between automated volumes from 3D "
        r"reconstructions and gold-standard MRI volumes, by cohort, slab "
        r"distance and reconstruction method. Each cell reports $r$ with its "
        r"Fisher $z$ 95\% confidence interval, computed across subjects on " +
        space_txt + r" within a single region, so it is not inflated by the "
        r"between-region range of structure sizes." + test_txt + summary_txt +
        r" Dashes indicate cells with fewer than three paired observations or "
        r"zero variance. Abbreviations: " + abbr + r"."
    )


def build_pearson_latex(df: pd.DataFrame, labels) -> str:
    """LaTeX table: rows = regions, columns = methods, cells = r [95% CI]."""
    m_tests = int(df["m_tests"].iloc[0]) if len(df) else 0

    n_lead = 1 if SHOW_N_COLUMN else 0
    ncol = 1 + n_lead + len(METHODS)
    colspec = "l" + ("c" if SHOW_N_COLUMN else "") + "c" * len(METHODS)

    header = r"\textbf{Region}"
    if SHOW_N_COLUMN:
        header += r" & \textbf{n}"
    header += "".join(r" & \textbf{%s}" % METHOD_ABBR[m_] for m_ in METHODS)
    header += r" \\"

    present = [s for s in SECTION_ORDER_TABLE if s in set(df["Section"])]
    row_order = list(labels) + (["Summary"] if SUMMARY_ROW_MODE != "none" else [])

    L = [r"\begin{table*}[h!]", r"\centering",
         r"\caption{%s}" % _pearson_caption(m_tests),
         r"\label{app:volume_pearson_joint}",
         r"\begin{tabular}{%s}" % colspec, r"\toprule"]

    for si, sec in enumerate(present):
        L.append(r"\multicolumn{%d}{c}{\textbf{%s}} \\"
                 % (ncol, SECTION_HEADER.get(sec, sec)))
        L.append(r"\midrule")
        if si == 0:
            L.append(header)
        d_sec = df[df["Section"] == sec]
        for lab in row_order:
            d_lab = d_sec[d_sec["Region"] == lab]
            if d_lab.empty:
                continue
            if lab == "Summary":
                L.append(r"\cmidrule(lr){1-%d}" % ncol)
                name = SUMMARY_ROW_LABEL[SUMMARY_ROW_MODE]
            else:
                name = lab
            cells = []
            for method in METHODS:
                rr = d_lab[d_lab["Method"] == method]
                cells.append(_fmt_r(rr.iloc[0]) if len(rr) else "--")
            fields = [name]
            if SHOW_N_COLUMN:
                fields.append(str(int(d_lab["n_subjects"].max())))
            fields.extend(cells)
            L.append("%-24s & %s \\\\" % (fields[0], " & ".join(fields[1:])))
        L.append(r"\bottomrule" if si == len(present) - 1 else r"\midrule")

    L += [r"\end{tabular}", r"\end{table*}"]
    return "\n".join(L)


def write_pearson_outputs(m_all: pd.DataFrame, labels, out_dir: str,
                          df: pd.DataFrame = None) -> list:
    """Compute, write the LaTeX table and the long-format CSV. Returns paths."""
    if df is None:
        df = compute_pearson_table(m_all, labels)
    csv_path = os.path.join(out_dir, "volume_pearson_by_region.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    tex_path = os.path.join(out_dir, "volume_pearson_table_joint.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(build_pearson_latex(df, labels) + "\n")

    if len(df):
        m_tests = int(df["m_tests"].iloc[0])
        print(f"[pearson] {CORR_CORRECTION} over m = {m_tests} region-level tests; "
              f"correlation space = {CORR_SPACE}; summary row = {SUMMARY_ROW_MODE}")
        thin = df[(df["RowType"] == "region") & (df["n_obs"].between(1, 5))]
        if len(thin):
            print(f"[pearson] warning: {len(thin)} cell(s) with n < 6; the "
                  f"Fisher interval is wide and r is unstable there.")
        if not SHOW_N_COLUMN:
            print("[pearson] note: n is omitted from the LaTeX table and "
                  "retained in volume_pearson_by_region.csv.")
    return [tex_path, csv_path]


# =============================================================================
# MAIN() WIRING -- replace the concordance block in the host script with this
# =============================================================================
def concordance_outputs(m_uw, m_mad, m_all, labels, out_dir) -> list:
    outputs = []

    # 1. Summary (pooled over regions) concordance figure.
    fig, _ = make_concordance_figure(m_uw, m_mad, region=None)
    for ext in ("pdf", "svg"):
        path = os.path.join(out_dir, f"task_2.1_uwmadrc_volume_concordance.{ext}")
        fig.savefig(path, bbox_inches="tight", dpi=300)
        outputs.append(path)
    plt.close(fig)

    # 2. One concordance figure per region.
    region_paths, _ = make_region_concordance_figures(m_uw, m_mad, labels, out_dir)
    outputs += region_paths

    # 3. Per-region Pearson correlation table (LaTeX + CSV), computed once.
    pearson = compute_pearson_table(m_all, labels)
    outputs += write_pearson_outputs(m_all, labels, out_dir, df=pearson)

    # 4. Forest plot of the region-level correlations, one figure per section.
    outputs += make_forest_figures(pearson, labels, out_dir)
    return outputs


# =============================================================================
# FOREST PLOT OF REGION-LEVEL CORRELATIONS
# =============================================================================
# Row order across the method panels. "mean_r" sorts by the Fisher mean taken
# across methods and sections so the rows align and the panels stay comparable;
# "alpha" keeps the alphabetical region order; "given" preserves `labels`.
FOREST_SORT = "mean_r"               # "mean_r" | "alpha" | "given"
FOREST_SHOW_VALUES = False           # print r [CI] beside each row
FOREST_ROW_HEIGHT = 0.46             # inches per region row
FOREST_FIG_SUBDIR = "forest_by_cohort"

# Confidence intervals are drawn recessive so the point estimates carry the
# figure: neutral grey, thinner and semi-transparent.
FOREST_CI_COLOR = "0.45"
FOREST_CI_ALPHA = 0.45
FOREST_CI_LW = 1.3

FOREST_MARKER = "o"
FOREST_MARKER_SIZE = 10.0
# Thin dark rim so the palest slab-distance colour stays legible on white.
FOREST_MARKER_EDGE = "0.05"
FOREST_MARKER_EDGE_LW = 0.05
FOREST_SUMMARY_MARKER = "D"          # diamond, same size as the region dots
FOREST_SUMMARY_SIZE = 10.0

# How the horizontal line of a multi-section row is drawn:
#   "estimates"    -> one line per region spanning the point estimates, all
#                     markers on that line, distinguished only by colour.
#                     Confidence intervals are not shown (they remain in the
#                     CSV and the LaTeX table).
#   "ci_envelope"  -> one line spanning min(ci_lo) to max(ci_hi) across the
#                     sections. Reads as a single interval, which it is not.
#   "per_section"  -> one interval per section, vertically dodged by
#                     FOREST_DODGE. Keeps every interval, at the cost of
#                     three lines per region row.
# Single-section figures (MADRC) always draw the Fisher interval.
FOREST_LINE_MODE = "estimates"       # "estimates" | "ci_envelope" | "per_section"

# Vertical offset between sections, used by "per_section" only.
FOREST_DODGE = 0.26

# One figure per cohort. Each entry maps a figure title to the sections drawn
# on the same rows; UW overlays its three slab distances, MADRC has one.
FOREST_GROUPS = [("MADRC", [MADRC_LABEL]), ("UW", ["4mm", "8mm", "12mm"])]

# Combine both cohorts into a single figure (rows = cohort, columns = method),
# mirroring the layout of the concordance figure. UW is placed first to match.
FOREST_JOINT = True
FOREST_JOINT_GROUPS = [("UW", ["4mm", "8mm", "12mm"]), ("MADRC", [MADRC_LABEL])]

# In the joint figure the MADRC row has a single section, so a line there would
# be a confidence interval while the UW line spans slab-distance estimates.
# "none" keeps one meaning per graphical element; "ci" restores the interval.
FOREST_JOINT_MADRC_LINE = "none"     # "none" | "ci"


def _section_color(section: str) -> str:
    """Match the forest plot to the concordance figures' colour coding."""
    if section == MADRC_LABEL:
        return MADRC_COLOR
    return DISTANCE_COLORS.get(section, "#7f7f7f")


def _summary_tick_label() -> str:
    return (SUMMARY_ROW_LABEL[SUMMARY_ROW_MODE]
            .replace(r"\textit{", "").replace("}", "").replace("$z$", "z"))


def _forest_row_order(d: pd.DataFrame, labels, sort: str) -> list:
    """Row order, shared by every panel and every section in one figure."""
    if sort == "alpha":
        return sorted(labels, key=str.lower)
    if sort == "given":
        return list(labels)
    key = {}
    for lab in labels:
        sel = d[(d["Region"] == lab) & (d["RowType"] == "region")]
        st = fisher_mean_r(sel["r"], sel["n_obs"])
        key[lab] = st["r"] if np.isfinite(st["r"]) else -np.inf
    return sorted(labels, key=lambda l: key[l])      # lowest at the bottom


def _dodges(k: int, mode: str) -> np.ndarray:
    """Vertical offsets so the first listed section sits at the top of a row."""
    if k <= 1 or mode != "per_section":
        return np.zeros(k if k > 0 else 1)
    return ((k - 1) / 2.0 - np.arange(k)) * FOREST_DODGE


def _row_span(rows: list, mode: str):
    """
    Horizontal extent of one region row, given its per-section statistics.

    rows is a list of (r, ci_lo, ci_hi) tuples, one per section present.
    Returns (lo, hi) or None when the row cannot be drawn as a single line.
    """
    if not rows:
        return None
    if mode == "ci_envelope":
        lo = [t[1] for t in rows if np.isfinite(t[1])]
        hi = [t[2] for t in rows if np.isfinite(t[2])]
        return (min(lo), max(hi)) if lo and hi else None
    vals = [t[0] for t in rows if np.isfinite(t[0])]
    if len(vals) < 2:
        return None
    return min(vals), max(vals)


def _draw_line(ax, lo_, hi_, y):
    """The recessive grey line of one region row."""
    if lo_ is None or not (np.isfinite(lo_) and np.isfinite(hi_)):
        return
    ax.plot([lo_, hi_], [y, y], color=FOREST_CI_COLOR, lw=FOREST_CI_LW,
            alpha=FOREST_CI_ALPHA, solid_capstyle="butt", zorder=2)


def _draw_marker(ax, r_, y, color, marker, size):
    if np.isfinite(r_):
        ax.plot([r_], [y], marker, color=color, ms=size,
                mec=FOREST_MARKER_EDGE, mew=FOREST_MARKER_EDGE_LW, zorder=3)


def _paint_forest_panel(ax, d_m, present, order, y_pos, y_sum, has_summary,
                        line_mode, offsets, show_values, madrc_line):
    """Draw one method panel for one cohort row. Returns nothing."""
    ax.axvline(0.0, ls="--", lw=1.0, color="0.55", alpha=0.8, zorder=1)
    if has_summary:
        ax.axhline(-0.75, color="0.8", lw=0.8, zorder=1)

    rows_to_draw = [(lab, y_pos[lab], FOREST_MARKER, FOREST_MARKER_SIZE)
                    for lab in order]
    if has_summary:
        rows_to_draw.append(("Summary", y_sum, FOREST_SUMMARY_MARKER,
                             FOREST_SUMMARY_SIZE))

    single = len(present) == 1

    for lab, y, marker, size in rows_to_draw:
        stats = []
        for sec in present:
            row = d_m[(d_m["Section"] == sec) & (d_m["Region"] == lab)]
            stats.append(None if row.empty else
                         (float(row["r"].iloc[0]), float(row["ci_lo"].iloc[0]),
                          float(row["ci_hi"].iloc[0])))

        if single:
            if madrc_line == "ci" and stats[0] is not None:
                _draw_line(ax, stats[0][1], stats[0][2], y)
        elif line_mode == "per_section":
            for si, t in enumerate(stats):
                if t is not None:
                    _draw_line(ax, t[1], t[2], y + offsets[si])
        else:
            span = _row_span([t for t in stats if t is not None], line_mode)
            if span is not None:
                _draw_line(ax, span[0], span[1], y)

        for si, t in enumerate(stats):
            if t is None:
                continue
            _draw_marker(ax, t[0], y + offsets[si], _section_color(present[si]),
                         marker, size)
            if show_values and single and np.isfinite(t[0]):
                txt = (f"{t[0]:.2f} [{t[1]:.2f}, {t[2]:.2f}]"
                       if np.isfinite(t[1]) else f"{t[0]:.2f}")
                ax.text(1.02, y, txt, fontsize=10, va="center",
                        transform=ax.get_yaxis_transform(), clip_on=False)


def _forest_axis_cosmetics(ax, order, y_pos, y_sum, has_summary, n_rows, x_lo):
    ax.set_xlim(x_lo, 1.03)
    ax.set_ylim(y_sum - 0.9, n_rows - 0.4)
    ticks = list(range(n_rows)) + ([y_sum] if has_summary else [])
    names = order + ([_summary_tick_label()] if has_summary else [])
    ax.set_yticks(ticks)
    ax.set_yticklabels(names, fontsize=20)
    ax.grid(True, axis="x", alpha=0.25, ls=":")
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=20)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _x_floor(d: pd.DataFrame) -> float:
    finite = d[["r", "ci_lo", "ci_hi"]].to_numpy(dtype=float)
    finite = finite[np.isfinite(finite)]
    x_lo = max(-1.0, float(np.min(finite)) - 0.08) if finite.size else -1.0
    return min(x_lo, -0.05)


def make_forest_figure(df: pd.DataFrame, sections, labels, title: str = None,
                       sort: str = None, show_values: bool = None,
                       line_mode: str = None):
    """
    Single-cohort forest: one panel per method, one row per region.

    `sections` is a list. With one entry each region carries its Fisher
    interval; with several (the UW slab distances) all sections share one row
    and are distinguished only by colour, the line spanning them as set by
    FOREST_LINE_MODE.
    """
    sort = FOREST_SORT if sort is None else sort
    show_values = FOREST_SHOW_VALUES if show_values is None else show_values
    line_mode = FOREST_LINE_MODE if line_mode is None else line_mode

    d_all = df[df["Section"].isin(sections)]
    if d_all.empty:
        return None
    present = [s for s in sections if s in set(d_all["Section"])]
    order = _forest_row_order(d_all, labels, sort)
    offsets = _dodges(len(present), line_mode)
    has_summary = bool((d_all["Region"] == "Summary").any())

    y_pos = {lab: i for i, lab in enumerate(order)}
    y_sum, n_rows = -1.5, len(order)
    height = max(4.2, FOREST_ROW_HEIGHT * (n_rows + 3))

    fig, axes = plt.subplots(1, len(METHODS), figsize=(5.6 * len(METHODS), height),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    x_lo = _x_floor(d_all)

    for c, method in enumerate(METHODS):
        ax = axes[c]
        _paint_forest_panel(ax, d_all[d_all["Method"] == method], present, order,
                            y_pos, y_sum, has_summary, line_mode, offsets,
                            show_values, "ci")
        _forest_axis_cosmetics(ax, order, y_pos, y_sum, has_summary, n_rows, x_lo)
        ax.set_title(f"{PANEL_LETTERS[c]} {METHOD_DISPLAY[method]}",
                     fontsize=20, pad=8)
        ax.set_xlabel("Pearson r", fontsize=20)

    bottom = 0.0
    if len(present) > 1:
        for sec in present:
            axes[0].plot([], [], FOREST_MARKER, color=_section_color(sec),
                         ms=FOREST_MARKER_SIZE, mec=FOREST_MARKER_EDGE,
                         mew=FOREST_MARKER_EDGE_LW, label=sec, ls="none")
        handles, names = axes[0].get_legend_handles_labels()
        bottom = 0.9 / height
        fig.legend(handles, names, loc="lower center", ncol=len(present),
                   frameon=False, fontsize=20, title="Slab distance",
                   title_fontsize=20, bbox_to_anchor=(0.5, 0.0))

    if title:
        fig.suptitle(title, fontsize=20, y=0.99)
    fig.tight_layout(rect=[0, bottom, 1, 0.96 if title else 1.0])
    return fig


def make_joint_forest_figure(df: pd.DataFrame, labels, groups=None,
                             sort: str = None, line_mode: str = None):
    """
    Both cohorts in one figure: rows = cohort, columns = method, mirroring the
    layout of the concordance figure.

    The region ordering is computed once over every cohort, section and method,
    so the rows align across all six panels.

    Note that the grey line means different things in the two rows unless
    FOREST_JOINT_MADRC_LINE is "none": in the UW row it spans the slab-distance
    estimates, in the MADRC row it is a confidence interval. The default is
    "none" so that one graphical element carries one meaning.
    """
    groups = FOREST_JOINT_GROUPS if groups is None else groups
    sort = FOREST_SORT if sort is None else sort
    line_mode = FOREST_LINE_MODE if line_mode is None else line_mode

    rows = []
    for cohort, sections in groups:
        present = [s for s in sections if s in set(df["Section"])]
        if present:
            rows.append((cohort, present))
    if not rows:
        return None

    all_sections = [s for _, secs in rows for s in secs]
    d_all = df[df["Section"].isin(all_sections)]
    order = _forest_row_order(d_all, labels, sort)
    has_summary = bool((d_all["Region"] == "Summary").any())
    y_pos = {lab: i for i, lab in enumerate(order)}
    y_sum, n_rows = -1.5, len(order)
    x_lo = _x_floor(d_all)

    panel_h = max(4.2, FOREST_ROW_HEIGHT * (n_rows + 3))
    fig, axes = plt.subplots(len(rows), len(METHODS),
                             figsize=(5.6 * len(METHODS), panel_h * len(rows)),
                             sharex=True, sharey=True, squeeze=False)

    for r, (cohort, present) in enumerate(rows):
        d_coh = d_all[d_all["Section"].isin(present)]
        offsets = _dodges(len(present), line_mode)
        for c, method in enumerate(METHODS):
            ax = axes[r][c]
            _paint_forest_panel(ax, d_coh[d_coh["Method"] == method], present,
                                order, y_pos, y_sum, has_summary, line_mode,
                                offsets, False, FOREST_JOINT_MADRC_LINE)
            _forest_axis_cosmetics(ax, order, y_pos, y_sum, has_summary,
                                   n_rows, x_lo)
            if r == 0:
                ax.set_title(f"{PANEL_LETTERS[c]} {METHOD_DISPLAY[method]}",
                             fontsize=20, pad=8)
            if r == len(rows) - 1:
                ax.set_xlabel("Pearson r", fontsize=20)
            if c == 0:
                ax.set_ylabel(cohort, fontsize=20, labelpad=12)

    # One legend for every section drawn anywhere in the figure.
    proxy_ax = axes[0][0]
    for _, present in rows:
        for sec in present:
            proxy_ax.plot([], [], FOREST_MARKER, color=_section_color(sec),
                          ms=FOREST_MARKER_SIZE, mec=FOREST_MARKER_EDGE,
                          mew=FOREST_MARKER_EDGE_LW, ls="none",
                          label=SECTION_HEADER.get(sec, sec))
    handles, names = proxy_ax.get_legend_handles_labels()
    total_h = panel_h * len(rows)
    fig.legend(handles, names, loc="lower center", ncol=len(names),
               frameon=False, fontsize=20, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=[0, 0.8 / total_h, 1, 1.0])
    return fig


def make_forest_figures(df: pd.DataFrame, labels, out_dir: str) -> list:
    """
    Forest outputs. FOREST_JOINT controls whether the cohorts are combined into
    one figure or written separately.
    """
    sub_dir = os.path.join(out_dir, FOREST_FIG_SUBDIR)
    os.makedirs(sub_dir, exist_ok=True)
    paths = []

    def _save(fig, stem):
        for ext in ("pdf", "svg"):
            path = os.path.join(sub_dir, f"{stem}.{ext}")
            fig.savefig(path, bbox_inches="tight", dpi=300)
            paths.append(path)
        plt.close(fig)

    if FOREST_JOINT:
        fig = make_joint_forest_figure(df, labels)
        if fig is not None:
            _save(fig, "task_2.1_uwmadrc_volcorr_forest")
        return paths

    for title, sections in FOREST_GROUPS:
        present = [s for s in sections if s in set(df["Section"])]
        if not present:
            continue
        fig = make_forest_figure(df, present, labels, title=title)
        if fig is not None:
            _save(fig, f"forest_r_{_slug(title)}")
    return paths