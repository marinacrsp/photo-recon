"""
=============================================================================
VOLUME METRIC TABLES -- MAE, Pearson r, and the joint variant
=============================================================================

Drop-in module for `build_volume_correlations_joint.py`. Emits three LaTeX
tables from a single computation pass, so the values cannot diverge between
them:

    volume_error_table.tex      rows = regions, cols = methods, cells = MAE
    volume_pearson_table.tex    rows = regions, cols = methods, cells = r
    volume_joint_table.tex      rows = regions, cols = method x metric

plus one long-format CSV holding every quantity, including those omitted from
the compact joint layout.

Requires the module-level configuration already present in the host script:
METHODS, METHOD_ABBR, SECTION_ORDER_TABLE, SECTION_HEADER. If the per-region
Pearson patch is already applied, delete `fit_stats` and `fisher_mean_r` from
one of the two files rather than keeping duplicate definitions.

Expected columns of m_all: Subject, Label, Method, Distance, Volume_mm3, Ref_mm3.
=============================================================================
"""

from __future__ import annotations

import os
import itertools
import numpy as np
import pandas as pd
from scipy.stats import linregress


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
CELL_STYLE = "pm"
CELL_PREC = 3 
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
#   "mae_ml"      -> mean |V_pred - V_ref| in mL. Dominated by large structures
#                    and by the occasional segmentation failure.
#   "medae_ml"    -> median absolute error in mL. Robust to failures.
#   "marvd"       -> mean |V_pred - V_ref| / V_ref, dimensionless.
#   "medarvd"     -> median of the same, the convention used elsewhere in the
#                    manuscript for skewed distributions.
ERROR_METRIC = "mae_ml"

# Dispersion printed beside the point estimate.
#   "auto" -> SD for mean-based metrics, IQR for median-based ones
#   "none" -> point estimate only, the most compact option
ERROR_DISPERSION = "auto"            # "auto" | "none" | "sd" | "iqr"

# --- correlation -------------------------------------------------------------
# Space in which r is computed. Must match the space used in the figures.
CORR_SPACE = "raw"                   # "raw" | "log10"
PEARSON_CI = True                    # append the Fisher z 95% interval

# --- summary rows ------------------------------------------------------------
# Error: arithmetic mean or median of the region-level values.
# Correlation: Fisher z average weighted by n - 3, never an arithmetic mean.
SUMMARY_ROW = True

# --- joint table -------------------------------------------------------------
#   "grouped" -> six numeric columns, MAE block then r block, with a rule
#                between them. Point estimates only.
#   "paired"  -> three columns, each cell "MAE (r)". Narrowest, densest.
JOINT_LAYOUT = "grouped"             # "grouped" | "paired"

# Mark the best method per region. No significance test backs this, so it is a
# descriptive aid only and the caption says so.
BOLD_BEST = False

# Report the number of specimens per region as a leading column.
SHOW_N_COLUMN = False

MIN_N = 3                            # below this a cell is a dash

ERROR_DISPLAY = {
    "mae_ml": (r"mean absolute volume error, "
               r"$\frac{1}{n}\sum_i |V^{(i)}_{\mathrm{pred}} - "
               r"V^{(i)}_{\mathrm{ref}}|$, in mL", "MAE [mL]"),
    "medae_ml": (r"median absolute volume error in mL", "MedAE [mL]"),
    "marvd": (r"mean absolute relative volume difference, "
              r"$\frac{1}{n}\sum_i |V^{(i)}_{\mathrm{pred}} - "
              r"V^{(i)}_{\mathrm{ref}}| / V^{(i)}_{\mathrm{ref}}$", "MARVD"),
    "medarvd": (r"median absolute relative volume difference", "MedARVD"),
}
ERROR_IS_MEAN = {"mae_ml": True, "medae_ml": False,
                 "marvd": True, "medarvd": False}
ERROR_DECIMALS = {"mae_ml": 1, "medae_ml": 1, "marvd": 3, "medarvd": 3}

# =============================================================================
# STATISTICS
# =============================================================================
def _tx(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    return np.log10(v) if CORR_SPACE == "log10" else v


def _abs_errors(pred, ref) -> np.ndarray:
    """Per-specimen absolute error in the units selected by ERROR_METRIC."""
    pred = np.asarray(pred, dtype=float)
    ref = np.asarray(ref, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        if ERROR_METRIC in ("mae_ml", "medae_ml"):
            e = np.abs(pred - ref) / 1000.0          # mm^3 -> mL
        else:
            e = np.abs(pred - ref) / ref
            e = np.where(ref > 0, e, np.nan)
    return e[np.isfinite(e)]


def error_stats(pred, ref) -> dict:
    """Point estimate and dispersion of the volume error for one cell."""
    e = _abs_errors(pred, ref)
    out = dict(n_err=int(e.size), err=np.nan, err_lo=np.nan, err_hi=np.nan)
    if e.size < MIN_N:
        return out
    if ERROR_IS_MEAN[ERROR_METRIC]:
        out.update(err=float(np.mean(e)),
                   err_lo=float(np.mean(e) - np.std(e, ddof=1)),
                   err_hi=float(np.mean(e) + np.std(e, ddof=1)))
    else:
        out.update(err=float(np.median(e)),
                   err_lo=float(np.percentile(e, 25)),
                   err_hi=float(np.percentile(e, 75)))
    return out


def fit_stats(x, y) -> dict:
    """
    OLS fit and Pearson r with Fisher z 95% interval and residual SD.

    NaN-safe: degenerate inputs return NaNs rather than raising, so empty cells
    propagate as dashes downstream.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = int(x.size)
    out = dict(n=n, a=np.nan, b=np.nan, r=np.nan, p=np.nan,
               rsd=np.nan, ci_lo=np.nan, ci_hi=np.nan)
    if n < MIN_N or np.ptp(x) == 0 or np.ptp(y) == 0:
        return out
    fit = linregress(x, y)
    resid = y - (fit.intercept + fit.slope * x)
    out.update(a=float(fit.intercept), b=float(fit.slope), r=float(fit.rvalue),
               p=float(fit.pvalue),
               rsd=float(np.std(resid, ddof=2)) if n > 2 else np.nan)
    if n > 3 and abs(fit.rvalue) < 1.0:
        z, se = np.arctanh(fit.rvalue), 1.0 / np.sqrt(n - 3.0)
        out["ci_lo"] = float(np.tanh(z - 1.96 * se))
        out["ci_hi"] = float(np.tanh(z + 1.96 * se))
    return out


def fisher_mean_r(rs, ns) -> dict:
    """
    Weighted Fisher z average of the region-level coefficients.

    An arithmetic mean of r is biased because r is not additive; averaging on
    the z scale and back-transforming is the standard remedy.
    """
    rs = np.asarray(rs, dtype=float)
    ns = np.asarray(ns, dtype=float)
    ok = np.isfinite(rs) & np.isfinite(ns) & (np.abs(rs) < 1.0) & (ns > 3)
    out = dict(r=np.nan, ci_lo=np.nan, ci_hi=np.nan, k=int(ok.sum()))
    if not ok.any():
        return out
    w = ns[ok] - 3.0
    z_bar = float(np.sum(w * np.arctanh(rs[ok])) / np.sum(w))
    se = 1.0 / np.sqrt(np.sum(w))
    out.update(r=float(np.tanh(z_bar)), ci_lo=float(np.tanh(z_bar - 1.96 * se)),
               ci_hi=float(np.tanh(z_bar + 1.96 * se)))
    return out


# =============================================================================
# COMPUTATION (single pass, shared by all three tables)
# =============================================================================
def compute_volume_metrics(m_all: pd.DataFrame, labels) -> pd.DataFrame:
    """Long-format per (section, region, method) error and correlation."""
    present = [s for s in SECTION_ORDER_TABLE if s in set(m_all["Distance"])]
    rows = []

    for sec in present:
        d_sec = m_all[m_all["Distance"] == sec]
        for lab in list(labels):
            d_lab = d_sec[d_sec["Label"] == lab]
            for method in METHODS:
                d = d_lab[d_lab["Method"] == method]
                st = fit_stats(_tx(d["Ref_mm3"]), _tx(d["Volume_mm3"]))
                es = error_stats(d["Volume_mm3"], d["Ref_mm3"])
                rows.append(dict(
                    Section=sec, SectionLabel=SECTION_HEADER.get(sec, sec),
                    Region=lab, Method=method, RowType="region",
                    n_obs=st["n"],
                    n_subjects=int(d["Subject"].nunique()) if len(d) else 0,
                    **es,
                    **{k: st[k] for k in ("r", "ci_lo", "ci_hi", "p", "b", "a",
                                          "rsd")}))
    df = pd.DataFrame(rows)

    if SUMMARY_ROW and len(df):
        summary = []
        for sec in present:
            for method in METHODS:
                sel = df[(df["Section"] == sec) & (df["Method"] == method)
                         & (df["RowType"] == "region")]
                fm = fisher_mean_r(sel["r"], sel["n_obs"])
                e = sel["err"].to_numpy(dtype=float)
                e = e[np.isfinite(e)]
                # The error summary is an average of the region-level values,
                # so every region carries equal weight regardless of its size.
                # An observation-weighted average would be dominated by the
                # cortex and white matter.
                point = (float(np.mean(e)) if (e.size and
                                               ERROR_IS_MEAN[ERROR_METRIC])
                         else float(np.median(e)) if e.size else np.nan)
                summary.append(dict(
                    Section=sec, SectionLabel=SECTION_HEADER.get(sec, sec),
                    Region="Summary", Method=method, RowType="summary",
                    n_obs=int(sel["n_obs"].sum()),
                    n_subjects=int(sel["n_subjects"].max()) if len(sel) else 0,
                    n_err=int(sel["n_err"].sum()), err=point,
                    err_lo=np.nan, err_hi=np.nan,
                    r=fm["r"], ci_lo=fm["ci_lo"], ci_hi=fm["ci_hi"],
                    p=np.nan, b=np.nan, a=np.nan, rsd=np.nan))
        df = pd.concat([df, pd.DataFrame(summary)], ignore_index=True)

    return df


# =============================================================================
# CELL FORMATTING
# =============================================================================
def _dispersion_mode() -> str:
    if ERROR_DISPERSION != "auto":
        return ERROR_DISPERSION
    return "sd" if ERROR_IS_MEAN[ERROR_METRIC] else "iqr"


def _fmt_err(row, with_dispersion: bool = True) -> str:
    v = row["err"]
    if not np.isfinite(v):
        return "--"
    dec = ERROR_DECIMALS[ERROR_METRIC]
    cell = f"{v:.{dec}f}"
    mode = _dispersion_mode()
    if with_dispersion and mode != "none" and row["RowType"] == "region" \
            and np.isfinite(row["err_lo"]) and np.isfinite(row["err_hi"]):
        if mode == "sd":
            half = (row["err_hi"] - row["err_lo"]) / 2.0
            cell += r"\,$\pm$\,%.*f" % (dec, half)
        else:
            cell += r"\,[%.*f, %.*f]" % (dec, row["err_lo"], dec, row["err_hi"])
    return cell


def _fmt_r(row, with_ci: bool = True) -> str:
    v = row["r"]
    if not np.isfinite(v):
        return "--"
    cell = f"{v:.2f}"
    if with_ci and PEARSON_CI and np.isfinite(row["ci_lo"]) \
            and np.isfinite(row["ci_hi"]):
        cell += r"\,[%.2f, %.2f]" % (row["ci_lo"], row["ci_hi"])
    return cell


def _best_method(d_lab: pd.DataFrame, column: str) -> str | None:
    """Method with the lowest error or the highest r, or None if ambiguous."""
    d = d_lab[np.isfinite(d_lab[column])]
    if len(d) < 2:
        return None
    idx = d[column].idxmin() if column == "err" else d[column].idxmax()
    return d.loc[idx, "Method"]


def _maybe_bold(cell: str, is_best: bool) -> str:
    return r"\textbf{%s}" % cell if (BOLD_BEST and is_best and cell != "--") \
        else cell


# =============================================================================
# SHARED TABLE SKELETON
# =============================================================================
def _sections(df: pd.DataFrame) -> list:
    return [s for s in SECTION_ORDER_TABLE if s in set(df["Section"])]


def _row_order(labels) -> list:
    return list(labels) + (["Summary"] if SUMMARY_ROW else [])


def _row_name(lab: str) -> str:
    if lab != "Summary":
        return lab
    return (r"\textit{Mean over regions}" if ERROR_IS_MEAN[ERROR_METRIC]
            else r"\textit{Median over regions}")


def _emit(df, labels, ncol, colspec, header_lines, row_fn, caption, label):
    L = [r"\begin{table*}[h!]", r"\centering",
         r"\caption{%s}" % caption, r"\label{%s}" % label,
         r"\begin{tabular}{%s}" % colspec, r"\toprule"]
    present = _sections(df)
    for si, sec in enumerate(present):
        L.append(r"\multicolumn{%d}{c}{\textbf{%s}} \\"
                 % (ncol, SECTION_HEADER.get(sec, sec)))
        L.append(r"\midrule")
        if si == 0:
            L.extend(header_lines)
        d_sec = df[df["Section"] == sec]
        for lab in _row_order(labels):
            d_lab = d_sec[d_sec["Region"] == lab]
            if d_lab.empty:
                continue
            if lab == "Summary":
                L.append(r"\cmidrule(lr){1-%d}" % ncol)
            L.append(row_fn(d_lab, lab))
        L.append(r"\bottomrule" if si == len(present) - 1 else r"\midrule")
    L += [r"\end{tabular}", r"\end{table*}"]
    return "\n".join(L)


def _n_prefix(d_lab: pd.DataFrame) -> list:
    return [str(int(d_lab["n_subjects"].max()))] if SHOW_N_COLUMN else []


def _abbr_sentence() -> str:
    return "Abbreviations: " + ", ".join(
        "%s = %s" % (METHOD_ABBR[m_], m_) for m_ in METHODS) + "."


def _space_sentence() -> str:
    return (r"Coefficients are computed on $\log_{10}$-transformed volumes"
            if CORR_SPACE == "log10"
            else r"Coefficients are computed on untransformed volumes")


def _bold_sentence() -> str:
    return (r" The best value per region is set in bold; this is a descriptive "
            r"aid and is not supported by a statistical test."
            if BOLD_BEST else "")


def _summary_sentence(kind: str) -> str:
    if not SUMMARY_ROW:
        return ""
    if kind == "error":
        return (r" The final row of each section averages the region-level "
                r"values with equal weight per region, so it is not dominated "
                r"by the largest structures.")
    return (r" The final row of each section is the Fisher $z$ average of the "
            r"region-level coefficients, weighted by $n-3$ and "
            r"back-transformed; it is not an arithmetic mean of $r$.")


# =============================================================================
# TABLE 1 -- VOLUME ERROR
# =============================================================================
def build_error_latex(df: pd.DataFrame, labels) -> str:
    long_txt, short_txt = ERROR_DISPLAY[ERROR_METRIC]
    mode = _dispersion_mode()
    disp_txt = {"sd": r" Values are given as mean\,$\pm$\,standard deviation "
                      r"across specimens.",
                "iqr": r" Values are given as median\,[interquartile range] "
                       r"across specimens.",
                "none": ""}[mode]

    ncol = 1 + (1 if SHOW_N_COLUMN else 0) + len(METHODS)
    colspec = "l" + ("c" if SHOW_N_COLUMN else "") + "c" * len(METHODS)
    header = r"\textbf{Region}" + (r" & \textbf{n}" if SHOW_N_COLUMN else "") \
        + "".join(r" & \textbf{%s}" % METHOD_ABBR[m_] for m_ in METHODS) + r" \\"

    def row_fn(d_lab, lab):
        best = _best_method(d_lab, "err")
        cells = []
        for m_ in METHODS:
            rr = d_lab[d_lab["Method"] == m_]
            c = _fmt_err(rr.iloc[0]) if len(rr) else "--"
            cells.append(_maybe_bold(c, m_ == best))
        fields = [_row_name(lab)] + _n_prefix(d_lab) + cells
        return "%-26s & %s \\\\" % (fields[0], " & ".join(fields[1:]))

    caption = (r"Region-specific volumetric error of the three reconstruction "
               r"methods against the reference, by cohort and slab distance. "
               r"The reported quantity is the " + long_txt + r"." + disp_txt +
               _bold_sentence() + _summary_sentence("error") +
               r" Dashes indicate cells with fewer than %d paired specimens. "
               % MIN_N + _abbr_sentence())

    return _emit(df, labels, ncol, colspec, [header], row_fn, caption,
                 "app:volume_error")


# =============================================================================
# TABLE 2 -- PEARSON CORRELATION
# =============================================================================
def build_pearson_latex(df: pd.DataFrame, labels) -> str:
    ncol = 1 + (1 if SHOW_N_COLUMN else 0) + len(METHODS)
    colspec = "l" + ("c" if SHOW_N_COLUMN else "") + "c" * len(METHODS)
    header = r"\textbf{Region}" + (r" & \textbf{n}" if SHOW_N_COLUMN else "") \
        + "".join(r" & \textbf{%s}" % METHOD_ABBR[m_] for m_ in METHODS) + r" \\"

    def row_fn(d_lab, lab):
        best = _best_method(d_lab, "r")
        cells = []
        for m_ in METHODS:
            rr = d_lab[d_lab["Method"] == m_]
            c = _fmt_r(rr.iloc[0]) if len(rr) else "--"
            cells.append(_maybe_bold(c, m_ == best))
        fields = [_row_name(lab)] + _n_prefix(d_lab) + cells
        return "%-26s & %s \\\\" % (fields[0], " & ".join(fields[1:]))

    ci_txt = (r" Each cell reports $r$ with its Fisher $z$ 95\% confidence "
              r"interval." if PEARSON_CI else "")

    caption = (r"Region-specific Pearson correlation between "
               r"reconstruction-derived and reference volumes, by cohort and "
               r"slab distance. " + _space_sentence() + r" across specimens "
               r"within a single region, so the coefficient is not inflated by "
               r"the between-region range of structure sizes." + ci_txt +
               _bold_sentence() + _summary_sentence("corr") +
               r" Dashes indicate cells with fewer than %d paired specimens or "
               r"zero variance. " % MIN_N + _abbr_sentence())

    return _emit(df, labels, ncol, colspec, [header], row_fn, caption,
                 "app:volume_pearson")


# =============================================================================
# TABLE 3 -- JOINT
# =============================================================================
def build_joint_latex(df: pd.DataFrame, labels) -> str:
    _, short_txt = ERROR_DISPLAY[ERROR_METRIC]
    k = len(METHODS)

    if JOINT_LAYOUT == "grouped":
        ncol = 1 + 2 * k
        colspec = "l" + "c" * k + "c" * k
        head1 = (r"\textbf{Region} & \multicolumn{%d}{c}{\textbf{%s}} "
                 r"& \multicolumn{%d}{c}{\textbf{Pearson $r$}} \\"
                 % (k, short_txt, k))
        rule = r"\cmidrule(lr){2-%d}\cmidrule(lr){%d-%d}" % (1 + k, 2 + k, 1 + 2 * k)
        head2 = ("" + "".join(r" & \textbf{%s}" % METHOD_ABBR[m_] for m_ in METHODS) * 1
                 + "".join(r" & \textbf{%s}" % METHOD_ABBR[m_] for m_ in METHODS)
                 + r" \\")
        header_lines = [head1, rule, head2]

        def row_fn(d_lab, lab):
            best_e = _best_method(d_lab, "err")
            best_r = _best_method(d_lab, "r")
            cells = []
            for m_ in METHODS:
                rr = d_lab[d_lab["Method"] == m_]
                c = _fmt_err(rr.iloc[0], with_dispersion=False) if len(rr) else "--"
                cells.append(_maybe_bold(c, m_ == best_e))
            for m_ in METHODS:
                rr = d_lab[d_lab["Method"] == m_]
                c = _fmt_r(rr.iloc[0], with_ci=False) if len(rr) else "--"
                cells.append(_maybe_bold(c, m_ == best_r))
            return "%-26s & %s \\\\" % (_row_name(lab), " & ".join(cells))

        layout_txt = (r" Point estimates only; dispersion of the error and "
                      r"confidence intervals for $r$ are reported in "
                      r"Tables~\ref{app:volume_error} "
                      r"and~\ref{app:volume_pearson}.")
    else:
        ncol = 1 + k
        colspec = "l" + "c" * k
        header_lines = [r"\textbf{Region}" +
                        "".join(r" & \textbf{%s}" % METHOD_ABBR[m_]
                                for m_ in METHODS) + r" \\"]

        def row_fn(d_lab, lab):
            cells = []
            for m_ in METHODS:
                rr = d_lab[d_lab["Method"] == m_]
                if not len(rr):
                    cells.append("--")
                    continue
                e = _fmt_err(rr.iloc[0], with_dispersion=False)
                r_ = _fmt_r(rr.iloc[0], with_ci=False)
                cells.append("--" if (e == "--" and r_ == "--")
                             else f"{e} ({r_})")
            return "%-26s & %s \\\\" % (_row_name(lab), " & ".join(cells))

        layout_txt = (r" Each cell reports the error followed by the Pearson "
                      r"coefficient in parentheses. Dispersion and confidence "
                      r"intervals are reported in "
                      r"Tables~\ref{app:volume_error} "
                      r"and~\ref{app:volume_pearson}.")

    long_txt, _ = ERROR_DISPLAY[ERROR_METRIC]
    caption = (r"Region-specific volumetric error and correlation for the three "
               r"reconstruction methods, by cohort and slab distance. The error "
               r"is the " + long_txt + r"; the correlation is computed across "
               r"specimens within each region. " + _space_sentence() + r"." +
               layout_txt + _bold_sentence() +
               r" The two quantities are complementary: $r$ is invariant to an "
               r"affine rescaling of the reconstructed volumes and therefore "
               r"cannot detect a systematic over- or under-estimation, which is "
               r"what the error column measures. Dashes indicate cells with "
               r"fewer than %d paired specimens. " % MIN_N + _abbr_sentence())

    return _emit(df, labels, ncol, colspec, header_lines, row_fn, caption,
                 "app:volume_joint")


# =============================================================================
# WRITING
# =============================================================================
def write_volume_tables(m_all: pd.DataFrame, labels, out_dir: str,
                        df: pd.DataFrame = None) -> list:
    """Compute once, write three LaTeX tables and one long-format CSV."""
    if df is None:
        df = compute_volume_metrics(m_all, labels)
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, "volume_metrics_by_region.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    paths = [csv_path]

    for name, builder in (("volume_error_table", build_error_latex),
                          ("volume_pearson_table", build_pearson_latex),
                          ("volume_joint_table", build_joint_latex)):
        p = os.path.join(out_dir, f"{name}.tex")
        with open(p, "w", encoding="utf-8") as f:
            f.write(builder(df, labels) + "\n")
        paths.append(p)

    if len(df):
        print(f"[tables] error metric = {ERROR_METRIC} "
              f"({_dispersion_mode()} dispersion); correlation space = "
              f"{CORR_SPACE}; joint layout = {JOINT_LAYOUT}")
        thin = df[(df["RowType"] == "region") & df["n_obs"].between(1, 5)]
        if len(thin):
            print(f"[tables] warning: {len(thin)} cell(s) with n < 6; the "
                  f"Fisher interval is wide and r is unstable there.")
        empty = df[(df["RowType"] == "region") & (df["n_obs"] == 0)]
        if len(empty):
            combos = sorted(set(zip(empty["Section"], empty["Method"])))
            print(f"[tables] warning: {len(empty)} empty cell(s); check the "
                  f"case lists for {combos[:5]}"
                  f"{' ...' if len(combos) > 5 else ''}")
    return paths