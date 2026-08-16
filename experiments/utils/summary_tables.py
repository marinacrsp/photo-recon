"""
Summary (overall) error table for build_volume_correlations_joint.py.

Adds a third deliverable alongside the per-region error table and the per-region
p-value table: a compact table whose rows are conditions (MADRC, UW 4/8/12 mm)
and whose columns are the per-method aggregate error plus the pairwise
comparisons between methods.

Estimator
---------
Two-stage macro-average.

  Stage 1 (within specimen). For specimen s, method m, condition d:

      e_bar(s, m, d) = mean over regions rho of AbsErr_rel(s, m, d, rho)

  an UNWEIGHTED mean over regions, so that white matter and cortex do not
  dominate the summary purely by volume.

  Stage 2 (across specimens). Mean, SD, median and IQR of e_bar over specimens,
  and paired Wilcoxon signed-rank tests on e_bar between methods.

The independent unit is therefore the specimen, not the (specimen, region) pair.
Pooling all region rows and testing on that pool would treat the regions of one
specimen as independent replicates and inflate the effective n by roughly the
number of regions.

Region balancing
----------------
The macro-average is comparable across methods only if the region set entering
it is identical for every method within a given specimen and condition.
balance_regions() enforces this. Without it, a method that fails to recover a
difficult structure would be credited with a lower mean simply because that
structure is missing from its average alone.

Integration
-----------
Requires the frame returned by process() / add_normalized_error(), i.e. columns
Subject, Method, Distance, Label, AbsErr, AbsErr_rel, RelErr. Add to main():

    from summary_table import build_summary_outputs
    outputs += build_summary_outputs(m_all, OUT_DIR)
"""

from __future__ import annotations

import os
import itertools
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, ranksums

# =============================================================================
# CONFIGURATION (mirror the host script)
# =============================================================================
METHODS = ["Photo-recon", "Tricubic", "Imputed"]
METHOD_DISPLAY = {
    "Photo-recon": "Photo-recon",
    "Tricubic": "Cubic",
    "Imputed": "Imputed",
}
METHOD_ABBR = {"Photo-recon": "PR", "Tricubic": "Cub", "Imputed": "UNet"}

MADRC_LABEL = "MADRC"
SECTION_ORDER_TABLE = [MADRC_LABEL, "4mm", "8mm", "12mm"]
SECTION_HEADER = {
    "MADRC": "MADRC", "4mm": "UW -- 4 mm", "8mm": "UW -- 8 mm", "12mm": "UW -- 12 mm",
}

PVALUE_PAIRS = list(itertools.combinations(METHODS, 2))
TEST = "wilcoxon"          # "wilcoxon" (paired) or "ranksum"
Q_LEVEL = 0.05

# Multiplicity procedure:
#   "bh"          Benjamini-Hochberg. Controls the FDR. Adjusted q-values are
#                 reported and compared against Q_LEVEL. Valid under
#                 independence or positive regression dependency.
#   "by"          Benjamini-Yekutieli. Controls the FDR under arbitrary
#                 dependence; multiplies the BH adjustment by the harmonic
#                 number H(m), roughly 3.10 at m = 12.
#   "bonferroni"  Controls the FWER under arbitrary dependence. RAW p-values are
#                 reported and compared against the reduced threshold
#                 Q_LEVEL / m. Decision-equivalent to reporting min(1, m*p)
#                 against Q_LEVEL, but leaves the reported number unadjusted.
#   "none"        Raw p-values, no correction.
MULTIPLICITY = "bonferroni"

# Retained for backwards compatibility; MULTIPLICITY takes precedence.
APPLY_BH = False

# Render the pairwise comparison columns in the LaTeX summary table. When False
# the table reports descriptive statistics only; the tests are still computed
# and written to volume_error_summary.csv, so the inferential result remains
# available and auditable without occupying the manuscript table.
INCLUDE_PVALUES_IN_TABLE = False

# Emit a separate LaTeX table holding the summary-level pairwise comparisons.
EMIT_SUMMARY_PVALUE_TABLE = True

# Show the sample-size column in both summary tables. When False the column is
# removed and the per-condition specimen counts are stated in the caption
# instead, so the information is not lost.
SHOW_N_COLUMN = True

# Show the paired effect size beneath each q-value in that table: the median
# paired difference in the per-specimen error and the matched-pairs
# rank-biserial correlation.
SUMMARY_PVALUE_EFFECT_SIZE = True

# Report the aggregate error as a percentage rather than a fraction.
SUMMARY_AS_PERCENT = True

# Location statistic shown in the method columns:
#   "mean_sd"        -- mean +/- SD  (matches the per-region table)
#   "median_iqr"     -- median [Q1, Q3]  (matches the rank-based test)
#   "both"           -- mean +/- SD on the first line, median [IQR] beneath
SUMMARY_STATISTIC = "both"

# Order in which the two averages are taken.
#   "region_first"    Average over specimens first (reproducing the per-region
#                     table), then over regions. Each table cell is then the
#                     column mean of the corresponding block of the per-region
#                     table and can be verified by hand. The dispersion reported
#                     is ACROSS REGIONS, n = number of regions.
#   "specimen_first"  Average over regions within specimen, then over specimens.
#                     The dispersion reported is ACROSS SPECIMENS.
# On a balanced array the two give identical means; only the dispersion and the
# behaviour under missing cells differ.
SUMMARY_AGGREGATION = "region_first"

# Enforce an identical region set across methods within (Subject, Distance).
BALANCE_REGIONS = True

# Externally supplied BH family. Leave None to correct within this table alone;
# pass the raw p-values of the per-region table to correct jointly across both.
EXTERNAL_PVALUE_FAMILY = None


# =============================================================================
# REGION BALANCING
# =============================================================================
def balance_regions(m: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Restrict to regions observed for every method within (Subject, Distance).

    Returns the balanced frame. Rows removed here are still present in the
    per-region table; only the macro-average requires the balance.
    """
    if m.empty:
        return m
    n_meth = m["Method"].nunique()
    cnt = (m.drop_duplicates(["Subject", "Distance", "Label", "Method"])
             .groupby(["Subject", "Distance", "Label"])["Method"].size())
    keep = set(cnt[cnt == n_meth].index)
    idx = pd.MultiIndex.from_frame(m[["Subject", "Distance", "Label"]])
    mask = idx.isin(keep)
    if verbose and (~mask).any():
        lost = m.loc[~mask, ["Subject", "Distance", "Label"]].drop_duplicates()
        print(f"[summary] region balancing removed {len(lost)} "
              f"(Subject, Distance, Region) cell(s) not present for all methods:")
        print(lost.to_string(index=False))
    return m[mask].copy()


# =============================================================================
# STAGE 1 -- SPECIMEN-LEVEL MACRO-AVERAGE
# =============================================================================
def specimen_level(m: pd.DataFrame) -> pd.DataFrame:
    """Collapse regions into one value per (Subject, Method, Distance)."""
    return (m.groupby(["Subject", "Method", "Distance"], as_index=False)
              .agg(AbsErr_rel=("AbsErr_rel", "mean"),
                   AbsErr=("AbsErr", "mean"),
                   RelErr=("RelErr", "mean") if "RelErr" in m.columns
                          else ("AbsErr_rel", "mean"),
                   n_regions=("Label", "nunique")))


# =============================================================================
# STAGE 2 -- ACROSS-SPECIMEN STATISTICS
# =============================================================================
def region_level(m: pd.DataFrame) -> pd.DataFrame:
    """Per-region means over specimens: exactly the cells of the per-region table.

    One value per (Label, Method, Distance), the mean of AbsErr_rel over the
    specimens contributing that region. This is the intermediate the region-first
    aggregation averages over, so each summary cell equals the column mean of the
    corresponding block of the per-region table.
    """
    return (m.groupby(["Label", "Method", "Distance"], as_index=False)
              .agg(AbsErr_rel=("AbsErr_rel", "mean"),
                   n_specimens=("Subject", "nunique")))


def summary_stats(sl: pd.DataFrame, rl: pd.DataFrame | None = None) -> pd.DataFrame:
    """Condition x method statistics, in the configured aggregation order.

    sl is the specimen-level frame (always required: the tests are computed on
    it regardless of the aggregation order, since pairing is by specimen).
    rl is the region-level frame, required when SUMMARY_AGGREGATION is
    'region_first'.

    The dispersion returned refers to different units in the two modes, which is
    recorded in the 'sd_over' column so the caption cannot misdescribe it.
    """
    def q1(x): return np.percentile(x, 25)
    def q3(x): return np.percentile(x, 75)

    n_spec = (sl.groupby(["Distance", "Method"], as_index=False)
                .agg(n=("Subject", "nunique")))

    if SUMMARY_AGGREGATION == "region_first":
        if rl is None:
            raise ValueError("region_first aggregation requires the region-level frame.")
        g = (rl.groupby(["Distance", "Method"], as_index=False)
               .agg(mean=("AbsErr_rel", "mean"),
                    sd=("AbsErr_rel", "std"),
                    median=("AbsErr_rel", "median"),
                    q1=("AbsErr_rel", q1),
                    q3=("AbsErr_rel", q3),
                    n_regions=("Label", "nunique")))
        g["sd_over"] = "regions"
        g["n_sd"] = g["n_regions"]
        # Signed bias is not defined on the region-level frame; take it from sl.
        bias = (sl.groupby(["Distance", "Method"], as_index=False)
                  .agg(bias=("RelErr", "mean")))
        g = g.merge(bias, on=["Distance", "Method"], how="left")
    elif SUMMARY_AGGREGATION == "specimen_first":
        g = (sl.groupby(["Distance", "Method"], as_index=False)
               .agg(mean=("AbsErr_rel", "mean"),
                    sd=("AbsErr_rel", "std"),
                    median=("AbsErr_rel", "median"),
                    q1=("AbsErr_rel", q1),
                    q3=("AbsErr_rel", q3),
                    bias=("RelErr", "mean"),
                    n_regions=("n_regions", "median")))
        g["sd_over"] = "specimens"
        g["n_sd"] = np.nan
    else:
        raise ValueError(f"Unknown SUMMARY_AGGREGATION: {SUMMARY_AGGREGATION!r}")

    g = g.merge(n_spec, on=["Distance", "Method"], how="left")
    if SUMMARY_AGGREGATION == "specimen_first":
        g["n_sd"] = g["n"]
    return g


def summary_pvalue(sl: pd.DataFrame, distance: str,
                   method_a: str, method_b: str) -> float:
    """Paired test on the specimen-level macro-averaged errors. Raw p-value."""
    sel = sl[sl["Distance"] == distance]
    a = sel[sel["Method"] == method_a][["Subject", "AbsErr_rel"]]
    b = sel[sel["Method"] == method_b][["Subject", "AbsErr_rel"]]
    merged = a.merge(b, on="Subject", suffixes=("_a", "_b"))
    if len(merged) < 1:
        return np.nan
    try:
        if TEST == "ranksum":
            _, p = ranksums(merged["AbsErr_rel_a"], merged["AbsErr_rel_b"])
        else:
            _, p = wilcoxon(merged["AbsErr_rel_a"], merged["AbsErr_rel_b"])
        return float(p)
    except ValueError:
        return np.nan


# =============================================================================
# MULTIPLICITY
# =============================================================================
def benjamini_hochberg(pvals, dependence: str = "prds") -> np.ndarray:
    """BH (dependence='prds') or BY (dependence='arbitrary') adjusted q-values.

    Preserves NaNs and input order. BY multiplies the BH adjustment by the
    harmonic number H(m) = sum_{j=1}^{m} 1/j, which guarantees FDR control under
    arbitrary dependence at a substantial cost in power.
    """
    p = np.asarray(pvals, dtype=float)
    q = np.full_like(p, np.nan)
    idx = np.where(~np.isnan(p))[0]
    m = idx.size
    if m == 0:
        return q
    c_m = float(np.sum(1.0 / np.arange(1, m + 1))) if dependence == "arbitrary" else 1.0
    order = idx[np.argsort(p[idx])]
    adj = p[order] * m * c_m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    q[order] = np.clip(adj, 0.0, 1.0)
    return q


def adjust(pvals):
    """Apply the configured multiplicity procedure.

    Returns (values, threshold, reported_kind) where:
        values         the numbers to report, either adjusted q-values or the
                       raw p-values, depending on the procedure;
        threshold      the level against which those values are declared
                       significant (Q_LEVEL, or Q_LEVEL / m for Bonferroni);
        reported_kind  "q" or "p", used to label the table.

    Bonferroni deliberately leaves the reported values raw and lowers the
    threshold instead; this is decision-equivalent to reporting min(1, m*p)
    against Q_LEVEL and keeps the printed number interpretable on its own.
    """
    raw = np.asarray(pvals, dtype=float)
    m = int(np.count_nonzero(~np.isnan(raw)))

    if MULTIPLICITY == "bh":
        return benjamini_hochberg(raw, "prds"), Q_LEVEL, "q"
    if MULTIPLICITY == "by":
        return benjamini_hochberg(raw, "arbitrary"), Q_LEVEL, "q"
    if MULTIPLICITY == "bonferroni":
        return raw, (Q_LEVEL / m if m else Q_LEVEL), "p"
    if MULTIPLICITY == "none":
        return raw, Q_LEVEL, "p"
    raise ValueError(f"Unknown MULTIPLICITY: {MULTIPLICITY!r}")


def summary_pvalue_family(sl: pd.DataFrame):
    """Raw and BH-adjusted p-values over (condition x method pair).

    If EXTERNAL_PVALUE_FAMILY is set, its p-values are appended to the family
    before adjustment so that this table and the per-region table are corrected
    jointly; only the summary entries are returned.
    """
    present = [s for s in SECTION_ORDER_TABLE if s in set(sl["Distance"])]
    keys, raw = [], []
    for sec in present:
        for a, b in PVALUE_PAIRS:
            keys.append((sec, (a, b)))
            raw.append(summary_pvalue(sl, sec, a, b))

    if EXTERNAL_PVALUE_FAMILY:
        combined = list(raw) + list(EXTERNAL_PVALUE_FAMILY)
        vals_all, thresh, kind = adjust(combined)
        vals = np.asarray(vals_all)[:len(raw)]
        m_valid = int(np.count_nonzero(~np.isnan(np.asarray(combined, float))))
    else:
        vals, thresh, kind = adjust(raw)
        m_valid = int(np.count_nonzero(~np.isnan(np.asarray(raw, float))))

    return (present, keys, dict(zip(keys, raw)), dict(zip(keys, vals)),
            m_valid, thresh, kind)


# =============================================================================
# EFFECT SIZE
# =============================================================================
def paired_effect(sl: pd.DataFrame, distance: str,
                  method_a: str, method_b: str) -> dict:
    """Effect size for one summary comparison, on the paired specimen-level data.

    Returns:
        n        number of paired specimens
        delta    median of the paired differences e(a) - e(b); positive means
                 method a has the larger error. Reported as a location shift
                 because the Wilcoxon signed-rank test is a test on the median
                 of those differences, not on the difference of the medians.
        rbc      matched-pairs rank-biserial correlation in [-1, 1]:
                 (W+ - W-) / (W+ + W-), the standardised effect size for the
                 signed-rank test (Kerby, 2014). Zero-differences are excluded,
                 following the Wilcoxon convention used by scipy.
    """
    sel = sl[sl["Distance"] == distance]
    a = sel[sel["Method"] == method_a][["Subject", "AbsErr_rel"]]
    b = sel[sel["Method"] == method_b][["Subject", "AbsErr_rel"]]
    merged = a.merge(b, on="Subject", suffixes=("_a", "_b"))
    if merged.empty:
        return {"n": 0, "delta": np.nan, "rbc": np.nan}

    d = (merged["AbsErr_rel_a"] - merged["AbsErr_rel_b"]).to_numpy(float)
    out = {"n": int(len(d)), "delta": float(np.median(d)), "rbc": np.nan}

    nz = d[d != 0.0]
    if nz.size:
        ranks = pd.Series(np.abs(nz)).rank().to_numpy()
        w_pos = ranks[nz > 0].sum()
        w_neg = ranks[nz < 0].sum()
        total = w_pos + w_neg
        if total > 0:
            out["rbc"] = float((w_pos - w_neg) / total)
    return out


# =============================================================================
# FORMATTING
# =============================================================================
def _scale(x):
    return x * 100.0 if SUMMARY_AS_PERCENT else x


def _fmt_cell(row) -> str:
    if row is None or len(row) == 0:
        return "--"
    r = row.iloc[0]
    mean, sd = _scale(r["mean"]), _scale(r["sd"])
    med, q1, q3 = _scale(r["median"]), _scale(r["q1"]), _scale(r["q3"])
    ms = f"{mean:.1f} $\\pm$ {sd:.1f}" if np.isfinite(sd) else f"{mean:.1f}"
    mi = f"[{med:.1f}; {q1:.1f}--{q3:.1f}]"
    if SUMMARY_STATISTIC == "mean_sd":
        return ms
    if SUMMARY_STATISTIC == "median_iqr":
        return mi
    return r"\makecell{%s \\ \scriptsize %s}" % (ms, mi)


def _fmt_q(v: float, thresh: float = Q_LEVEL) -> str:
    """Format one reported value, bold when it clears the decision threshold.

    The threshold is passed in rather than read from Q_LEVEL, because under
    Bonferroni the reported number is a raw p-value and the threshold is
    Q_LEVEL / m.
    """
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "--"
    if v < 1e-4:
        return r"$\mathbf{<10^{-4}}$" if v < thresh else r"$<10^{-4}$"
    return r"$\mathbf{%.4f}$" % v if v < thresh else f"{v:.4f}"


def _multiplicity_clause(m_valid: int, thresh: float) -> str:
    """Caption text describing the correction actually applied.

    The procedure and the error rate it controls are stated explicitly, because
    'corrected for multiple comparisons' does not distinguish FDR from FWER, and
    the two support different claims.
    """
    test_name = (r"Wilcoxon signed-rank (paired) tests" if TEST == "wilcoxon"
                 else r"Wilcoxon rank-sum (Mann--Whitney) tests")
    if MULTIPLICITY == "bonferroni":
        return (r"Reported values are raw $p$-values from %s. To control the "
                r"family-wise error rate at $\alpha=%.2f$ across the $m=%d$ "
                r"comparisons in this table, significance is declared at the "
                r"Bonferroni-reduced threshold $\alpha/m=%.5f$; entries below "
                r"that threshold are shown in bold. This is decision-equivalent "
                r"to reporting $\min(1, m p)$ against $\alpha$, and holds under "
                r"arbitrary dependence between the tests."
                % (test_name, Q_LEVEL, m_valid, thresh))
    if MULTIPLICITY == "by":
        return (r"Reported values are Benjamini--Yekutieli adjusted $q$-values "
                r"from %s, controlling the false discovery rate across the "
                r"$m=%d$ comparisons in this table under arbitrary dependence; "
                r"entries with $q<%.2f$ are shown in bold."
                % (test_name, m_valid, thresh))
    if MULTIPLICITY == "none":
        return (r"Reported values are raw, uncorrected $p$-values from %s; "
                r"entries with $p<%.2f$ are shown in bold. No adjustment is "
                r"made for the $m=%d$ comparisons in this table."
                % (test_name, thresh, m_valid))
    return (r"Reported values are Benjamini--Hochberg adjusted $q$-values from "
            r"%s, controlling the false discovery rate across the $m=%d$ "
            r"comparisons in this table; entries with $q<%.2f$ are shown in "
            r"bold." % (test_name, m_valid, thresh))


def _n_sentence(counts: dict) -> str:
    """Per-condition specimen counts as a caption clause.

    Used when SHOW_N_COLUMN is False, so that the sample size remains reported
    even though it no longer occupies a column. Conditions sharing a count are
    grouped, which keeps the clause short for the three UW slab distances.
    """
    if not counts:
        return ""
    by_n = {}
    for sec, n in counts.items():
        by_n.setdefault(int(n), []).append(SECTION_HEADER[sec])
    if len(by_n) == 1:
        n = next(iter(by_n))
        return r" All comparisons rest on $n=%d$ specimens." % n
    parts = [r"$n=%d$ for %s" % (n, ", ".join(secs)) for n, secs in
             sorted(by_n.items(), reverse=True)]
    return r" Specimen counts: %s." % "; ".join(parts)


# =============================================================================
# LATEX
# =============================================================================
def build_summary_latex(stats: pd.DataFrame, sl: pd.DataFrame) -> str:
    present, keys, pmap, qmap, m_valid, thresh, kind = summary_pvalue_family(sl)
    unit = r"\%" if SUMMARY_AS_PERCENT else ""

    over = (stats["sd_over"].iloc[0] if len(stats) else "specimens")
    if over == "regions":
        base = (r"For each region the error is first averaged over specimens, "
                r"reproducing the cells of Table~\ref{app:volume_error_joint}; "
                r"each entry below is the mean of those region values over the "
                r"nine regions")
        disp = r", $\pm$ their standard deviation across regions"
    else:
        base = (r"For each specimen the error is first averaged over regions; "
                r"each entry below is the mean of that per-specimen statistic "
                r"over specimens")
        disp = r", $\pm$ its standard deviation across specimens"

    stat_clause = {
        "mean_sd": base + disp,
        "median_iqr": base + r", with the interquartile range",
        "both": base + disp + r", and beneath it the median and interquartile "
                r"range",
    }[SUMMARY_STATISTIC]

    caption_common = (
        r"Aggregate volume error per reconstruction method and condition. For "
        r"each specimen the absolute relative volume error "
        r"$|V_{\mathrm{method}} - V_{\mathrm{ref}}| / V_{\mathrm{ref}}$ is "
        r"computed per region. " + stat_clause +
        r". All averages are unweighted. Hemispheres are averaged "
        r"within specimen, and the region set is balanced across methods. "
        r"Abbreviations: PR = Photo-recon, "
        r"Cub = cubic interpolation, UNet = U-Net imputation. Gold-standard "
        r"volumes are obtained from MRI scans."
    )
    caption_stats = " " + _multiplicity_clause(m_valid, thresh)
    caption_xref = (
        r" Pairwise statistical comparisons between methods are reported in "
        r"Table~\ref{app:volume_pvalues_summary}."
    )
    counts = {sec: (stats[stats["Distance"] == sec]["n"].max()
                    if len(stats[stats["Distance"] == sec]) else 0)
              for sec in present}
    caption = caption_common + (caption_stats if INCLUDE_PVALUES_IN_TABLE
                                else caption_xref)
    caption += (r" $n$ is the number of specimens." if SHOW_N_COLUMN
                else _n_sentence(counts))

    n_lead = 2 if SHOW_N_COLUMN else 1          # leading non-method columns
    n_pcols = len(PVALUE_PAIRS) if INCLUDE_PVALUES_IN_TABLE else 0
    ncols = n_lead + len(METHODS) + n_pcols
    colspec = "l" + "c" * (ncols - 1)
    first_method_col = n_lead + 1
    lead_head = (r"\multirow{2}{*}{\textbf{Condition}}"
                 + (r" & \multirow{2}{*}{\textbf{n}}" if SHOW_N_COLUMN else ""))
    lead_blank = "& " * n_lead

    L = [r"\begin{table*}[h!]", r"\centering",
         r"\caption{%s}" % caption,
         r"\label{app:volume_error_summary}",
         r"\begin{tabular}{%s}" % colspec, r"\toprule"]

    if INCLUDE_PVALUES_IN_TABLE:
        head1 = (lead_head
                 + r" & \multicolumn{%d}{c}{\textbf{Volume error [%s]}}"
                   r" & \multicolumn{%d}{c}{\textbf{Pairwise $%s$}} \\"
                 % (len(METHODS), unit, len(PVALUE_PAIRS), kind))
        head2 = (r"\cmidrule(lr){%d-%d} \cmidrule(lr){%d-%d}"
                 % (first_method_col, n_lead + len(METHODS),
                    n_lead + len(METHODS) + 1, ncols))
        head3 = (lead_blank
                 + " & ".join(r"\textbf{%s}" % METHOD_DISPLAY[m_] for m_ in METHODS)
                 + " & "
                 + " & ".join(r"\textbf{%s vs %s}" % (METHOD_ABBR[a], METHOD_ABBR[b])
                              for a, b in PVALUE_PAIRS)
                 + r" \\")
    else:
        head1 = (lead_head
                 + r" & \multicolumn{%d}{c}{\textbf{Volume error [%s]}} \\"
                 % (len(METHODS), unit))
        head2 = r"\cmidrule(lr){%d-%d}" % (first_method_col, ncols)
        head3 = (lead_blank
                 + " & ".join(r"\textbf{%s}" % METHOD_DISPLAY[m_] for m_ in METHODS)
                 + r" \\")
    L += [head1, head2, head3, r"\midrule"]

    for sec in present:
        d = stats[stats["Distance"] == sec]
        n_spec = int(d["n"].max()) if len(d) else 0
        cells = [_fmt_cell(d[d["Method"] == m_]) for m_ in METHODS]
        if INCLUDE_PVALUES_IN_TABLE:
            cells += [_fmt_q(qmap.get((sec, (a, b)), np.nan), thresh)
                      for a, b in PVALUE_PAIRS]
        lead = f"{SECTION_HEADER[sec]:<14}" + (f" & {n_spec}" if SHOW_N_COLUMN else "")
        L.append("%s & %s \\\\" % (lead, " & ".join(cells)))

    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(L)


def build_summary_pvalue_latex(sl: pd.DataFrame) -> str:
    """Standalone table: pairwise comparisons of the per-specimen aggregate error.

    Rows are conditions, columns are method pairs. The unit of analysis is the
    specimen, so each cell rests on n paired observations rather than on
    n x (number of regions) rows.
    """
    present, keys, pmap, qmap, m_valid, thresh, kind = summary_pvalue_family(sl)
    unit = r", in percentage points," if SUMMARY_AS_PERCENT else ""
    scale = 100.0 if SUMMARY_AS_PERCENT else 1.0

    caption = (
        r"Pairwise statistical comparison of the aggregate volume error between "
        r"reconstruction methods, by condition. For each specimen the absolute "
        r"relative volume error is averaged, with equal weight, over regions "
        r"(Table~\ref{app:volume_error_summary}); the tests are performed on that "
        r"per-specimen statistic, so the unit of analysis is the specimen. "
        + _multiplicity_clause(m_valid, thresh)
    )
    if SUMMARY_PVALUE_EFFECT_SIZE:
        caption += (
            r" Beneath each value, $\Delta$ is the median paired difference "
            r"in the per-specimen error%s positive when the first method of the "
            r"pair has the larger error, and $r_{\mathrm{rb}}$ is the "
            r"matched-pairs rank-biserial correlation." % unit
        )
    caption += (
        r" Dashes indicate comparisons without paired samples. Abbreviations: "
        r"PR = Photo-recon, Cub = cubic interpolation, UNet = U-Net imputation."
    )

    # Paired-specimen counts, evaluated once so they can go in the caption.
    counts = {sec: max([paired_effect(sl, sec, a, b)["n"] for a, b in PVALUE_PAIRS]
                       or [0]) for sec in present}
    caption += (r" $n$ is the number of paired specimens." if SHOW_N_COLUMN
                else _n_sentence(counts))

    n_lead = 2 if SHOW_N_COLUMN else 1
    ncols = n_lead + len(PVALUE_PAIRS)
    header = (r"\textbf{Condition}"
              + (r" & \textbf{n}" if SHOW_N_COLUMN else "")
              + " & "
              + " & ".join(r"\textbf{%s vs %s}" % (METHOD_ABBR[a], METHOD_ABBR[b])
                           for a, b in PVALUE_PAIRS) + r" \\")

    L = [r"\begin{table*}[h!]", r"\centering",
         r"\caption{%s}" % caption,
         r"\label{app:volume_pvalues_summary}",
         r"\begin{tabular}{l%s}" % ("c" * (ncols - 1)), r"\toprule",
         header, r"\midrule"]

    for sec in present:
        cells = []
        for a, b in PVALUE_PAIRS:
            q = qmap.get((sec, (a, b)), np.nan)
            eff = paired_effect(sl, sec, a, b)
            if q is None or (isinstance(q, float) and np.isnan(q)):
                cells.append("--")
                continue
            qtxt = _fmt_q(q, thresh)
            if SUMMARY_PVALUE_EFFECT_SIZE and np.isfinite(eff["delta"]):
                rbc = (f"{eff['rbc']:+.2f}" if np.isfinite(eff["rbc"]) else "--")
                cells.append(
                    r"\makecell{%s \\ \scriptsize $\Delta=%+.1f$, "
                    r"$r_{\mathrm{rb}}=%s$}"
                    % (qtxt, eff["delta"] * scale, rbc))
            else:
                cells.append(qtxt)
        lead = (f"{SECTION_HEADER[sec]:<14}"
                + (f" & {counts.get(sec, 0)}" if SHOW_N_COLUMN else ""))
        L.append("%s & %s \\\\" % (lead, " & ".join(cells)))

    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(L)


# =============================================================================
# CSV
# =============================================================================
def write_summary_csv(stats: pd.DataFrame, sl: pd.DataFrame, out_dir: str) -> str:
    present, keys, pmap, qmap, m_valid, thresh, kind = summary_pvalue_family(sl)
    rows = []
    for sec in present:
        d = stats[stats["Distance"] == sec]
        row = {"Condition": SECTION_HEADER[sec], "Distance": sec,
               "n_specimens": int(d["n"].max()) if len(d) else 0,
               "n_regions": float(d["n_regions"].median()) if len(d) else np.nan,
               "aggregation": SUMMARY_AGGREGATION,
               "sd_over": (d["sd_over"].iloc[0] if len(d) else "")}
        for m_ in METHODS:
            r = d[d["Method"] == m_]
            for stat in ("mean", "sd", "median", "q1", "q3", "bias", "n_sd"):
                row[f"{METHOD_ABBR[m_]}_{stat}"] = (float(r[stat].iloc[0])
                                                    if len(r) else np.nan)
        for a, b in PVALUE_PAIRS:
            key = (sec, (a, b))
            tag = f"{METHOD_ABBR[a]}_vs_{METHOD_ABBR[b]}"
            eff = paired_effect(sl, sec, a, b)
            row[f"p_raw_{tag}"] = pmap.get(key, np.nan)
            row[f"q_bh_{tag}"] = qmap.get(key, np.nan)
            row[f"n_pairs_{tag}"] = eff["n"]
            row[f"median_delta_{tag}"] = eff["delta"]
            row[f"rank_biserial_{tag}"] = eff["rbc"]
        rows.append(row)
    path = os.path.join(out_dir, "volume_error_summary.csv")
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    return path


def write_specimen_level_csv(sl: pd.DataFrame, out_dir: str) -> str:
    """Per-specimen macro-averaged errors, the input to every test above."""
    path = os.path.join(out_dir, "volume_error_per_specimen.csv")
    (sl.sort_values(["Distance", "Method", "Subject"])
       .to_csv(path, index=False, encoding="utf-8-sig"))
    return path


# =============================================================================
# ENTRY POINT
# =============================================================================
def build_summary_outputs(m_all: pd.DataFrame, out_dir: str) -> list:
    """Produce the summary LaTeX table, its p-value companion, and two CSVs."""
    m = balance_regions(m_all) if BALANCE_REGIONS else m_all
    sl = specimen_level(m)
    rl = region_level(m)
    stats = summary_stats(sl, rl)

    _, _, _, _, m_valid, thresh, kind = summary_pvalue_family(sl)
    if INCLUDE_PVALUES_IN_TABLE:
        where = "rendered in the descriptive table"
    elif EMIT_SUMMARY_PVALUE_TABLE:
        where = "rendered in volume_pvalue_summary_table.tex"
    else:
        where = "written to volume_error_summary.csv only"
    print(f"[summary] {MULTIPLICITY} over m = {m_valid} valid test(s), "
          f"threshold = {thresh:.5f}, reporting raw {kind}-values; {where}; "
          f"family = "
          f"{'joint with the per-region table' if EXTERNAL_PVALUE_FAMILY else 'this table alone'}")

    outputs = []

    tex = os.path.join(out_dir, "volume_error_summary_table.tex")
    with open(tex, "w", encoding="utf-8") as f:
        f.write(build_summary_latex(stats, sl) + "\n")
    outputs.append(tex)

    if EMIT_SUMMARY_PVALUE_TABLE:
        tex_p = os.path.join(out_dir, "volume_pvalue_summary_table.tex")
        with open(tex_p, "w", encoding="utf-8") as f:
            f.write(build_summary_pvalue_latex(sl) + "\n")
        outputs.append(tex_p)

    outputs.append(write_summary_csv(stats, sl, out_dir))
    outputs.append(write_specimen_level_csv(sl, out_dir))
    return outputs