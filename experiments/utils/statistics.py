"""
Statistics. Every quantity reported anywhere in the package is computed here
and nowhere else, so a figure and a table cannot disagree about the same number.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import linregress, wilcoxon, ranksums

from .config import CORR_SPACE, MIN_N, TEST, CORRECTION


# =============================================================================
# TRANSFORMS
# =============================================================================
def tx(v) -> np.ndarray:
    """Map volumes into the space in which correlations are computed."""
    v = np.asarray(v, dtype=float)
    if CORR_SPACE != "log10":
        return v
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.log10(v)
    return np.where(v > 0, out, np.nan)


# =============================================================================
# CORRELATION
# =============================================================================
def fit_stats(x, y) -> dict:
    """
    Least-squares fit and Pearson r with a Fisher z interval and residual SD.

    NaN-safe: degenerate input returns NaNs rather than raising, so an empty
    cell propagates to a dash instead of aborting the run.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = int(x.size)
    out = dict(n=n, a=np.nan, b=np.nan, r=np.nan, p=np.nan, rsd=np.nan,
               ci_lo=np.nan, ci_hi=np.nan)
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
    Inverse-variance average of correlations on the Fisher z scale.

    The sampling variance of r is approximately (1 - rho^2)^2 / n and therefore
    depends on the value being estimated, so coefficients of different magnitude
    are not directly averageable. On the z scale the variance is approximately
    1 / (n - 3), independent of rho, which is why the weight is n - 3.
    """
    rs, ns = np.asarray(rs, dtype=float), np.asarray(ns, dtype=float)
    ok = np.isfinite(rs) & np.isfinite(ns) & (np.abs(rs) < 1.0) & (ns > 3)
    out = dict(r=np.nan, ci_lo=np.nan, ci_hi=np.nan, k=int(ok.sum()),
               n=int(np.nansum(ns[ok])) if ok.any() else 0)
    if not ok.any():
        return out
    w = ns[ok] - 3.0
    z_bar = float(np.sum(w * np.arctanh(rs[ok])) / np.sum(w))
    se = 1.0 / np.sqrt(np.sum(w))
    out.update(r=float(np.tanh(z_bar)),
               ci_lo=float(np.tanh(z_bar - 1.96 * se)),
               ci_hi=float(np.tanh(z_bar + 1.96 * se)))
    return out


# =============================================================================
# MULTIPLICITY
# =============================================================================
def bonferroni(pvals) -> np.ndarray:
    """Bonferroni-adjusted p-values; preserves NaNs and input order."""
    p = np.asarray(pvals, dtype=float)
    q = np.full_like(p, np.nan)
    idx = np.where(~np.isnan(p))[0]
    if idx.size:
        q[idx] = np.clip(p[idx] * idx.size, 0.0, 1.0)
    return q


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
    adj = np.minimum.accumulate(adj[::-1])[::-1]   # monotone from the top
    q[order] = np.clip(adj, 0.0, 1.0)
    return q


def adjust(pvals, method: str = None) -> np.ndarray:
    """Apply the configured correction. 'none' returns the input unchanged."""
    method = CORRECTION if method is None else method
    if method == "bonferroni":
        return bonferroni(pvals)
    if method == "bh":
        return benjamini_hochberg(pvals)
    return np.asarray(pvals, dtype=float)


def n_valid(pvals) -> int:
    return int(np.count_nonzero(~np.isnan(np.asarray(pvals, dtype=float))))


# =============================================================================
# PAIRED TESTS
# =============================================================================
def paired_pvalue(m: pd.DataFrame, section: str, label: str,
                  method_a: str, method_b: str,
                  column: str = "AbsErr") -> float:
    """
    Raw p for one (section, region, pair), paired on specimen.

    The inner merge is what makes the test paired: a specimen present for one
    method only is dropped. Correction is applied once over the whole family by
    the caller, never here.
    """
    sel = (m["Distance"] == section) & (m["Label"] == label)
    a = m[sel & (m["Method"] == method_a)][["Subject", column]]
    b = m[sel & (m["Method"] == method_b)][["Subject", column]]
    if a.empty or b.empty:
        return np.nan
    merged = a.merge(b, on="Subject", suffixes=("_a", "_b"))
    if len(merged) < 1:
        return np.nan
    x = merged[f"{column}_a"].to_numpy(dtype=float)
    y = merged[f"{column}_b"].to_numpy(dtype=float)
    if np.allclose(x - y, 0.0):
        return np.nan
    try:
        if TEST == "ranksum":
            return float(ranksums(x, y).pvalue)
        return float(wilcoxon(x, y).pvalue)
    except (ValueError, ZeroDivisionError):
        return np.nan


def pvalue_family(m: pd.DataFrame, labels, sections, pairs,
                  column: str = "AbsErr") -> pd.DataFrame:
    """
    Every pairwise test of the table, in one frame, corrected once.

    Returning a frame rather than two dicts is what stops the table and the CSV
    from recomputing the tests independently and disagreeing about them.
    Columns: Section, Region, Method_1, Method_2, p, p_adj, n_pairs, m_tests.
    """
    rows = []
    for sec in sections:
        for lab in labels:
            for a, b in pairs:
                sel = (m["Distance"] == sec) & (m["Label"] == lab)
                n_pairs = len(
                    m[sel & (m["Method"] == a)][["Subject"]]
                    .merge(m[sel & (m["Method"] == b)][["Subject"]],
                           on="Subject"))
                rows.append(dict(Section=sec, Region=lab, Method_1=a,
                                 Method_2=b, n_pairs=n_pairs,
                                 p=paired_pvalue(m, sec, lab, a, b, column)))
    df = pd.DataFrame(rows)
    if not len(df):
        return df
    df["p_adj"] = adjust(df["p"].to_numpy())
    df["m_tests"] = n_valid(df["p"])
    return df