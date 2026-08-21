"""
Figures built from the analysis frame.

Shared panel machinery lives at the top; each figure supplies only what is
specific to it. The two-by-three grid (rows = cohort, columns = method) is
described once in `cohort_method_grid`.

Figures that live in their own modules, such as the per-label normalised
concordance and the forest plot, import `panel_groups` and `save_figure` from
here so that colour, ordering and output conventions stay identical.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .config import (METHODS, METHOD_DISPLAY, DISTANCES, MADRC_LABEL,
                     PANEL_LETTERS, POINT_ORDER, LINE_ORDER, FIT_COLOR,
                     FIG_FORMATS, FIG_DPI, BASE_FONTSIZE, section_color)
from .statistics import fit_stats


# =============================================================================
# SHARED HELPERS
# =============================================================================
def save_figure(fig, out_dir: str, stem: str, close: bool = True) -> list:
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for ext in FIG_FORMATS:
        p = os.path.join(out_dir, f"{stem}.{ext}")
        fig.savefig(p, bbox_inches="tight", dpi=FIG_DPI)
        paths.append(p)
    if close:
        plt.close(fig)
    return paths


def panel_groups(sub: pd.DataFrame, by_distance: bool) -> list:
    """
    Split one panel's data into coloured subgroups.

    UW panels split by slab distance; MADRC has a single group. Returning
    (name, frame, colour, legend_label) keeps every figure's colour and legend
    behaviour identical.
    """
    if by_distance:
        return [(d, sub[sub["Distance"] == d], section_color(d), d)
                for d in DISTANCES]
    return [(MADRC_LABEL, sub, section_color(MADRC_LABEL), MADRC_LABEL)]


def ordered_legend(axes) -> tuple:
    """Collect handles across every panel, in a fixed display order."""
    seen, order = {}, []
    for ax in np.ravel(axes):
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l and not l.startswith("_") and l not in seen:
                seen[l] = h
                order.append(l)
    known = POINT_ORDER + LINE_ORDER
    labels = [l for l in known if l in seen] + [l for l in order if l not in known]
    return [seen[l] for l in labels], labels


def cohort_method_grid(figsize, sharex=True, sharey=True):
    """The two-by-three grid used by every concordance-style figure."""
    return plt.subplots(2, len(METHODS), figsize=figsize,
                        sharex=sharex, sharey=sharey)


def decorate_panel(ax, row, col, cohort, xlabel, ylabel, square=True):
    ax.grid(True, alpha=0.25, ls=":")
    ax.set_axisbelow(True)
    if square:
        ax.set_aspect("equal", adjustable="box")
    if row == 0:
        ax.set_title(f"{PANEL_LETTERS[col]} {METHOD_DISPLAY[METHODS[col]]}",
                     fontsize=BASE_FONTSIZE, pad=8)
    if col == 0:
        ax.set_ylabel(f"{cohort}\n{ylabel}", fontsize=BASE_FONTSIZE)
    if row == 1:
        ax.set_xlabel(xlabel, fontsize=BASE_FONTSIZE)


# =============================================================================
# LOG-LOG CONCORDANCE
# =============================================================================
def _loglog_limits(frames, pad_frac: float = 0.05) -> tuple:
    vals = []
    for mm in frames:
        if mm is not None and not mm.empty:
            vals.append(np.log10(mm["Ref_mm3"].to_numpy(dtype=float)))
            vals.append(np.log10(mm["Volume_mm3"].to_numpy(dtype=float)))
    if not vals:
        return 2.0, 5.5
    v = np.concatenate(vals)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 2.0, 5.5
    lo, hi = float(v.min()), float(v.max())
    pad = (hi - lo) * pad_frac
    return lo - pad, hi + pad


def _draw_group(ax, x, y, color, lo, hi, point_label=None, line_label=None,
                line_color=None, band=True) -> dict:
    """Scatter, least-squares fit and 1.96 residual-SD band for one subgroup."""
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


def make_concordance_figure(m_uw: pd.DataFrame, m_mad: pd.DataFrame):
    """
    Measured against reference volume on log-log axes, with the identity line,
    a least-squares fit and a 1.96 residual-SD band.

    UW statistics are computed per slab distance, MADRC over the whole cohort.
    Returns (figure, stats_frame).
    """
    rows = [("UW", m_uw, True), ("MADRC", m_mad, False)]
    lo, hi = _loglog_limits([m_uw, m_mad])
    fig, axes = cohort_method_grid((18, 12))
    records = []

    for r, (cohort, mm, by_dist) in enumerate(rows):
        for c, method in enumerate(METHODS):
            ax = axes[r, c]
            ax.plot([lo, hi], [lo, hi], ls="--", color="black", lw=1.4,
                    alpha=0.6, zorder=1, label="y = x")
            sub = mm[mm["Method"] == method] if not mm.empty else mm

            if sub is not None and not sub.empty:
                for name, dd, color, plabel in panel_groups(sub, by_dist):
                    if dd is None or dd.empty:
                        continue
                    st = _draw_group(
                        ax,
                        np.log10(dd["Ref_mm3"].to_numpy(dtype=float)),
                        np.log10(dd["Volume_mm3"].to_numpy(dtype=float)),
                        color, lo, hi,
                        point_label=plabel if by_dist else None,
                        line_label=None if by_dist else "LS Fit",
                        line_color=None if by_dist else FIT_COLOR)
                    records.append(dict(Cohort=cohort, Section=name,
                                        Method=method, **st))

            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            decorate_panel(ax, r, c, cohort,
                           r"Ref. volume [mm$^3$, log$_{10}$]",
                           r"Pred. volume [mm$^3$, log$_{10}$]")

    handles, labels = ordered_legend(axes)
    if handles:
        axes[0, -1].legend(handles, labels, fontsize=BASE_FONTSIZE,
                           loc="lower right", handlelength=1.2,
                           handletextpad=0.5)
    fig.subplots_adjust(wspace=0.02, hspace=0.08, left=0.11)
    return fig, pd.DataFrame.from_records(records)


# =============================================================================
# RELATIVE ERROR AGAINST REFERENCE VOLUME
# =============================================================================
def make_relative_error_figure(m_uw: pd.DataFrame, m_mad: pd.DataFrame,
                               log_x: bool = True):
    """
    Signed relative volume error against reference volume, with bias and limits
    of agreement per panel. This is a Bland-Altman plot in relative units.

    `log_x` controls both the transform and the axis label, which previously
    disagreed: the label claimed a log scale while linear volumes were plotted.
    """
    rows = [("UW", m_uw, True), ("MADRC", m_mad, False)]

    def xvals(mm):
        v = mm["Ref_mm3"].to_numpy(dtype=float)
        return np.log10(v) if log_x else v

    def relerr(mm):
        return (mm["Diff"] / mm["Ref_mm3"] * 100.0).to_numpy(dtype=float)

    pooled_y, pooled_x = [], []
    for _, mm, _ in rows:
        if mm is not None and not mm.empty:
            pooled_y.extend(relerr(mm).tolist())
            pooled_x.extend(xvals(mm).tolist())

    if pooled_y:
        lo, hi = np.percentile(pooled_y, [2, 98])
        mrg = (hi - lo) * 0.10
        y_lo, y_hi = lo - mrg, hi + mrg
    else:
        y_lo, y_hi = -50.0, 50.0
    if pooled_x:
        x_lo, x_hi = min(pooled_x), max(pooled_x)
        xm = (x_hi - x_lo) * 0.04
        x_lo, x_hi = x_lo - xm, x_hi + xm
    else:
        x_lo, x_hi = 0.0, 1.0

    fig, axes = cohort_method_grid((18, 9.5), sharex=True, sharey=True)
    for r, (cohort, mm, by_dist) in enumerate(rows):
        for c, method in enumerate(METHODS):
            ax = axes[r][c]
            sub = mm[mm["Method"] == method] if not mm.empty else mm
            if sub is not None and not sub.empty:
                for _, dd, color, plabel in panel_groups(sub, by_dist):
                    if dd is None or dd.empty:
                        continue
                    ax.scatter(xvals(dd), relerr(dd), s=36, alpha=0.55,
                               color=color, edgecolors="none",
                               label=plabel if by_dist else None, zorder=2)
                y = relerr(sub)
                bias, sd = float(np.mean(y)), float(np.std(y, ddof=1))
                ax.text(0.03, 0.97,
                        f"Bias = {bias:+.1f} %\n"
                        f"LoA [{bias - 1.96 * sd:+.1f}, {bias + 1.96 * sd:+.1f}] %",
                        transform=ax.transAxes, fontsize=BASE_FONTSIZE - 4,
                        va="top",
                        bbox=dict(boxstyle="round", facecolor="wheat",
                                  alpha=0.55))
                ax.axhline(bias, color=FIT_COLOR, lw=1.2, zorder=1)

            ax.axhline(0, color="black", lw=1.0, ls="--", alpha=0.6, zorder=1)
            ax.set_ylim(y_lo, y_hi)
            ax.set_xlim(x_lo, x_hi)
            xlabel = (r"Ref. volume [mm$^3$, log$_{10}$]" if log_x
                      else r"Ref. volume [mm$^3$]")
            decorate_panel(ax, r, c, cohort, xlabel,
                           "Relative volume error (%)", square=False)

    handles, labels = ordered_legend(axes)
    if handles:
        axes[0][-1].legend(handles, labels, fontsize=BASE_FONTSIZE,
                           loc="lower right", title="Slab distance",
                           title_fontsize=BASE_FONTSIZE - 4)
    fig.tight_layout()
    return fig