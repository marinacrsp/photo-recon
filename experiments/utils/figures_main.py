"""
Region-wise volume correspondence, three regions per figure, stacked vertically.

Layout of one figure
--------------------
Six rows by three columns. Rows are grouped in pairs, one pair per region:

                    PR      Cubic    UNet
    Cortex   UW      .        .        .
             MADRC   .        .        .
    WM       UW      .        .        .
             MADRC   .        .        .
    Ventricle UW     .        .        .
             MADRC   .        .        .

Each panel is a scatter of reconstruction-derived against reference volume for
one region, cohort and method, in mL on linear axes. Axis limits are shared
within a region block, so the three methods are directly comparable, and set
independently between blocks, since structure volumes differ by orders of
magnitude.

Volumes are not normalised here: a panel holding a single region has no
between-region size range, so physical units are both available and clearer.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .config import (METHODS, METHOD_ABBR, METHOD_DISPLAY, DISTANCES,
                     MADRC_LABEL, PANEL_LETTERS, BASE_FONTSIZE, section_color)
from .figures import panel_groups, ordered_legend, save_figure
from .statistics import fit_stats


# =============================================================================
# CONFIGURATION
# =============================================================================
# Three regions per figure, nine in total. Grouped by structure size so that a
# figure's three blocks have comparable axis ranges and the reader is not asked
# to switch between millilitres and fractions of one within a single figure.
REGION_GROUPS = [
    ["Cortex", "WM", "Ventricle"],
    ["Thalamus", "Putamen", "Hippocampus"],
    ["Caudate", "Pallidum", "Amygdala"],
]
GROUP_NAMES = ["large", "mid", "small"]

REGION_TITLES = {"WM": "White matter"}      # fallback is the label itself

COHORT_ROWS = ["UW", "MADRC"]

# Figure geometry. Width is the controlled quantity: panel size is derived from
# it, so changing the target width rescales the figure without touching
# anything else. The height is then solved for, so that every panel slot is
# exactly square and the equal-aspect constraint introduces no extra gap.
FIG_WIDTH_IN = 6.5                   # 7.1 fills a two-column journal page
PANEL_ASPECT_EQUAL = True            # False frees the height at the cost of
                                     # making departures from y = x harder to judge

# Spacing, in units of one panel height or width.
#   ROW_HSPACE   between the two cohort rows of the same region: tight, since
#                they share an axis range and belong together
#   BLOCK_HSPACE between region blocks, as a fraction of a whole block: wide,
#                since a new region means a new axis range
ROW_HSPACE = 0.08
BLOCK_HSPACE = 0.18
COL_WSPACE = 0.10

# Figure margins, in figure coordinates. The left margin holds three text
# lanes: the rotated region name, the shared y-axis label and the cohort label.
LEFT_MARGIN = 0.24
RIGHT_MARGIN = 0.99
TOP_MARGIN = 0.965
BOTTOM_MARGIN = 0.1

REGION_LABEL_X = 0.015               # outermost lane
SUPYLABEL_X = 0.1                  # middle lane

# Repeat the axis names once per region block instead of once per figure. With
# three blocks stacked vertically, a single centred label sits far from the top
# and bottom blocks, so the reader has to travel to find the units.
YLABEL_PER_BLOCK = True
XLABEL_PER_BLOCK = False             # each block already carries its own ticks

# Type sizes, absolute rather than offsets from BASE_FONTSIZE, which is set for
# the much larger pooled figures and is far too big for a 1.8 in panel.
TICK_FONTSIZE = 12
TITLE_FONTSIZE = 12
AXIS_LABEL_FONTSIZE = 12
COHORT_FONTSIZE = 12
REGION_FONTSIZE = 12
LEGEND_FONTSIZE = 12

UNITS = "mL"                         # "mL" | "mm3"
LIMITS_SCOPE = "region"              # "region" | "figure"
PAD_FRAC = 0.06

SHOW_FIT = False
SHOW_BAND = False

ANNOT = "none"                       # "r" | "none"
ANNOT_CI = False
ANNOT_FONTSIZE = 8
ANNOT_DY = 0.085
ANNOT_LOC = "upper left"             # "upper left" | "lower right"

POINT_SIZE = 12
POINT_ALPHA = 0.60

def _to_units(v):
    v = np.asarray(v, dtype=float)
    return v / 1000.0 if UNITS == "mL" else v
 
 
def _unit_label() -> str:
    return "mL" if UNITS == "mL" else r"mm$^3$"
 
 
def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(s)).strip("_").lower()
 
 
# =============================================================================
# LIMITS
# =============================================================================
def _square_limits(frames) -> tuple:
    vals = []
    for d in frames:
        if d is not None and not d.empty:
            vals.append(_to_units(d["Ref_mm3"].to_numpy()))
            vals.append(_to_units(d["Volume_mm3"].to_numpy()))
    if not vals:
        return 0.0, 1.0
    v = np.concatenate(vals)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0, 1.0
    lo, hi = float(v.min()), float(v.max())
    if hi <= lo:
        hi = lo + 1.0
    pad = (hi - lo) * PAD_FRAC
    return max(0.0, lo - pad), hi + pad
 
 
def _limits_map(frames, regions) -> dict:
    present = [f for f in frames if f is not None and not f.empty]
    if LIMITS_SCOPE == "figure":
        lim = _square_limits([f[f["Label"].isin(regions)] for f in present])
        return {r: lim for r in regions}
    return {r: _square_limits([f[f["Label"] == r] for f in present])
            for r in regions}
 
 
# =============================================================================
# ANNOTATION
# =============================================================================
def _region_r(pearson_df, section: str, region: str, method: str):
    nan3 = (np.nan, np.nan, np.nan)
    if pearson_df is None or not len(pearson_df):
        return nan3
    d = pearson_df[(pearson_df["Section"] == section)
                   & (pearson_df["Region"] == region)
                   & (pearson_df["Method"] == method)]
    if "RowType" in d.columns:
        d = d[d["RowType"] == "region"]
    if not len(d):
        return nan3
    row = d.iloc[0]
    return (float(row.get("r", np.nan)), float(row.get("ci_lo", np.nan)),
            float(row.get("ci_hi", np.nan)))
 
 
def _annot_lines(pearson_df, region: str, method: str, by_dist: bool) -> list:
    out = []
    # DISTANCES is ordered largest-first for draw order; annotate small-first.
    sections = list(reversed(DISTANCES)) if by_dist else [MADRC_LABEL]
    for sec in sections:
        r, lo, hi = _region_r(pearson_df, sec, region, method)
        if not np.isfinite(r):
            continue
        txt = (r"%s: $r$ = %.2f" % (sec, r)) if by_dist else (r"$r$ = %.2f" % r)
        if ANNOT_CI and np.isfinite(lo) and np.isfinite(hi):
            txt += r" [%.2f, %.2f]" % (lo, hi)
        out.append((txt, section_color(sec if by_dist else MADRC_LABEL)))
    return out
 
 
# =============================================================================
# FIGURE
# =============================================================================
def make_region_group_figure(m_uw: pd.DataFrame, m_mad: pd.DataFrame,
                             regions, pearson_df=None):
    """
    One figure for a group of regions. Rows = region x cohort, columns = method.
    """
    regions = list(regions)
    n_rows = len(regions) * len(COHORT_ROWS)
    n_cols = len(METHODS)
    lims = _limits_map([m_uw, m_mad], regions)
    frames = {"UW": (m_uw, True), "MADRC": (m_mad, False)}
 
    if ANNOT == "r" and pearson_df is None:
        print("[regionfig] warning: ANNOT='r' but no pearson_df was passed; "
              "panels will be drawn without annotation")
 
    # Panel width is what the target width leaves after the label lanes and the
    # inter-column gaps. The figure height is then solved so that a panel slot
    # is exactly square: with equal aspect a non-square slot would be shrunk to
    # fit, opening a gap that no spacing parameter controls.
    n_blocks = len(regions)
    panel_w = (FIG_WIDTH_IN * (RIGHT_MARGIN - LEFT_MARGIN)
               / (n_cols + (n_cols - 1) * COL_WSPACE))
    block_units = len(COHORT_ROWS) + (len(COHORT_ROWS) - 1) * ROW_HSPACE
    total_units = block_units * (n_blocks + (n_blocks - 1) * BLOCK_HSPACE)
    fig_h = panel_w * total_units / max(TOP_MARGIN - BOTTOM_MARGIN, 0.1)
 
    fig = plt.figure(figsize=(FIG_WIDTH_IN, fig_h))
    # Nested grids: the outer one spaces the region blocks, the inner one the
    # two cohort rows within a block. A single grid cannot hold two different
    # vertical gaps.
    outer = fig.add_gridspec(n_blocks, 1, hspace=BLOCK_HSPACE,
                             left=LEFT_MARGIN, right=RIGHT_MARGIN,
                             top=TOP_MARGIN, bottom=BOTTOM_MARGIN)
    axes = np.empty((n_rows, n_cols), dtype=object)
    for b in range(n_blocks):
        inner = outer[b].subgridspec(len(COHORT_ROWS), n_cols,
                                     hspace=ROW_HSPACE, wspace=COL_WSPACE)
        for j in range(len(COHORT_ROWS)):
            for c in range(n_cols):
                axes[b * len(COHORT_ROWS) + j][c] = fig.add_subplot(inner[j, c])
 
    for b, region in enumerate(regions):
        lo, hi = lims.get(region, (0.0, 1.0))
        for j, cohort in enumerate(COHORT_ROWS):
            r_i = b * len(COHORT_ROWS) + j
            mm, by_dist = frames[cohort]
            is_block_bottom = (j == len(COHORT_ROWS) - 1)
 
            for c, method in enumerate(METHODS):
                ax = axes[r_i][c]
                ax.plot([lo, hi], [lo, hi], ls="--", color="black", lw=1.2,
                        alpha=0.6, zorder=1, label="y = x")
 
                sub = (mm[(mm["Method"] == method) & (mm["Label"] == region)]
                       if mm is not None and not mm.empty else None)
                if sub is not None and not sub.empty:
                    for _, dd, color, plabel in panel_groups(sub, by_dist):
                        if dd is None or dd.empty:
                            continue
                        x, y = _to_units(dd["Ref_mm3"]), _to_units(dd["Volume_mm3"])
                        ax.scatter(x, y, s=POINT_SIZE, alpha=POINT_ALPHA,
                                   color=color, edgecolors="none",
                                   label=plabel, zorder=3)
                        if SHOW_FIT:
                            st = fit_stats(x, y)
                            if np.isfinite(st["b"]):
                                xg = np.linspace(lo, hi, 100)
                                yg = st["a"] + st["b"] * xg
                                ax.plot(xg, yg, color=color, lw=1.3,
                                        alpha=0.85, zorder=4)
                                if SHOW_BAND and np.isfinite(st["rsd"]):
                                    ax.fill_between(
                                        xg, yg - 1.96 * st["rsd"],
                                        yg + 1.96 * st["rsd"], color=color,
                                        alpha=0.10, zorder=2)
 
                    if ANNOT == "r":
                        lines = _annot_lines(pearson_df, region, method,
                                             by_dist)
                        for i, (txt, color) in enumerate(lines):
                            if ANNOT_LOC == "lower right":
                                x0, y0, ha = 0.97, 0.03 + ANNOT_DY * (
                                    len(lines) - 1 - i), "right"
                                va = "bottom"
                            else:
                                x0, y0, ha = 0.035, 0.965 - ANNOT_DY * i, "left"
                                va = "top"
                            ax.text(x0, y0, txt, transform=ax.transAxes,
                                    fontsize=ANNOT_FONTSIZE, ha=ha, va=va,
                                    color=color)
 
                ax.set_xlim(lo, hi)
                ax.set_ylim(lo, hi)
                if PANEL_ASPECT_EQUAL:
                    ax.set_aspect("equal", adjustable="box")
                ax.grid(True, alpha=0.25, ls=":")
                ax.set_axisbelow(True)
                ax.tick_params(labelsize=TICK_FONTSIZE)
 
                # Method titles only on the very first row of the figure.
                if r_i == 0:
                    ax.set_title(f"{PANEL_LETTERS[c]} {METHOD_DISPLAY[METHODS[c]]}",
                                 fontsize=TITLE_FONTSIZE, pad=5)
                # Tick labels only where they carry information: on the lower
                # row of each block, since the two cohorts share a range, and
                # in the first column. The axis names themselves are shared by
                # the whole figure, so no per-axes label is set.
                if not is_block_bottom:
                    ax.tick_params(labelbottom=False)
                if c == 0:
                    ax.set_ylabel(cohort, fontsize=COHORT_FONTSIZE, labelpad=4)
                else:
                    ax.tick_params(labelleft=False)
 
    handles, names = ordered_legend(axes)
    if handles:
        # The identity line belongs at the end; ordered_legend keeps the
        # subgroup order but places it wherever it was first drawn.
        names = [n for n in names if n != "y = x"] + \
                (["y = x"] if "y = x" in names else [])
        lut = dict(zip(*ordered_legend(axes)[::-1]))
        fig.legend([lut[n] for n in names], names, loc="lower center",
                   ncol=min(len(names), 6), frameon=False,
                   fontsize=LEGEND_FONTSIZE,
                   bbox_to_anchor=(0.5 * (LEFT_MARGIN + RIGHT_MARGIN), 0.001),
                   markerscale=2.0, columnspacing=1.2, handletextpad=0.4)
 
    # Centred on the panel area, not the figure: the left margin holds three
    # text lanes, so the figure midpoint sits well left of the panels.
    panel_mid_x = 0.5 * (LEFT_MARGIN + RIGHT_MARGIN)
    fig.supxlabel(f"Reference volume [{_unit_label()}]",
                  fontsize=AXIS_LABEL_FONTSIZE, x=panel_mid_x,
                  y=BOTTOM_MARGIN * 0.55)
    if not YLABEL_PER_BLOCK:
        fig.supylabel(f"Reconstruction-derived volume [{_unit_label()}]",
                      fontsize=AXIS_LABEL_FONTSIZE, x=SUPYLABEL_X,
                      y=0.5 * (BOTTOM_MARGIN + TOP_MARGIN))
    _add_region_labels(fig, axes, regions)
    return fig
 
 
def _add_region_labels(fig, axes, regions) -> None:
    """
    Rotated region name spanning the two cohort rows of its block, plus a rule
    separating consecutive blocks.
 
    Positions are read from the axes after layout, so the labels follow the
    panels instead of being pinned to hard-coded coordinates.
    """
    k = len(COHORT_ROWS)
    for b, region in enumerate(regions):
        top = axes[b * k][0].get_position()
        bottom = axes[b * k + k - 1][0].get_position()
        y = 0.5 * (top.y1 + bottom.y0)
        # Pinned to its own lane rather than offset from the axes, which is
        # what previously drove it into the y-axis labels.
        fig.text(REGION_LABEL_X, y, REGION_TITLES.get(region, region),
                 rotation=90, ha="center", va="center",
                 fontsize=REGION_FONTSIZE, fontweight="bold")
        # One axis name per block, centred on that block, so the units are
        # never far from the panels they describe.
        if YLABEL_PER_BLOCK:
            fig.text(SUPYLABEL_X, y,
                     f"Reconstruction-derived volume [{_unit_label()}]",
                     rotation=90, ha="center", va="center",
                     fontsize=AXIS_LABEL_FONTSIZE)
        if XLABEL_PER_BLOCK:
            last_col = axes[b * k][len(METHODS) - 1].get_position()
            fig.text(0.5 * (bottom.x0 + last_col.x1),
                     bottom.y0 - 0.018,
                     f"Reference volume [{_unit_label()}]",
                     ha="center", va="top", fontsize=AXIS_LABEL_FONTSIZE)
        if b > 0:
            prev = axes[b * k - 1][0].get_position()
            last_col = axes[b * k][len(METHODS) - 1].get_position()
            y_rule = 0.5 * (prev.y0 + top.y1)
            fig.add_artist(plt.Line2D([REGION_LABEL_X + 0.01, last_col.x1],
                                      [y_rule, y_rule],
                                      transform=fig.transFigure,
                                      color="0.75", lw=0.9))
 
 
def write_region_group_figures(m_uw, m_mad, out_dir: str, pearson_df=None,
                               groups=None, names=None) -> list:
    """One figure per group of regions. Returns the written paths."""
    groups = REGION_GROUPS if groups is None else list(groups)
    names = GROUP_NAMES if names is None else list(names)
    paths = []
    for i, regions in enumerate(groups):
        present = [r for r in regions
                   if (m_uw is not None and r in set(m_uw["Label"]))
                   or (m_mad is not None and r in set(m_mad["Label"]))]
        if not present:
            print(f"[regionfig] group {i + 1} has no data, skipped")
            continue
        if len(present) < len(regions):
            missing = [r for r in regions if r not in present]
            print(f"[regionfig] group {i + 1}: no data for {missing}")
        fig = make_region_group_figure(m_uw, m_mad, present, pearson_df)
        tag = names[i] if i < len(names) else _slug("_".join(present))
        paths += save_figure(fig, out_dir,
                             f"fig_volume_correspondence_{i + 1}_{tag}")
    print(f"[regionfig] {len(paths) // 2} figure(s); units = {UNITS}; "
          f"limits = {LIMITS_SCOPE}")
    return paths