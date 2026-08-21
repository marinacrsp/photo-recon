"""
=============================================================================
PATCH -- per-label normalised concordance scatter
=============================================================================

Extends `build_volume_correlations_joint.py`. Requires METHODS, METHOD_DISPLAY,
DISTANCES, MADRC_LABEL, DISTANCE_COLORS, MADRC_COLOR, PANEL_LETTERS.

Why normalise per label
-----------------------
On the log-log concordance plot the points lie close to the identity mainly
because regional volumes span roughly three orders of magnitude: the ordering
of structure sizes dominates, and the within-region agreement, which is what
the per-region Pearson table reports, is invisible. Removing the between-region
scale places every region on common axes and makes the figure the visual
counterpart of the table.

The scaling rule
----------------
BOTH axes are transformed by ONE scaler, estimated from the REFERENCE volumes
of that label only. Scaling the predicted and the reference values by their own
ranges would force both onto [0, 1] independently and would map a method with a
systematic volume bias onto the identity line, fabricating agreement. The
scaler is therefore fitted on the reference alone and applied unchanged to the
prediction.

Scalers are estimated per (cohort, label) over the unique reference value of
each specimen, so that all slab distances of a cohort share one scaler and the
panels remain comparable.
=============================================================================
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
# =============================================================================
# CONFIGURATION
# =============================================================================
#   "ref_robust" -> (V - median_ref) / IQR_ref. Resistant to the segmentation
#                   failures visible in the log-log figure. Recommended.
#   "ref_z"      -> (V - mean_ref) / sd_ref. The natural companion to Pearson r,
#                   but the mean and sd are themselves moved by a failure.
#   "ref_mean"   -> V / mean_ref. Centred on 1, reads as a fraction of the
#                   typical volume of that structure.
#   "ref_minmax" -> (V - min_ref) / (max_ref - min_ref). Literal 0-1 axes, but a
#                   single failed segmentation sets the minimum and compresses
#                   the rest of the range into a sliver.
NORM_MODE = "ref_robust"

# Axis limits in normalised units. None lets the data decide, with padding.
NORM_LIMITS = None                   # e.g. (-3.0, 3.0) or (0.0, 1.0)
NORM_PAD_FRAC = 0.05

# Points outside the axis window are drawn on the border as open triangles
# rather than silently dropped, so a failure cannot disappear from the figure.
NORM_MARK_CLIPPED = True

# Colour encoding of the point cloud.
#   "distance" -> matches the current figure and the forest plot
#   "region"   -> reveals which structure carries the disagreement, which is
#                 the question the normalised view is able to answer
NORM_COLOR_BY = "distance"           # "distance" | "region"

# Point size and opacity per cohort. UW panels carry roughly three times the
# points of MADRC, so a single setting either saturates the UW core or leaves
# the MADRC cloud too faint.
NORM_POINT_STYLE = {
    "UW":    dict(s=34, alpha=0.38),
    "MADRC": dict(s=44, alpha=0.55),
}
NORM_POINT_SIZE = 30        # fallback
NORM_POINT_ALPHA = 0.40
NORM_FONTSIZE = 20
NORM_SHOW_FIT = False                # a fit across labels mixes regions; off
NORM_ANNOTATE_CCC = True             # Lin concordance, computed on normalised units
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

DISTANCE_COLORS = {"4mm": "#d2691e" , "8mm": "#e9967a", "12mm": "#ffcba4"}
MADRC_COLOR = "#A894EE"

NORM_ANNOT = "fisher_r"              # "fisher_r" | "ccc" | "none"
 
NORM_ANNOT_CI = False                # append the interval to the annotation
NORM_ANNOT_FONTSIZE = 20
NORM_ANNOT_DY = 0.075                # vertical step between annotation lines
 

# =============================================================================
# SCALERS
# =============================================================================
def _scaler_params(ref: np.ndarray) -> dict:
    ref = np.asarray(ref, dtype=float)
    ref = ref[np.isfinite(ref)]
    if ref.size < 3:
        return dict(loc=np.nan, scale=np.nan)
    if NORM_MODE == "ref_robust":
        q1, q3 = np.percentile(ref, [25, 75])
        return dict(loc=float(np.median(ref)), scale=float(q3 - q1))
    if NORM_MODE == "ref_z":
        return dict(loc=float(np.mean(ref)), scale=float(np.std(ref, ddof=1)))
    if NORM_MODE == "ref_mean":
        return dict(loc=0.0, scale=float(np.mean(ref)))
    return dict(loc=float(np.min(ref)),
                scale=float(np.max(ref) - np.min(ref)))


def build_scalers(m: pd.DataFrame) -> dict:
    """
    label -> {loc, scale}, estimated on the unique reference volume per specimen.

    Ref_mm3 is repeated across methods and slab distances, so it must be
    de-duplicated first or the scaler is fitted on replicated values.
    """
    out = {}
    uniq = m.drop_duplicates(subset=["Subject", "Label"])[
        ["Subject", "Label", "Ref_mm3"]]
    for lab, d in uniq.groupby("Label"):
        p = _scaler_params(d["Ref_mm3"].to_numpy(dtype=float))
        if np.isfinite(p["scale"]) and p["scale"] > 0:
            out[lab] = p
    return out


def apply_scaler(v, lab: str, scalers: dict) -> np.ndarray:
    p = scalers.get(lab)
    v = np.asarray(v, dtype=float)
    if p is None:
        return np.full(v.shape, np.nan)
    return (v - p["loc"]) / p["scale"]


def normalise_frame(m: pd.DataFrame, scalers: dict) -> pd.DataFrame:
    """Add RefN and VolN columns, both under the label's reference scaler."""
    d = m.copy()
    d["RefN"] = np.nan
    d["VolN"] = np.nan
    for lab, idx in d.groupby("Label").groups.items():
        if lab not in scalers:
            continue
        d.loc[idx, "RefN"] = apply_scaler(d.loc[idx, "Ref_mm3"], lab, scalers)
        d.loc[idx, "VolN"] = apply_scaler(d.loc[idx, "Volume_mm3"], lab, scalers)
    return d


# =============================================================================
# CONCORDANCE STATISTIC
# =============================================================================
def lin_ccc(x, y) -> float:
    """
    Lin's concordance correlation coefficient on the normalised units.

    Unlike Pearson r, this penalises departure from the identity line rather
    than from the best-fitting line, so it detects a systematic bias. Computed
    on the pooled normalised cloud it summarises within-region agreement across
    every region at once.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 3:
        return np.nan
    vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
    cov = np.cov(x, y, ddof=1)[0, 1]
    den = vx + vy + (np.mean(x) - np.mean(y)) ** 2
    return float(2.0 * cov / den) if den > 0 else np.nan


# =============================================================================
# FIGURE
# =============================================================================
def _axis_label(kind: str) -> str:
    unit = {"ref_robust": r"(V $-$ med$_{\mathrm{ref}}$) / IQR$_{\mathrm{ref}}$",
            "ref_z": r"(V $-$ $\mu_{\mathrm{ref}}$) / $\sigma_{\mathrm{ref}}$",
            "ref_mean": r"V / $\mu_{\mathrm{ref}}$",
            "ref_minmax": r"min-max scaled"}[NORM_MODE]
    lead = "Pred." if kind == "y" else "Ref."
    return f"{lead} volume, per-region normalised\n{unit}"


def _limits(frames) -> tuple:
    if NORM_LIMITS is not None:
        return NORM_LIMITS
    vals = []
    for d in frames:
        if d is not None and not d.empty:
            vals.append(d["RefN"].to_numpy(dtype=float))
            vals.append(d["VolN"].to_numpy(dtype=float))
    if not vals:
        return (-3.0, 3.0)
    v = np.concatenate(vals)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return (-3.0, 3.0)
    # Robust window: a failed segmentation must not set the axis range. Points
    # outside are drawn on the border instead of expanding the view.
    lo, hi = np.percentile(v, [0.5, 99.5])
    pad = (hi - lo) * NORM_PAD_FRAC
    return float(lo - pad), float(hi + pad)


def _region_palette(labels) -> dict:
    cmap = plt.get_cmap("tab10")
    return {lab: cmap(i % 10) for i, lab in enumerate(sorted(labels))}

# =============================================================================
# LOOKUP INTO THE CORRELATION TABLE
# =============================================================================

def _summary_row(pearson_df: pd.DataFrame, section: str, method: str):
    """
    The section-level summary of the correlation table, as (r, ci_lo, ci_hi).
 
    Accepts either the per-region Pearson frame (RowType in {region, summary,
    fisher_mean}) or the combined metrics frame. Returns NaNs when the section
    has no summary row, so a missing value prints as a dash rather than as a
    silently recomputed number.
    """
    nan3 = (np.nan, np.nan, np.nan)
    if pearson_df is None or not len(pearson_df):
        return nan3
    d = pearson_df[(pearson_df["Section"] == section)
                   & (pearson_df["Method"] == method)]
    if not len(d):
        return nan3
    summ = d[d["RowType"] != "region"]
    if not len(summ):
        return nan3
    row = summ.iloc[0]
    return (float(row.get("r", np.nan)),
            float(row.get("ci_lo", np.nan)),
            float(row.get("ci_hi", np.nan)))
 
 
def _annot_lines(pearson_df, cohort: str, method: str, by_dist: bool):
    """(text, colour) pairs for one panel."""
    out = []
    sections = list(DISTANCES) if by_dist else [MADRC_LABEL]
    # DISTANCES is ordered largest-first for draw order; annotate small-first.
    if by_dist:
        sections = sections[::-1]
    for sec in sections:
        r, lo, hi = _summary_row(pearson_df, sec, method)
        if not np.isfinite(r):
            continue
        label = f"{sec}: " if by_dist else ""
        txt = r"%s$\bar{r}_z$ = %.2f" % (label, r)
        if NORM_ANNOT_CI and np.isfinite(lo) and np.isfinite(hi):
            txt += r" [%.2f, %.2f]" % (lo, hi)
        color = (DISTANCE_COLORS.get(sec, "#333333") if by_dist
                 else MADRC_COLOR)
        out.append((txt, color))
    return out

# =============================================================================
# FIGURE
# =============================================================================
def make_normalised_concordance_figure(m_uw: pd.DataFrame, m_mad: pd.DataFrame,
                                       labels=None, pearson_df=None):
    """
    Rows = cohort, columns = method, points = one specimen and region.
 
    `pearson_df` supplies the annotation. Without it the panels are drawn
    unannotated rather than annotated with a recomputed value.
    """
    scal_uw = build_scalers(m_uw) if not m_uw.empty else {}
    scal_mad = build_scalers(m_mad) if not m_mad.empty else {}

    # Normalize the volume magnitudes based on the criterion defined upstairs, default is Robust
    n_uw = normalise_frame(m_uw, scal_uw) if not m_uw.empty else m_uw
    n_mad = normalise_frame(m_mad, scal_mad) if not m_mad.empty else m_mad
 
    if NORM_ANNOT == "fisher_r" and pearson_df is None:
        print("[normconc] warning: NORM_ANNOT='fisher_r' but no pearson_df was "
              "passed; panels will be drawn without annotation")
 
    rows = [("UW", n_uw, True), ("MADRC", n_mad, False)]
    lo, hi = _limits([n_uw, n_mad])
    regions = sorted(set(labels) if labels is not None
                     else set(pd.concat([n_uw, n_mad])["Label"]))
    rpal = _region_palette(regions)
 
    fig, axes = plt.subplots(2, 3, figsize=(18, 12.6), sharex=True, sharey=True)
 
    for r_i, (coh, mm, by_dist) in enumerate(rows):
        for c, method in enumerate(METHODS):
            ax = axes[r_i, c]
            ax.plot([lo, hi], [lo, hi], ls="--", color="black", lw=1.4,
                    alpha=0.6, zorder=1, label="y = x")
            sub = mm[mm["Method"] == method] if not mm.empty else mm
            if sub is not None and not sub.empty:
                if NORM_COLOR_BY == "region":
                    groups = [(sub[sub["Label"] == lab], rpal[lab], lab)
                              for lab in regions]
                elif by_dist:
                    groups = [(sub[sub["Distance"] == d],
                               DISTANCE_COLORS.get(d, "#7f7f7f"), d)
                              for d in DISTANCES]
                else:
                    groups = [(sub, MADRC_COLOR, MADRC_LABEL)]

                style = NORM_POINT_STYLE.get(coh,
                                             dict(s=NORM_POINT_SIZE,
                                                  alpha=NORM_POINT_ALPHA))
                for dd, color, plabel in groups:
                    if dd is None or dd.empty:
                        continue
                    x = dd["RefN"].to_numpy(dtype=float)
                    y = dd["VolN"].to_numpy(dtype=float)
                    ok = np.isfinite(x) & np.isfinite(y)
                    inside = ok & (x >= lo) & (x <= hi) & (y >= lo) & (y <= hi)
                    ax.scatter(x[inside], y[inside], s=style["s"],
                               alpha=style["alpha"], color=color,
                               edgecolors="none", label=plabel, zorder=3)
                    if NORM_MARK_CLIPPED and (ok & ~inside).any():
                        ax.scatter(np.clip(x[ok & ~inside], lo, hi),
                                   np.clip(y[ok & ~inside], lo, hi),
                                   s=style["s"] + 24, marker="^",
                                   facecolors="none", edgecolors=color,
                                   linewidths=1.2, zorder=4)
 
                if NORM_ANNOT == "fisher_r":
                    for i, (txt, color) in enumerate(
                            _annot_lines(pearson_df, coh, method, by_dist)):
                        ax.text(0.04, 0.965 - NORM_ANNOT_DY * i, txt,
                                transform=ax.transAxes,
                                fontsize=NORM_ANNOT_FONTSIZE-2, va="top",
                                color='#333333')
                elif NORM_ANNOT == "ccc":
                    ccc = lin_ccc(sub["RefN"], sub["VolN"])
                    if np.isfinite(ccc):
                        ax.text(0.04, 0.965, r"$\rho_c$ = %.2f" % ccc,
                                transform=ax.transAxes,
                                fontsize=NORM_ANNOT_FONTSIZE-2, va="top",
                                color="#333333")
 
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, alpha=0.25, ls=":")
            ax.set_axisbelow(True)
            if r_i == 0:
                ax.set_title(f"{PANEL_LETTERS[c]} {METHOD_DISPLAY[method]}",
                             fontsize=NORM_FONTSIZE, pad=8)
            if c == 0:
                ax.set_ylabel(f"{coh}\nPred. volume",
                              fontsize=NORM_FONTSIZE)
 
    # One centred x-label instead of three, and a legend with room below it.
    fig.supxlabel("Ref. volume", fontsize=NORM_FONTSIZE,
                  y=0.085)
 
    handles, names = {}, []
    for ax in np.ravel(axes):
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l and not l.startswith("_") and l not in handles:
                handles[l] = h
                names.append(l)
    # Small-to-large slab distance, then MADRC, then the identity line.
    order = [d for d in reversed(list(DISTANCES)) if d in handles]
    order += [l for l in names if l not in order and l != "y = x"]
    order += ["y = x"] if "y = x" in handles else []
    if order:
        leg = fig.legend([handles[l] for l in order], order, loc="lower center",
                         ncol=min(len(order), 6), frameon=False,
                         fontsize=NORM_FONTSIZE, bbox_to_anchor=(0.5, 0.005),
                         markerscale=2.4)
        for lh in leg.legend_handles:
            if hasattr(lh, "set_alpha"):
                lh.set_alpha(1.0)
 
    fig.subplots_adjust(wspace=0.03, hspace=0.08, left=0.11, bottom=0.17)
    return fig


def write_normalised_concordance(m_uw, m_mad, out_dir: str,
                                 labels=None, pearson_df=None) -> list:
    os.makedirs(out_dir, exist_ok=True)

    fig = make_normalised_concordance_figure(m_uw, m_mad, labels, pearson_df=pearson_df)
    if fig is None:
        return []
    paths = []
    for ext in ("pdf", "svg"):
        p = os.path.join(out_dir,
                         f"task_2.1_uwmadrc_volume_concordance_normalised.{ext}")
        fig.savefig(p, bbox_inches="tight", dpi=300)
        paths.append(p)
    plt.close(fig)
    print(f"[normconc] mode = {NORM_MODE}; colour = {NORM_COLOR_BY}; "
          f"clipped points marked = {NORM_MARK_CLIPPED}")
    return paths