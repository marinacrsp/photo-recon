"""
Central configuration for the joint UW + MADRC volume analysis.

Everything that a reader might want to change without reading code lives here:
cohort and method naming, label selection, palettes, statistical conventions and
figure defaults. No module below this one defines a constant of its own.
"""

from __future__ import annotations

import itertools

import matplotlib.pyplot as plt

# =============================================================================
# COHORTS, METHODS, SECTIONS
# =============================================================================
METHODS = ["Photo-recon", "Tricubic", "Imputed"]
METHOD_DISPLAY = {
    "Photo-recon": "3D reconstruction \n of slab photographs",
    "Tricubic": "Cubic",
    "Imputed": "Imputed",
}
METHOD_ABBR = {"Photo-recon": "PR", "Tricubic": "Cubic", "Imputed": "UNet"}

# Ordered largest-first: later draws sit on top of earlier ones in the figures.
DISTANCES = ["12mm", "8mm", "4mm"]
MADRC_LABEL = "MADRC"

# Display order in the tables and figures, smallest slab distance first.
SECTION_ORDER_TABLE = [MADRC_LABEL, "4mm", "8mm", "12mm"]
SECTION_HEADER = {
    MADRC_LABEL: "MADRC",
    "4mm": "UW -- 4 mm",
    "8mm": "UW -- 8 mm",
    "12mm": "UW -- 12 mm",
}

PANEL_LETTERS = ["(a)", "(b)", "(c)"]

# =============================================================================
# LABEL SELECTION
# =============================================================================
# Selection is by original SegId. Left and right identifiers sharing a name
# collapse to one bilateral region.
LABEL_NAMES = {
    2: "WM", 3: "Cortex", 4: "Ventricle", 10: "Thalamus", 11: "Caudate",
    12: "Putamen", 13: "Pallidum", 17: "Hippocampus", 18: "Amygdala",
    41: "WM", 42: "Cortex", 43: "Ventricle", 49: "Thalamus", 50: "Caudate",
    51: "Putamen", 52: "Pallidum", 53: "Hippocampus", 54: "Amygdala",
}
ALLOWED_SEGIDS = set(LABEL_NAMES)
RENAME_TO_CANONICAL = True

# How the two hemispheres are combined into one bilateral value.
#   "sum"  -> total volume of the structure, the conventional definition
#   "mean" -> mean hemisphere volume, half the total
# The same rule is applied to the reference, so correlations are unaffected;
# absolute errors and the mL axis of the error table are not.
HEMISPHERE_COMBINE = "sum"

# =============================================================================
# QUALITY CONTROL
# =============================================================================
# Rows whose predicted-to-reference volume ratio falls below this threshold are
# treated as segmentation failures. 0.0 disables the filter and keeps every row.
# Failures are always counted and reported regardless of the threshold.
SEG_FAILURE_RATIO = 0.0

# =============================================================================
# STATISTICS
# =============================================================================
# Space in which correlations are computed. Must match what the captions claim.
CORR_SPACE = "raw"                   # "raw" | "log10"

PVALUE_PAIRS = list(itertools.combinations(METHODS, 2))
TEST = "wilcoxon"                    # "wilcoxon" (paired) | "ranksum"

# Multiplicity control for the pairwise error table.
#   "bonferroni" | "bh" | "none"
# The caption is generated from this value, so the printed quantity and its
# label cannot disagree.
CORRECTION = "bonferroni"
ALPHA = 0.05
BOLD_SIGNIFICANT = False

MIN_N = 3                            # fewer paired observations prints a dash

# =============================================================================
# FIGURE STYLE
# =============================================================================
DISTANCE_COLORS = {"4mm": "#d2691e", "8mm": "#e9967a", "12mm": "#ffcba4"}
MADRC_COLOR = "#A894EE"
FIT_COLOR = "#463F61"

POINT_ORDER = ["4mm", "8mm", "12mm"]
LINE_ORDER = ["y = x", "LS Fit"]

FIG_FORMATS = ("pdf", "svg")
FIG_DPI = 300
BASE_FONTSIZE = 20


def apply_style() -> None:
    """Apply the shared rcParams. Called once by the pipeline."""
    plt.rcParams.update({
        "font.size": BASE_FONTSIZE,
        "axes.labelsize": BASE_FONTSIZE,
        "xtick.labelsize": BASE_FONTSIZE,
        "ytick.labelsize": BASE_FONTSIZE,
        "legend.fontsize": BASE_FONTSIZE,
        "font.family": ["Carlito", "Calibri", "DejaVu Sans"],
    })


def section_color(section: str) -> str:
    """One colour rule, shared by every figure in the package."""
    if section == MADRC_LABEL:
        return MADRC_COLOR
    return DISTANCE_COLORS.get(section, "#7f7f7f")