"""
Replacement for the label-filtering / merge / hemisphere-combining stage of
build_volume_correlations_joint.py.

Procedure
---------
1. Whitelist. Retain only the SegIds in LABEL_NAMES, in both the prediction and
   the reference. Everything else (cerebellum, brainstem, CSF, accumbens,
   ventral DC, WM-hypointensities, inferior lateral ventricle, ...) is removed
   before any comparison is formed.

2. One-to-one error, lateralized. For each specimen and each whitelisted SegId
   present in the prediction, the error is computed against the SAME SegId in
   that specimen's reference. The join is an inner join on (Subject, SegId), so
   reference structures whose hemisphere was not reconstructed have no
   counterpart and are dropped.

3. Hemisphere averaging. The per-hemisphere errors are averaged into the
   canonical bilateral region name, over whichever hemispheres were matched:
   one hemisphere yields that hemisphere's error unchanged, two hemispheres
   yield their unweighted mean.

Ordering note
-------------
The collapse in step 3 acts on the ERROR, not on the volume. Because
mean|V_h - R_h| >= |mean(V_h) - mean(R_h)|, collapsing volumes first would let
opposite-signed hemispheric errors cancel. Consequently the output satisfies
AbsErr >= |Diff|, and AbsErr must NOT be recomputed downstream as Diff.abs().
See the replacement for add_normalized_error() at the end of this file.

Output contract
---------------
Columns consumed by the rest of the script are preserved:
    Subject, Method, Distance, Label, Volume_mm3, Ref_mm3, Diff,
    AbsErr, AbsErr_rel
plus the diagnostics: Region, Hemis, n_hemi, RelErr.
Volume_mm3 and Ref_mm3 are hemisphere MEANS, not sums, so that unilateral and
bilateral specimens occupy the same magnitude scale on the log10 axes of
make_joint_figure() and make_concordance_figure().
"""

from __future__ import annotations

import re
import numpy as np
import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================

LABEL_NAMES = {
    2: "WM", 3: "Cortex", 4: "Ventricle", 10: "Thalamus", 11: "Caudate",
    12: "Putamen", 13: "Pallidum", 17: "Hippocampus", 18: "Amygdala",
    41: "WM", 42: "Cortex", 43: "Ventricle", 49: "Thalamus", 50: "Caudate",
    51: "Putamen", 52: "Pallidum", 53: "Hippocampus", 54: "Amygdala",
}
ALLOWED_SEGIDS = set(LABEL_NAMES)

# FreeSurfer aseg convention: 1-39 left / midline, 40-59 right.
HEMI_OF = {sid: ("L" if sid < 40 else "R") for sid in ALLOWED_SEGIDS}

VERBOSE_QC = True

# Optional. Off by default, since it is not part of the requested procedure.
# When set to "method", an observation is kept only if it exists for every
# method within its slab distance; this is what makes the arms of the paired
# Wilcoxon tests in pvalue() rest on the same anatomical support.
COMPLETE_CASE_SCOPE = None


# =============================================================================
# STEP 1 -- WHITELIST
# =============================================================================
def apply_label_whitelist(df: pd.DataFrame) -> pd.DataFrame:
    """Keep whitelisted SegIds; attach canonical region name and hemisphere.

    Laterality is retained as a column rather than erased from the label,
    because the SegId is the join key that makes step 2 one-to-one.
    """
    if df.empty:
        return df
    if "SegId" not in df.columns:
        raise KeyError("apply_label_whitelist requires a 'SegId' column.")
    out = df[df["SegId"].isin(ALLOWED_SEGIDS)].copy()
    out["Region"] = out["SegId"].map(LABEL_NAMES)
    out["Hemi"] = out["SegId"].map(HEMI_OF)
    return out


# =============================================================================
# STEP 2 -- LATERALIZED ONE-TO-ONE ERROR
# =============================================================================
def merge_with_ref(df: pd.DataFrame, ref_df: pd.DataFrame) -> pd.DataFrame:
    """Join prediction to reference on (Subject, SegId) and form the errors.

    Inner join semantics implement the intended asymmetry: a structure present
    in the reference but not predicted for that specimen is dropped, since it
    has no counterpart to be compared against.
    """
    ref_long = (ref_df.rename(columns={"Volume_mm3": "Ref_mm3"})
                      [["Subject", "SegId", "Ref_mm3"]]
                      .drop_duplicates(["Subject", "SegId"]))

    probe = df.merge(ref_long, on=["Subject", "SegId"], how="left")
    orphan = probe[probe["Ref_mm3"].isna()]
    if not orphan.empty and VERBOSE_QC:
        pairs = orphan[["Subject", "SegId", "Region", "Hemi"]].drop_duplicates()
        print(f"[merge] {len(pairs)} predicted (Subject, SegId) pair(s) have no "
              f"reference counterpart and were dropped:")
        print(pairs.to_string(index=False))

    m = probe[probe["Ref_mm3"].notna()].copy()
    m = m[(m["Ref_mm3"] > 0) & (m["Volume_mm3"] > 0)].copy()

    # Per-hemisphere magnitude errors.
    m["Diff_hemi"] = m["Volume_mm3"] - m["Ref_mm3"]
    m["AbsErr_hemi"] = m["Diff_hemi"].abs()
    m["RelErr_hemi"] = m["Diff_hemi"] / m["Ref_mm3"]          # signed
    m["AbsErr_rel_hemi"] = m["AbsErr_hemi"] / m["Ref_mm3"]    # magnitude

    # Backwards-compatible alias, in case any diagnostic consumes it pre-collapse.
    m["Diff"] = m["Diff_hemi"]
    return m


# =============================================================================
# STEP 3 -- HEMISPHERE AVERAGING OF THE ERRORS
# =============================================================================
def collapse_hemispheres(m: pd.DataFrame) -> pd.DataFrame:
    """Average the per-hemisphere errors into the canonical bilateral region.

    Unweighted mean over the matched hemispheres: a specimen contributing one
    hemisphere passes that hemisphere's error through unchanged, a specimen
    contributing both contributes their arithmetic mean. Volumes are averaged
    on the same hemisphere set so that Ref_mm3 remains on a single-hemisphere
    scale for every specimen.
    """
    if m.empty:
        return m

    keys = ["Subject", "Method", "Distance", "Region"]
    g = (m.groupby(keys, as_index=False)
           .agg(Volume_mm3=("Volume_mm3", "mean"),
                Ref_mm3=("Ref_mm3", "mean"),
                Diff=("Diff_hemi", "mean"),
                AbsErr=("AbsErr_hemi", "mean"),
                RelErr=("RelErr_hemi", "mean"),
                AbsErr_rel=("AbsErr_rel_hemi", "mean"),
                n_hemi=("Hemi", "nunique"),
                Hemis=("Hemi", lambda s: "".join(sorted(set(s))))))

    g["Label"] = g["Region"]  # column name expected downstream

    # Sanity invariant of the ordering: cancellation cannot occur.
    if VERBOSE_QC:
        viol = int((g["AbsErr"] < g["Diff"].abs() - 1e-9).sum())
        if viol:
            print(f"[collapse] WARNING: {viol} row(s) violate AbsErr >= |Diff|; "
                  f"inspect the aggregation.")
    return g


# =============================================================================
# OPTIONAL -- COMPLETE-CASE RESTRICTION (disabled by default)
# =============================================================================
def restrict_complete_cases(m: pd.DataFrame, scope=COMPLETE_CASE_SCOPE) -> pd.DataFrame:
    """Keep only (Subject, SegId) units observed in every cell of the design.

    Applied between step 2 and step 3 when enabled. Relevant only if the paired
    p-value table is reported; leaving it off maximises the number of retained
    observations, at the cost of methods being compared over slightly different
    anatomical supports for specimens with partial coverage.
    """
    if m.empty or scope is None:
        return m

    obs = m.drop_duplicates(["Subject", "SegId", "Method", "Distance"])

    if scope == "method":
        n_req = m["Method"].nunique()
        cnt = obs.groupby(["Subject", "SegId", "Distance"]).size()
        keep = set(cnt[cnt == n_req].index)
        idx = pd.MultiIndex.from_frame(m[["Subject", "SegId", "Distance"]])
    elif scope == "method+distance":
        n_req = m[["Method", "Distance"]].drop_duplicates().shape[0]
        cnt = obs.groupby(["Subject", "SegId"]).size()
        keep = set(cnt[cnt == n_req].index)
        idx = pd.MultiIndex.from_frame(m[["Subject", "SegId"]])
    else:
        raise ValueError(f"Unknown COMPLETE_CASE_SCOPE: {scope!r}")

    mask = idx.isin(keep)
    if (~mask).any() and VERBOSE_QC:
        lost = m.loc[~mask, ["Subject", "SegId", "Region", "Hemi"]].drop_duplicates()
        print(f"[complete-case:{scope}] dropped {int((~mask).sum())} row(s) "
              f"covering {len(lost)} (Subject, SegId) pair(s):")
        print(lost.to_string(index=False))
    return m[mask].copy()


# =============================================================================
# QC
# =============================================================================
_HEMI_TOKEN = re.compile(r"(?:^|[-_])(left|right|lh|rh)(?:$|[-_])", re.IGNORECASE)


def subject_hemi_token(subject: str):
    """Laterality declared by the specimen identifier, or None."""
    mt = _HEMI_TOKEN.search(subject)
    if not mt:
        return None
    return "L" if mt.group(1).lower() in ("left", "lh") else "R"


def qc_report(m: pd.DataFrame, tag: str) -> None:
    """Report the matched support per specimen and flag laterality conflicts.

    Because the join keys on SegId, a reconstruction that emits left-hemisphere
    labels for a right-hemisphere specimen would be compared against the
    contralateral reference without raising an error. This is the detector.
    """
    if m.empty or not VERBOSE_QC:
        return
    print(f"[{tag}] matched support per specimen:")
    for subject, d in m.groupby("Subject"):
        hemis = sorted(set(d["Hemi"]))
        declared = subject_hemi_token(subject)
        note = ""
        if declared is not None and declared not in hemis:
            note = f"  <-- CONFLICT: identifier says {declared}, data has {hemis}"
        elif declared is not None and len(hemis) > 1:
            note = f"  <-- note: identifier says {declared}, both hemispheres matched"
        print(f"    {subject:<24} hemispheres={hemis} "
              f"segids={d['SegId'].nunique()}{note}")


# =============================================================================
# PIPELINE ENTRY POINT
# =============================================================================
def process(df: pd.DataFrame, ref_df: pd.DataFrame, tag: str) -> pd.DataFrame:
    df = apply_label_whitelist(df)          # step 1
    ref_df = apply_label_whitelist(ref_df)  # step 1
    if df.empty or ref_df.empty:
        raise RuntimeError(f"All {tag} rows removed by the SegId allowlist.")

    m = merge_with_ref(df, ref_df)          # step 2
    if m.empty:
        raise RuntimeError(f"{tag}: no (Subject, SegId) pair matched the reference.")
    qc_report(m, tag)

    m = restrict_complete_cases(m, COMPLETE_CASE_SCOPE)  # optional
    m = collapse_hemispheres(m)                          # step 3

    if VERBOSE_QC:
        n_bilat = int((m["n_hemi"] == 2).sum())
        print(f"[{tag}] final: {len(m)} region observations, "
              f"{m['Subject'].nunique()} specimen(s), "
              f"{m['Label'].nunique()} region(s); "
              f"{n_bilat} bilateral / {len(m) - n_bilat} unilateral.")
    return m


# =============================================================================
# REPLACEMENT FOR add_normalized_error()
# =============================================================================
def add_normalized_error(m_all: pd.DataFrame) -> pd.DataFrame:
    """Pass-through. AbsErr and AbsErr_rel are already the hemisphere-averaged
    magnitudes produced by collapse_hemispheres(); recomputing them here as
    Diff.abs() / Ref_mm3 would silently revert to the volume-first collapse and
    reintroduce inter-hemispheric cancellation."""
    m = m_all.copy()
    missing = {"AbsErr", "AbsErr_rel"} - set(m.columns)
    if missing:
        raise KeyError(f"Expected error columns absent: {sorted(missing)}. "
                       f"process() must run before add_normalized_error().")
    return m