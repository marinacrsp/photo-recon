"""
Visual comparison of the 4 mm ground truth against the 8 mm and 12 mm imputations.

Two tools:

  profile(...)     MAE against the ground truth for every slab, both arms, with
                   retained slabs marked. The single most diagnostic figure: the
                   curves must dip to near zero at each arm's retained slabs. If
                   they do not, the geometry is wrong and no metric is valid.

  show_period(...) One full interleaving period as an image grid. With factors 2
                   and 3 the pattern repeats every 6 slabs, so a 7-row window
                   shows every combination of retained and withheld.

Geometry comes from the affines. Slab axes are derived, not assumed, because
they differ between subjects (axis 1 for 18-0086, axis 2 for 17-0333), and an
isotropic volume has no thickest axis to identify.
"""

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

MASK_THRESHOLD = 5
OUTDIR = "./compare"


# =============================================================================
# Geometry
# =============================================================================

def voxel_sizes(aff):
    return np.sqrt((aff[:3, :3] ** 2).sum(axis=0))


def slab_axis(aff):
    """Thickest axis. Valid for the photo recons; degenerate for isotropic."""
    return int(np.argmax(voxel_sizes(aff)))


def slab_axis_iso(aff_gt, ax_gt, aff_iso):
    """Isotropic volumes have equal voxel sizes, so take the axis from the pair."""
    col = (np.linalg.inv(aff_iso) @ aff_gt)[:3, ax_gt]
    return int(np.argmax(np.abs(col))), float(col[int(np.argmax(np.abs(col)))])


class Arm:
    """One thickness condition: its factor, grid offset and isotropic volume."""

    def __init__(self, tag, vol_iso, aff_iso, aff_lo, aff_gt, ax_gt, t4):
        self.tag = tag
        self.vol = vol_iso
        self.ax, self.slope = slab_axis_iso(aff_gt, ax_gt, aff_iso)
        self.t_src = float(voxel_sizes(aff_lo)[slab_axis(aff_lo)])
        self.factor = int(round(self.t_src / t4))
        self.off = (self.t_src - 1.0) / 2.0
        self.t4 = t4
        print(f"  {tag:>5}: t_src {self.t_src:.4f}  factor {self.factor}  "
              f"offset {self.off:.4f}  iso axis {self.ax} slope {self.slope:+.3f}"
              f"{'  FLIPPED' if self.slope < 0 else ''}")

    def r(self, idx):
        """Exact, generally fractional, isotropic index of ground-truth slab idx."""
        return self.t4 * idx + self.off

    def retained(self, idx):
        return idx % self.factor == 0

    def plane(self, idx):
        """Interpolated plane, or None if outside the reconstructed extent."""
        r = self.r(idx)
        lo = int(np.floor(r)); hi = lo + 1; w = r - lo
        if lo < 0 or hi >= self.vol.shape[self.ax]:
            return None
        a = np.take(self.vol, lo, axis=self.ax).astype(np.float64)
        b = np.take(self.vol, hi, axis=self.ax).astype(np.float64)
        return (1 - w) * a + w * b

    def dist_to_retained(self, idx):
        """Millimetres to the nearest retained slab."""
        return abs(idx - self.factor * round(idx / self.factor)) * self.t4


def build_arms(aff_gt, arms_spec):
    ax_gt = slab_axis(aff_gt)
    t4 = float(voxel_sizes(aff_gt)[ax_gt])
    print(f"ground truth: slab axis {ax_gt}, thickness {t4:.4f} mm")
    return ax_gt, t4, [Arm(tag, v, ai, al, aff_gt, ax_gt, t4)
                       for tag, v, ai, al in arms_spec]


# =============================================================================
# Helpers
# =============================================================================

def clip8(a):
    return np.clip(a, 0, 255).astype(np.uint8)


def gt_plane(I_gt, idx, ax_gt):
    return np.take(I_gt, idx, axis=ax_gt).astype(np.float64)


def diff_map(gt, pred):
    d = np.abs(gt - pred)
    return d.mean(axis=-1) if d.ndim == 3 else d


def tissue_fraction(gt):
    m = gt.sum(axis=-1) > MASK_THRESHOLD if gt.ndim == 3 else gt > MASK_THRESHOLD
    return float(m.mean()), m


# =============================================================================
# Tool 1: MAE profile
# =============================================================================

def profile(I_gt, ax_gt, arms, subject_id, outdir=OUTDIR, masked=True):
    n = I_gt.shape[ax_gt]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})
    colors = {"8mm": "#1f77b4", "12mm": "#d62728"}
    fracs = np.zeros(n)

    for arm in arms:
        mae = np.full(n, np.nan)
        for idx in range(n):
            p = arm.plane(idx)
            gt = gt_plane(I_gt, idx, ax_gt)
            fracs[idx], m = tissue_fraction(gt)
            if p is None or p.shape != gt.shape:
                continue
            if masked and m.any():
                mm = m if gt.ndim == 2 else np.broadcast_to(m[..., None], gt.shape)
                mae[idx] = float(np.abs(gt - p)[mm].mean())
            else:
                mae[idx] = float(np.abs(gt - p).mean())

        c = colors.get(arm.tag, None)
        ax1.plot(np.arange(n), mae, "-", lw=1.2, color=c, label=f"{arm.tag} imputation")
        kept = np.arange(0, n, arm.factor)
        ax1.plot(kept, mae[kept], "o", ms=6, color=c, mfc="white", mew=1.6,
                 label=f"{arm.tag} retained (control)")
        keptv = mae[kept][np.isfinite(mae[kept])]
        wh = np.array([mae[i] for i in range(n) if i % arm.factor], float)
        wh = wh[np.isfinite(wh)]
        print(f"  {arm.tag:>5}: retained MAE {keptv.mean():.3f}  "
              f"withheld MAE {wh.mean():.3f}  ratio {wh.mean()/max(keptv.mean(),1e-9):.2f}"
              f"  ({int(np.isnan(mae).sum())} slabs outside extent)")

    ax1.set_ylabel("MAE vs 4 mm ground truth" + (" (tissue mask)" if masked else ""))
    ax1.legend(fontsize=8, ncol=2)
    ax1.grid(alpha=0.25)
    ax1.set_title(f"{subject_id}: open circles are real photographs the network "
                  f"never predicted, so they must sit near zero", fontsize=10)

    ax2.fill_between(np.arange(n), fracs, color="0.6", lw=0)
    ax2.set_ylabel("tissue\nfraction", fontsize=8)
    ax2.set_xlabel("ground-truth slab index")
    ax2.grid(alpha=0.25)

    fig.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    p = os.path.join(outdir, f"profile_{subject_id}.png")
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return p


# =============================================================================
# Tool 2: one interleaving period as a grid
# =============================================================================

def show_period(I_gt, ax_gt, arms, subject_id, start=0, outdir=OUTDIR):
    """Rows are consecutive slabs covering one full period (lcm of the factors)."""
    per = int(np.lcm.reduce([a.factor for a in arms]))
    n = I_gt.shape[ax_gt]
    idxs = [k for k in range(start, start + per + 1) if k < n]

    ncol = 1 + 2 * len(arms)
    fig, axes = plt.subplots(len(idxs), ncol,
                             figsize=(2.35 * ncol, 2.3 * len(idxs)), squeeze=False)

    # shared difference scale across the whole figure
    alld = []
    for k in idxs:
        gt = gt_plane(I_gt, k, ax_gt)
        for arm in arms:
            p = arm.plane(k)
            if p is not None and p.shape == gt.shape:
                alld.append(diff_map(gt, p).ravel())
    vmax = max(float(np.percentile(np.concatenate(alld), 99)), 1.0) if alld else 1.0

    for row, k in enumerate(idxs):
        gt = gt_plane(I_gt, k, ax_gt)
        axes[row][0].imshow(clip8(gt)[..., :3] if gt.ndim == 3 else clip8(gt),
                            **({} if gt.ndim == 3 else dict(cmap="gray", vmin=0, vmax=255)))
        axes[row][0].set_xlabel(f"slab {k}", fontsize=7)
        if row == 0:
            axes[row][0].set_title("4 mm ground truth", fontsize=9)

        tags = []
        for a_i, arm in enumerate(arms):
            p = arm.plane(k)
            col_img, col_dif = 1 + 2 * a_i, 2 + 2 * a_i
            if p is None or p.shape != gt.shape:
                for c in (col_img, col_dif):
                    axes[row][c].text(0.5, 0.5, "outside\nextent", ha="center",
                                      va="center", fontsize=8, color="0.5",
                                      transform=axes[row][c].transAxes)
            else:
                axes[row][col_img].imshow(
                    clip8(p)[..., :3] if p.ndim == 3 else clip8(p),
                    **({} if p.ndim == 3 else dict(cmap="gray", vmin=0, vmax=255)))
                d = diff_map(gt, p)
                axes[row][col_dif].imshow(d, cmap="inferno", vmin=0, vmax=vmax)
                axes[row][col_img].set_xlabel(f"r = {arm.r(k):.2f}", fontsize=7)
                axes[row][col_dif].set_xlabel(f"MAE {d.mean():.2f}", fontsize=7)
            if row == 0:
                axes[row][col_img].set_title(f"{arm.tag} imputation", fontsize=9)
                axes[row][col_dif].set_title(f"|diff| {arm.tag}", fontsize=9)
            tags.append(f"{arm.tag}:"
                        + ("RET" if arm.retained(k)
                           else f"{arm.dist_to_retained(k):.1f}mm"))

        axes[row][0].set_ylabel(f"{k}\n" + "\n".join(tags), fontsize=7.5)
        for c in range(ncol):
            axes[row][c].set_xticks([]); axes[row][c].set_yticks([])

    fig.suptitle(f"{subject_id}   slabs {idxs[0]}-{idxs[-1]}   "
                 f"RET = real photograph, otherwise distance to nearest retained",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(outdir, exist_ok=True)
    p = os.path.join(outdir, f"period_{subject_id}_{idxs[0]:03d}-{idxs[-1]:03d}.png")
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return p


# =============================================================================
# Entry point
# =============================================================================

def run(I_gt, aff_gt, vol8, aff8_iso, aff8, vol12, aff12_iso, aff12,
        subject_id, periods=(0, 12, 24), outdir=OUTDIR):
    ax_gt, t4, arms = build_arms(aff_gt, [
        ("8mm", vol8, aff8_iso, aff8),
        ("12mm", vol12, aff12_iso, aff12),
    ])
    print(f"\nprofile for {subject_id}:")
    out = [profile(I_gt, ax_gt, arms, subject_id, outdir)]
    for s in periods:
        if s < I_gt.shape[ax_gt]:
            out.append(show_period(I_gt, ax_gt, arms, subject_id, s, outdir))
    return out