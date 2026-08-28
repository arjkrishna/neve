#!/usr/bin/env python3
"""The raw TopBrain segmentations, all vessels, two patients per image.

Companion to figure_topbrain_pairs.py, which shows what was BUILT from this
data. This shows what was downloaded: every one of the 42 annotated vessel
classes, in the release's own ITK-SNAP colours, so the R-ICA that was actually
used can be seen in the context of everything that was left on the table.

Colours and names come from the release's own labelmap
(itksnap_labelmap_txt/labelmap_topbrain_mr.txt), not from anything inferred.
That file confirms label 4 = R-ICA, and it is also the cleanest evidence that
the coverage starts at the carotid bifurcation: the most proximal carotid
classes in the whole scheme are R-ICA (4) and R-ECA (35), both of which arise
FROM the common carotid, and there is no CCA class at all.

Each patient is a separate scan in its own frame, so a single shared box would
be meaningless. Instead every panel gets a cube of the SAME SIZE centred on
that patient's own vessels: comparable in scale without pretending the frames
are registered.

Runs on the host (nibabel + skimage + matplotlib); no container needed.

    python monitoring/figure_topbrain_raw_pairs.py [out_dir] [--step 2]
"""
import argparse
import glob
import os
import re
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

RELEASE = "topbrain_data/TopBrain_Data_Release_Batches1n2_081425"
MASKS = os.path.join(RELEASE, "labelsTr_topbrain_mr")
LABELMAP = os.path.join(RELEASE, "itksnap_labelmap_txt", "labelmap_topbrain_mr.txt")

# RAS: +x right, +y anterior, +z superior. A right-lateral view (camera on
# +x) puts the R-ICA siphon in profile, which is the curve that matters here;
# the branch-frame angle used by the other figures means nothing in RAS.
ELEV, AZIM = 10, 4
HERO = 4                    # R-ICA: the vessel the graft actually uses
HERO_ALPHA = 0.85
OTHER_ALPHA = 0.30


def read_labelmap(path):
    """{index: (name, (r,g,b) in 0-1)} from the release's ITK-SNAP file."""
    out = {}
    pat = re.compile(r'\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+\S+\s+\d+\s+\d+\s+"([^"]*)"')
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.lstrip().startswith("#") or not line.strip():
                continue
            m = pat.match(line)
            if m:
                idx = int(m.group(1))
                if idx == 0:
                    continue
                out[idx] = (m.group(5),
                            (int(m.group(2)) / 255.0, int(m.group(3)) / 255.0,
                             int(m.group(4)) / 255.0))
    return out


def surfaces_of(mask_path, labels, step):
    """[(idx, verts_world_mm, faces)] for every label present in the mask."""
    import nibabel as nib
    from skimage import measure

    img = nib.load(mask_path)
    data = np.asarray(img.dataobj)
    aff = img.affine
    spacing = np.sqrt((aff[:3, :3] ** 2).sum(axis=0))

    out = []
    for idx in sorted(labels):
        m = data == idx
        if not m.any():
            continue
        # Crop to this label's own bounding box: marching cubes over the whole
        # 56 M-voxel volume for each of 42 labels would be absurd, and a vessel
        # occupies a tiny fraction of it. Pad by one so the surface closes.
        nz = np.argwhere(m)
        lo = np.maximum(nz.min(0) - 1, 0)
        hi = np.minimum(nz.max(0) + 2, m.shape)
        sub = m[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        if min(sub.shape) < 2:
            continue
        s = step if idx != HERO else max(1, step - 1)   # more detail on R-ICA
        try:
            v, f, _, _ = measure.marching_cubes(sub.astype(np.uint8), level=0.5,
                                                spacing=tuple(spacing),
                                                step_size=s)
        except (ValueError, RuntimeError):
            continue
        # marching cubes returns mm offsets from the CROPPED grid origin; shift
        # back by the crop, convert to voxel index, then to world mm.
        vox = v / spacing + lo
        world = (aff @ np.c_[vox, np.ones(len(vox))].T).T[:, :3]
        out.append((idx, world.astype(np.float32), f))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", default="saved/figs/topbrain_raw")
    ap.add_argument("--elev", type=float, default=ELEV)
    ap.add_argument("--azim", type=float, default=AZIM)
    ap.add_argument("--step", type=int, default=2,
                    help="marching-cubes step_size; lower = finer and slower")
    ap.add_argument("--limit", type=int, default=0, help="only N patients")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    lab = read_labelmap(LABELMAP)
    print("labelmap: %d classes, %d is %s" % (len(lab), HERO, lab[HERO][0]))

    files = sorted(glob.glob(os.path.join(MASKS, "*.nii.gz")))
    if a.limit:
        files = files[:a.limit]
    print("patients: %d  (step_size %d)" % (len(files), a.step), flush=True)

    cases = []
    for f in files:
        t = time.time()
        stem = os.path.basename(f).replace(".nii.gz", "")
        surf = surfaces_of(f, lab.keys(), a.step)
        tris = sum(len(s[2]) for s in surf)
        pts = np.concatenate([s[1] for s in surf], axis=0)
        cases.append({"stem": stem, "surf": surf,
                      "ctr": (pts.min(0) + pts.max(0)) / 2.0,
                      "extent": float(np.max(pts.max(0) - pts.min(0)))})
        print("  %-14s %2d vessels %7d tris  %.1fs"
              % (stem, len(surf), tris, time.time() - t), flush=True)

    RAD = max(c["extent"] for c in cases) / 2.0 * 1.05
    print("common cube half-size %.1f mm (each panel centred on its own "
          "patient)" % RAD, flush=True)

    def draw(ax, case):
        present = []
        for idx, v, f in case["surf"]:
            name, rgb = lab[idx]
            hero = idx == HERO
            ax.add_collection3d(Poly3DCollection(
                v[f], facecolor=(*rgb, HERO_ALPHA if hero else OTHER_ALPHA),
                edgecolor="none", zorder=6 if hero else 3))
            present.append((idx, name, rgb, hero))
        c = case["ctr"]
        ax.set_xlim(c[0] - RAD, c[0] + RAD)
        ax.set_ylim(c[1] - RAD, c[1] + RAD)
        ax.set_zlim(c[2] - RAD, c[2] + RAD)
        ax.view_init(elev=a.elev, azim=a.azim)
        ax.set_xlabel("x (RAS, mm)", fontsize=7, labelpad=-2)
        ax.set_ylabel("y (RAS, mm)", fontsize=7, labelpad=-2)
        ax.set_zlabel("z (RAS, mm)", fontsize=7, labelpad=-2)
        ax.tick_params(labelsize=6, pad=0)
        ax.set_title("%s   %d of %d annotated vessels present"
                     % (case["stem"], len(present), len(lab)), fontsize=9)
        return present

    from matplotlib.lines import Line2D
    n = 0
    for i in range(0, len(cases), 2):
        pair = cases[i:i + 2]
        fig = plt.figure(figsize=(8.5 * len(pair), 9.5))
        present = []
        for j, case in enumerate(pair):
            ax = fig.add_subplot(1, len(pair), j + 1, projection="3d")
            p = draw(ax, case)
            if j == 0:
                present = p
        handles = [Line2D([], [], color=rgb, lw=5,
                          label=("%s  <- grafted" % name) if hero else name)
                   for _, name, rgb, hero in
                   sorted(present, key=lambda r: (not r[3], r[0]))]
        fig.legend(handles=handles, loc="upper left", fontsize=5.5, ncol=3,
                   frameon=True, framealpha=0.9, handlelength=1.4,
                   columnspacing=1.0, labelspacing=0.25,
                   borderpad=0.4, handletextpad=0.5)
        ids = "_".join(c["stem"].replace("topcow_", "") for c in pair)
        fig.tight_layout()
        path = os.path.join(a.out, "topbrain_raw_%02d_%s.png" % (i // 2 + 1, ids))
        fig.savefig(path, dpi=160)
        plt.close(fig)
        n += 1
        print("wrote %s" % os.path.basename(path), flush=True)

    print("done: %d images in %s" % (n, a.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
