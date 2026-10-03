# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\tag_ctwall_common.py (= /sp/v5_va/tag_ctwall_common.py in the build containers)
"""tag_ctwall_common.py -- loader adapter + shared constants for the tag_ctwall agent (the real CT wall of TaG-Net
patient A's vertebral arteries by the study's ct_seg4 method).

The study tool vmr_ct_seg3.py (ct_seg4 = VMR_OPEN_BRIGHT=1) reads VMR 0248 through vmr_common.load_vmr() /
build_stations(). This module supplies the SAME data structures for TaG-Net patient A, so tag_ct_seg.py (a copy of
vmr_ct_seg3.py) runs the identical algorithm:

  load_tag(with_ct=True) -> {"ct": {"array" [k,j,i] int16 HU, "spacing_mm" (x,y,z) > 0, "origin_mm", "dims_ijk"},
                             "vessels": {"right_vertebral", "left_vertebral", "basilar", "right_subclavian",
                                         "left_subclavian"} each {"points" (n,3) RAS mm, "radius_mm" (n,)}}
  build_stations(vmr)    -> vmr_common.build_stations, verbatim logic (rva / lva origin -> merge, ba, rsub / lsub
                            +-30 mm around each VA origin), on these vessels.

Frame: RAS mm = the NIfTI qform world of the TaG-Net CTA (nibabel affine diag(-0.405, -0.405, 0.4)). The CT class
of vmr_common assumes world = origin + ijk * spacing with positive spacing, so the crop is FLIPPED along i and j
(LPS voxel order -> RAS voxel order) and the origin set accordingly; a check samples the flipped crop and the
original NIfTI at the GT voxel centres and requires identical HU.

Inputs (all from the tag_labels agent, read-only):
  VA   = gt/A_{side}.json  vertebral.points (ostium -> merge, the VMR loader's rules) / radius_mm
  SCA  = gt/A_{side}.json  subclavian.points / radius_mm (whole SCA path; build_stations keeps +-30 mm)
  BA   = gt/A_right.json   basilar.points / radius_mm (the right confluence is the lower one, z -505.8)
  radius_mm = the image half-maximum equal-area radius (port of the ct_seg3 rays, pass 2). VMR's prior R was the
  expert contour r_eq, which ran 1.15-1.19 x the CT ray r_eq (VMR seg_qc: 1.64 / 1.38 R, 1.86 / 1.62 L). The
  scale factor TAG_R_SCALE (default 1.0) multiplies every prior radius (sensitivity run: 1.17).

LICENCE: TaG-Net data has no stated licence -- internal benchmarking only; everything derived stays under the
gitignored carotid_data/va_sources/derived/tagnet/.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/sp/v5_va")

TAG = "/repo/carotid_data/va_sources/tagnet/SampleData"
CTA = os.path.join(TAG, "002/CTA.nii.gz")
GTNII = os.path.join(TAG, "001/labeled_cl_gt.nii.gz")
DER = "/repo/carotid_data/va_sources/derived/tagnet"
GT_DIR = os.path.join(DER, "gt")
SEG_ROOT = os.path.join(DER, "ct_seg")
QC_DIR = os.path.join(SEG_ROOT, "qc")
RES = "/sp/v5_va/results"
TSV3_1P5 = os.path.join(DER, "totalseg_v3_1p5")
TSV3 = os.path.join(DER, "totalseg_v3")
TS_1P5 = os.path.join(DER, "totalseg_1p5")
VOXEL_ACQ = (0.405, 0.405, 0.4)      # acquisition voxel of the TaG-Net CTA (header zooms 0.40527344 / 0.4)
VMR_VOXEL = (0.4004, 0.4004, 0.7)
LICENCE = ("TaG-Net sample data (Yao et al., IEEE TMI 2023;42(11):3155-3166, doi 10.1109/TMI.2023.3240825; "
           "https://github.com/PRESENT-Y/TaG-Net). NO data licence stated: internal benchmarking and calibration "
           "only; nothing derived may enter a tracked file, a shipped anatomy or the repo outside the gitignored "
           "carotid_data/va_sources/.")
R_SCALE = float(os.environ.get("TAG_R_SCALE", "1.0"))


def jdump(obj, path):
    def conv(o):
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.bool_):
            return bool(o)
        raise TypeError(type(o))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=conv)


def load_gt(side):
    return json.load(open(os.path.join(GT_DIR, "A_%s.json" % side)))


def vessels(r_scale=None):
    k = R_SCALE if r_scale is None else r_scale
    out = {}
    for side in ("right", "left"):
        g = load_gt(side)
        out[side + "_vertebral"] = {"points": np.asarray(g["vertebral"]["points"], float),
                                    "radius_mm": k * np.asarray(g["vertebral"]["radius_mm"], float)}
        out[side + "_subclavian"] = {"points": np.asarray(g["subclavian"]["points"], float),
                                     "radius_mm": k * np.asarray(g["subclavian"]["radius_mm"], float)}
    g = load_gt("right")
    out["basilar"] = {"points": np.asarray(g["basilar"]["points"], float),
                      "radius_mm": k * np.asarray(g["basilar"]["radius_mm"], float)}
    return out


def load_ct_crop(lo_ras, hi_ras, margin_mm=20.0, check=True):
    """CTA crop around [lo, hi] (RAS mm) + margin, as the VMR loader's ct dict: array [k,j,i] int16, positive
    spacing, world = origin + (i,j,k) * spacing. i and j are flipped (the NIfTI voxel axes run L / P)."""
    import nibabel as nib
    im = nib.load(CTA)
    A = im.affine
    assert abs(A[0, 1]) + abs(A[0, 2]) + abs(A[1, 0]) + abs(A[1, 2]) + abs(A[2, 0]) + abs(A[2, 1]) < 1e-6, A
    assert A[0, 0] < 0 and A[1, 1] < 0 and A[2, 2] > 0, A
    sp = np.abs(np.diag(A)[:3])
    sh = np.array(im.shape[:3])
    lo = np.asarray(lo_ras, float) - margin_mm
    hi = np.asarray(hi_ras, float) + margin_mm
    Ai = np.linalg.inv(A)
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    ijk = corners @ Ai[:3, :3].T + Ai[:3, 3]
    a = np.maximum(np.floor(ijk.min(0)).astype(int), 0)
    b = np.minimum(np.ceil(ijk.max(0)).astype(int), sh - 1)
    blk = np.asarray(im.dataobj[a[0]:b[0] + 1, a[1]:b[1] + 1, a[2]:b[2] + 1]).astype(np.int16)
    blk = blk[::-1, ::-1, :]                       # i, j flipped: x and y now increase with the index
    arr = np.ascontiguousarray(blk.transpose(2, 1, 0))   # [k, j, i]
    org = np.array([A[0, 3] - sp[0] * b[0], A[1, 3] - sp[1] * b[1], A[2, 3] + sp[2] * a[2]])
    ct = {"array": arr, "spacing_mm": sp, "origin_mm": org, "dims_ijk": np.array(arr.shape[::-1]),
          "nifti_block_ijk": [a.tolist(), b.tolist()], "source": CTA}
    if check:
        # HU at a few hundred GT voxel centres: NIfTI index vs the flipped crop's world sampler
        g = np.asanyarray(nib.load(GTNII).dataobj)
        vox = np.argwhere(g[a[0]:b[0] + 1, a[1]:b[1] + 1, a[2]:b[2] + 1] > 0) + a
        vox = vox[np.linspace(0, len(vox) - 1, min(400, len(vox))).astype(int)]
        hu_n = np.asarray(im.dataobj[vox[:, 0].min():vox[:, 0].max() + 1, vox[:, 1].min():vox[:, 1].max() + 1,
                                     vox[:, 2].min():vox[:, 2].max() + 1])
        hu_n = hu_n[vox[:, 0] - vox[:, 0].min(), vox[:, 1] - vox[:, 1].min(), vox[:, 2] - vox[:, 2].min()]
        W = vox @ A[:3, :3].T + A[:3, 3]
        q = np.round((W - org) / sp).astype(int)
        hu_c = arr[q[:, 2], q[:, 1], q[:, 0]]
        ct["frame_check"] = {"n": int(len(vox)), "max_abs_diff_HU": int(np.abs(hu_c.astype(int) - hu_n.astype(int)).max()),
                             "max_index_residual": float(np.abs((W - org) / sp - q).max())}
        assert ct["frame_check"]["max_abs_diff_HU"] == 0, ct["frame_check"]
    return ct


def load_tag(with_ct=True, r_scale=None):
    """The VMR loader's shape for TaG-Net patient A."""
    V = vessels(r_scale)
    out = {"vessels": V, "licence": LICENCE}
    if with_ct:
        allp = []
        for side in ("right", "left"):
            allp.append(V[side + "_vertebral"]["points"])
            # build_stations keeps the SCA +-30 mm around the VA origin
            sp_ = V[side + "_subclavian"]["points"]
            o = V[side + "_vertebral"]["points"][0]
            k = int(np.argmin(np.linalg.norm(sp_ - o, axis=1)))
            ss = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(sp_, axis=0), axis=1))]
            allp.append(sp_[np.abs(ss - ss[k]) <= 32.0])
        allp.append(V["basilar"]["points"])
        allp = np.vstack(allp)
        out["ct"] = load_ct_crop(allp.min(0), allp.max(0))
    return out


def build_stations(vmr):
    """vmr_common.build_stations, verbatim (same _stations, same +-30 mm subclavian window)."""
    from vmr_common import _stations, arclen
    V = vmr["vessels"]
    ves = {"rva": _stations(V["right_vertebral"]["points"], V["right_vertebral"]["radius_mm"]),
           "lva": _stations(V["left_vertebral"]["points"], V["left_vertebral"]["radius_mm"]),
           "ba": _stations(V["basilar"]["points"], V["basilar"]["radius_mm"])}
    for side, key in (("right", "rsub"), ("left", "lsub")):
        sub = V[side + "_subclavian"]
        sp, sr = np.asarray(sub["points"]), np.asarray(sub["radius_mm"])
        o = ves[side[0] + "va"]["P"][0]
        k = int(np.argmin(np.linalg.norm(sp - o, axis=1)))
        ss = arclen(sp)
        m = np.abs(ss - ss[k]) <= 30.0
        ves[key] = _stations(sp[m], sr[m])
    return ves


from scipy.spatial import cKDTree as _cKDTree  # noqa: E402


class KTree64(_cKDTree):
    """Exact ownership for va_metrics without editing it: va_metrics.cKDTree is replaced by this subclass, whose
    query(k=24) becomes query(k=64) (k >= 64 is where the verifier found the results stop changing). Every other
    call (k=1, sparse_distance_matrix) is unchanged. va_metrics uses k=24 in exactly two places: the owner of a
    union cell in measure_union_branch (W1-W3, W6, W7) and the owner of a native wall cell in wall_native (W4
    fitted, W5)."""
    K_FROM, K_TO = 24, 64

    def query(self, x, k=1, **kw):
        if k == self.K_FROM:
            k = min(self.K_TO, self.n)
        return super().query(x, k=k, **kw)


def exact_ownership(vm, on=True):
    vm.cKDTree = KTree64 if on else _cKDTree
