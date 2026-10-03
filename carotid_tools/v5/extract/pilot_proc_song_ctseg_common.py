# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\pilot_proc_song_ctseg_common.py (= /sp/v5_va/pilot_proc_song_ctseg_common.py in the build containers)
# SONG COPY (pilot_proc_song, 2026-10-01 21:20): pilot_proc_ia172_ctseg_common.py with the song paths (thin_slice/song), pilot_proc_song_ imports and the Song meta keys; diffs beyond that are stated in the docstring / marked 'song'.
"""pilot_proc_song_ctseg_common.py = pilot_proc_ia172_ctseg_common.py (the 172-series pilot agent's adapter, read-only;
copied 21:20) for the Song 2024 pilot donors (pilot_proc_song). Diff:
  (1) song paths and licence;
  (2) load_ct_crop = the AortaSeg pilot's LAS-capable loader (pilot_ctwall_common.load_ct_crop, 3rd instance: the gzipped
      volume is read ONCE; axes with a negative affine diagonal are flipped to positive spacing; a frame check at <= 400
      seed points requires identical HU) -- Song CTs are LAS, float64;
  (3) PILOT_CT env = the CT the edge is found on: "shipped" (default; thin_slice/song/<case>/ct.nii.gz, clipped at 0 HU) or
      "negzero" (derived/.../ct_negzero-100.nii.gz: -100 HU on every originally negative voxel) or "bg" (ct_bg-100) -- for
      the brief's clipped-vs-re-windowed edge comparison;
  (4) the AortaSeg pilot's PREPEND fix (pilot_ctwall_common.vessels, 3rd instance): when the VMTK line starts > 3 mm from
      the trace origin (Arterial-mask pinch in V1), the trace's own points from the origin up to the VMTK start are
      prepended (prior radius = median MISR of the first 10 mm); recorded in seed_info. PILOT_PREPEND=0 disables it.
The ia172 docstring follows:
loader adapter for the ct_seg4 CT-edge wall segmentation of a 172-series pilot
donor (the role tag_ctwall_common.py plays for TaG-Net patient A; same data structures, so pilot_proc_song_ct_seg.py, a
copy of tag_ct_seg.py = vmr_ct_seg3.py, runs the identical algorithm).

  load_donor(case, with_ct=True) -> {"ct": {"array" [k,j,i] int16 HU, "spacing_mm" (x,y,z) > 0, "origin_mm", "dims_ijk"},
                                     "vessels": {"right_vertebral", "left_vertebral", "basilar", "right_subclavian",
                                                 "left_subclavian"} (only those available) each {"points", "radius_mm"},
                                     "seed_info": per vessel, where the path came from}
  build_stations(d)              -> vmr_common.build_stations' logic (rva / lva origin -> end, ba, rsub / lsub +-30 mm
                                     around each VA origin), for the vessels present.

Seeds (the brief: "seeded from the VMTK trace"):
  VA   = the VMTK MISR centreline of Arterial's cut-out stage-A surface (<der>/arterial/tsba/<v>_stageA_vmtk.json), oriented
         origin -> end; prior radius R = its MISR, median-filtered over 5 mm (the pre-pass in the seg script replaces it by
         the image half-max r_eq, as tag_ctwall's prior was). Only sides that pass the donor gate (<der>/qc/qc.json
         passes_as_v5_donor). If the side's distal course failed QC (distal_ok false), the seed is cut at the dura landmark
         (ts_qc, mapped to the VMTK line by nearest point): the V3-loop gate has shown the course up to the dura is the VA.
  BA   = the shared stretch of the two traces (va_extract confluence.basilar_points) when both VA seeds run to the
         confluence; radius = the Arterial-mask EDT sampled on it (median-filtered); otherwise absent.
  SCA  = Arterial's own RSA / LSA labelled centreline (labelled_centerlines.json, its radius); fallback when the label is
         missing: the TS subclavian mask (1.5 mm copy), voxels within 40 mm of the VA origin, binned every 2 mm along
         their principal axis, bin centroids as the path and the bin's max EDT as the radius.
CT: the native ct.nii.gz (RAS, positive diagonal affine, no flip needed), cropped to the vessels + 20 mm.
Licence: CC BY 4.0 image (172-series, Zenodo 15697196); everything through Arterial is non-commercial.
"""
import json
import os
import sys

import numpy as np
from scipy import ndimage

sys.path.insert(0, "/sp/v5_va")

CASE = os.environ.get("PILOT_CASE", "")
SRC = "/repo/carotid_data/va_sources/thin_slice/song/%s" % CASE
DER = "/repo/carotid_data/va_sources/derived/thin_slice/song/%s" % CASE
SEG_ROOT = os.path.join(DER, "ct_seg")
R_SCALE = float(os.environ.get("PILOT_R_SCALE", "1.0"))
LICENCE = ("Song et al. 2024 CMHA (figshare 26965450), CC BY 4.0 image; derived through Arterial (CC BY-NC weights, "
           "PolyForm-NC code): non-commercial")
CT_KIND = os.environ.get("PILOT_CT", "shipped")
CT_PATH = {"shipped": os.path.join(SRC, "ct.nii.gz"), "negzero": os.path.join(DER, "ct_negzero-100.nii.gz"),
           "bg": os.path.join(DER, "ct_bg-100.nii.gz")}[CT_KIND]


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


def arclen(p):
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]


def _arterial_sa(side):
    lcp = os.path.join(DER, "arterial/runs/native/labelled_centerlines.json")
    if not os.path.isfile(lcp):
        return None
    segs = [s for s in json.load(open(lcp))["segments"] if s["class"] == side + "SA"]
    if not segs:
        return None
    return segs


def _sa_path(side, origin):
    """SCA path near the VA origin: the Arterial RSA / LSA label segment(s) passing closest to the origin."""
    segs = _arterial_sa(side)
    if segs:
        best, bd = None, 1e9
        for s in segs:
            P = np.asarray(s["points_ras_mm"], float)
            d = np.linalg.norm(P - origin, axis=1).min()
            if d < bd:
                best, bd = s, d
        if bd <= 6.0:
            P = np.asarray(best["points_ras_mm"], float)
            R = np.asarray(best["radius_mm"], float)
            # extend with touching segments so +-30 mm is covered
            return P, R, "Arterial %sSA label (nearest segment %.1f mm from the VA origin)" % (side, bd)
    # fallback: TS subclavian mask
    import nibabel as nib
    nm = "subclavian_artery_right" if side == "R" else "subclavian_artery_left"
    im = nib.load(os.path.join(DER, "totalseg_1p5/segmentations/%s.nii.gz" % nm))
    m = np.asanyarray(im.dataobj) > 0
    A = im.affine
    edt = ndimage.distance_transform_edt(m, sampling=np.sqrt((A[:3, :3] ** 2).sum(0)))
    ii = np.argwhere(m)
    W = (A @ np.c_[ii, np.ones(len(ii))].T)[:3].T
    near = np.linalg.norm(W - origin, axis=1) <= 40.0
    W, e = W[near], edt[tuple(ii[near].T)]
    c = W.mean(0)
    u = np.linalg.svd(W - c, full_matrices=False)[2][0]
    t = (W - c) @ u
    rows = []
    for a in np.arange(t.min(), t.max(), 2.0):
        mm = (t >= a) & (t < a + 2.0)
        if mm.sum() >= 3:
            rows.append((W[mm].mean(0), e[mm].max()))
    P = np.array([r[0] for r in rows])
    R = np.array([r[1] for r in rows])
    return P, R, "TS subclavian mask, 2 mm bins along the principal axis within 40 mm of the VA origin (no Arterial label)"


def _ba_path(conf):
    import nibabel as nib
    B = np.asarray(conf["basilar_points"], float)
    im = nib.load(os.path.join(DER, "arterial/runs/native/extracranial_vessels/segmentation.nii.gz"))
    A = im.affine
    sp = np.abs(np.diag(A)[:3])
    I = (np.linalg.inv(A) @ np.c_[B, np.ones(len(B))].T)[:3].T
    lo = np.maximum(np.floor(I.min(0)).astype(int) - 15, 0)
    hi = np.minimum(np.ceil(I.max(0)).astype(int) + 16, np.array(im.shape))
    sub = np.asanyarray(im.dataobj[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]) > 0
    edt = ndimage.distance_transform_edt(sub, sampling=sp)
    r = ndimage.map_coordinates(edt, (I - lo).T, order=1, mode="nearest")
    r = ndimage.median_filter(np.clip(r, 0.7, 4.0), size=11, mode="nearest")
    return B, r


def selected(side):
    """song diff (5) (23:30): the trace set chosen per side by pilot_proc_song_qc2.py (qc/select.json: tsba unless the
    unchanged tool's trace ts grades better); returns (ext, qc side record, va_extract dict)."""
    sp = os.path.join(DER, "qc/select.json")
    ext = "tsba"
    if os.path.isfile(sp):
        ext = json.load(open(sp))["sides"][side]["ext"]
    qf = "qc.json" if ext == "tsba" else "qc_%s.json" % ext
    qc_ = json.load(open(os.path.join(DER, "qc", qf)))
    ex_ = json.load(open(os.path.join(DER, "arterial/%s/va_extract.json" % ext)))
    return ext, qc_["sides"][side], ex_


def vessels():
    out, info = {}, {}
    full = {}
    exts = {}
    for side, v, nm in (("R", "rva", "right"), ("L", "lva", "left")):
        ext, q, ex = selected(side)
        exts[side] = (ext, ex)
        if not q.get("passes_as_v5_donor") and os.environ.get("PILOT_SEG_ALL", "0") != "1":
            info[nm + "_vertebral"] = "not segmented: the side fails the donor gate (%s)" % q.get("gate_fail_reasons")
            continue
        vj = json.load(open(os.path.join(DER, "arterial/%s/%s_stageA_vmtk.json" % (ext, v))))
        P, R = np.asarray(vj["points"], float), np.asarray(vj["radii"], float)
        T0 = np.asarray(ex["sides"][side]["trace"]["points"][0], float)
        if np.linalg.norm(P[-1] - T0) < np.linalg.norm(P[0] - T0):
            P, R = P[::-1], R[::-1]
        s = arclen(P)
        cut = None
        if not q.get("distal_ok"):
            du = (q.get("vmtk") or {}).get("landmarks_ts_qc_mm", {}).get("dura")
            if du is None:
                du = q["landmarks_used_mm"].get("dura")
            if du is not None:
                cut = float(du)
                k = int(np.searchsorted(s, cut)) + 1
                P, R = P[:k], R[:k]
        full[side] = cut is None
        # song diff (4): the AortaSeg pilot's PREPEND fix (VMTK line starting > 3 mm from the trace origin)
        prep_info = None
        Tall = np.asarray(ex["sides"][side]["trace"]["points"], float)
        if os.environ.get("PILOT_PREPEND", "1") != "0" and np.linalg.norm(P[0] - T0) > 3.0:
            jq = int(np.argmin(np.linalg.norm(Tall - P[0], axis=1)))
            sP = arclen(P)
            r0 = float(np.median(R[sP <= 10.0]))
            pre = Tall[:jq]
            if len(pre) >= 2:
                prep_info = {"prepended_trace_points": int(len(pre)), "prepended_length_mm": float(arclen(Tall[:jq + 1])[-1]),
                             "vmtk_start_to_trace_origin_mm": float(np.linalg.norm(T0 - P[0])), "prior_radius_mm": r0}
                P = np.vstack([pre, P])
                R = np.r_[np.full(len(pre), r0), R]
                s = arclen(P)
        step = float(np.median(np.diff(s))) if len(s) > 2 else 0.5
        R = ndimage.median_filter(R, size=max(3, int(round(5.0 / max(step, 1e-3))) | 1), mode="nearest")
        out[nm + "_vertebral"] = {"points": P, "radius_mm": R_SCALE * R}
        info[nm + "_vertebral"] = {"from": os.path.join(DER, "arterial/%s/%s_stageA_vmtk.json" % (ext, v)), "trace_set": ext,
                                   "cut_at_dura_mm": cut, "length_mm": round(float(arclen(P)[-1]), 1), "prepend": prep_info}
        Ps, Rs, how = _sa_path(side, P[0])
        out[nm + "_subclavian"] = {"points": Ps, "radius_mm": R_SCALE * Rs}
        info[nm + "_subclavian"] = how
    same = "R" in exts and "L" in exts and exts["R"][0] == exts["L"][0]   # song diff (5): BA only from one trace set
    conf = (exts["R"][1].get("confluence") or {}) if same else {}
    if same and full.get("R") and full.get("L") and conf.get("method") == "two traces join" and conf.get("basilar_points"):
        B, rb = _ba_path(conf)
        if len(B) >= 8:
            out["basilar"] = {"points": B, "radius_mm": R_SCALE * rb}
            info["basilar"] = "shared stretch of the two traces (confluence.basilar_points), r = Arterial-mask EDT"
    return out, info


def load_ct_crop(lo_ras, hi_ras, margin_mm=20.0, check_pts=None):
    """song diff (2)+(3): the AortaSeg pilot's pilot_ctwall_common.load_ct_crop (copied verbatim apart from the source
    path = CT_PATH): CT crop around [lo, hi] (RAS mm) + margin as the VMR loader's ct dict: array [k,j,i] int16, positive
    spacing, world = origin + (i,j,k) * spacing. Axes with a negative affine diagonal are flipped."""
    import nibabel as nib
    im = nib.load(CT_PATH)
    A = im.affine
    assert np.abs(A[:3, :3] - np.diag(np.diag(A[:3, :3]))).max() < 1e-6, A
    sp = np.abs(np.diag(A)[:3])
    sh = np.array(im.shape[:3])
    lo = np.asarray(lo_ras, float) - margin_mm
    hi = np.asarray(hi_ras, float) + margin_mm
    Ai = np.linalg.inv(A)
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    ijk = corners @ Ai[:3, :3].T + Ai[:3, 3]
    a = np.maximum(np.floor(ijk.min(0)).astype(int), 0)
    b = np.minimum(np.ceil(ijk.max(0)).astype(int), sh - 1)
    full = np.asanyarray(im.dataobj)   # read the gzipped volume ONCE (AortaSeg pilot, 3rd instance)
    blk = np.asarray(full[a[0]:b[0] + 1, a[1]:b[1] + 1, a[2]:b[2] + 1]).astype(np.int16)
    flip = np.diag(A)[:3] < 0
    org = np.zeros(3)
    for ax in range(3):
        if flip[ax]:
            blk = np.flip(blk, axis=ax)
            org[ax] = A[ax, 3] + A[ax, ax] * b[ax]
        else:
            org[ax] = A[ax, 3] + A[ax, ax] * a[ax]
    arr = np.ascontiguousarray(blk.transpose(2, 1, 0))   # [k, j, i]
    ct = {"array": arr, "spacing_mm": sp, "origin_mm": org, "dims_ijk": np.array(arr.shape[::-1]),
          "nifti_block_ijk": [a.tolist(), b.tolist()], "flipped_axes": flip.tolist(), "source": CT_PATH, "ct_kind": CT_KIND}
    if check_pts is not None and len(check_pts):
        W = np.asarray(check_pts, float)
        W = W[np.linspace(0, len(W) - 1, min(400, len(W))).astype(int)]
        vn = np.rint(W @ Ai[:3, :3].T + Ai[:3, 3]).astype(int)
        ok = np.all((vn >= a) & (vn <= b), axis=1)
        vn, W = vn[ok], W[ok]
        hu_n = np.array([int(full[i, j, k]) for i, j, k in vn])
        Wv = vn @ A[:3, :3].T + A[:3, 3]
        q = np.round((Wv - org) / sp).astype(int)
        hu_c = arr[q[:, 2], q[:, 1], q[:, 0]].astype(int)
        ct["frame_check"] = {"n": int(len(vn)), "max_abs_diff_HU": int(np.abs(hu_c - hu_n).max()),
                             "max_index_residual": float(np.abs((Wv - org) / sp - q).max())}
        assert ct["frame_check"]["max_abs_diff_HU"] == 0, ct["frame_check"]
    return ct


def load_donor(with_ct=True):
    V, info = vessels()
    out = {"vessels": V, "seed_info": info, "licence": LICENCE}
    if with_ct and V:
        allp = []
        for k, v in V.items():
            P = v["points"]
            if k.endswith("subclavian"):
                o = V[k.replace("subclavian", "vertebral")]["points"][0]
                s = arclen(P)
                j = int(np.argmin(np.linalg.norm(P - o, axis=1)))
                P = P[np.abs(s - s[j]) <= 32.0]
            allp.append(P)
        allp = np.vstack(allp)
        chk = np.vstack([V[k]["points"] for k in V if k.endswith("vertebral")])
        out["ct"] = load_ct_crop(allp.min(0), allp.max(0), check_pts=chk)   # song: frame check (LAS flip)
    return out


def build_stations(d):
    """vmr_common.build_stations' logic for the vessels present."""
    from vmr_common import _stations
    V = d["vessels"]
    ves = {}
    for nm, key in (("right_vertebral", "rva"), ("left_vertebral", "lva"), ("basilar", "ba")):
        if nm in V:
            ves[key] = _stations(V[nm]["points"], V[nm]["radius_mm"])
    for side, key in (("right", "rsub"), ("left", "lsub")):
        if side + "_subclavian" not in V or side[0] + "va" not in ves:
            continue
        sub = V[side + "_subclavian"]
        sp, sr = np.asarray(sub["points"]), np.asarray(sub["radius_mm"])
        o = ves[side[0] + "va"]["P"][0]
        k = int(np.argmin(np.linalg.norm(sp - o, axis=1)))
        ss = arclen(sp)
        m = np.abs(ss - ss[k]) <= 30.0
        ves[key] = _stations(sp[m], sr[m])
    return ves
