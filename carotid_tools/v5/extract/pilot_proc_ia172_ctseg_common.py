# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\pilot_proc_ia172_ctseg_common.py (= /sp/v5_va/pilot_proc_ia172_ctseg_common.py in the build containers)
"""pilot_proc_ia172_ctseg_common.py -- loader adapter for the ct_seg4 CT-edge wall segmentation of a 172-series pilot
donor (the role tag_ctwall_common.py plays for TaG-Net patient A; same data structures, so pilot_proc_ia172_ct_seg.py, a
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
SRC = "/repo/carotid_data/va_sources/thin_slice/ia172/%s" % CASE
DER = "/repo/carotid_data/va_sources/derived/thin_slice/ia172/%s" % CASE
SEG_ROOT = os.path.join(DER, "ct_seg")
R_SCALE = float(os.environ.get("PILOT_R_SCALE", "1.0"))
LICENCE = ("Head-CTA 172-series (Zenodo 15697196), CC BY 4.0 image; derived through Arterial (CC BY-NC weights, "
           "PolyForm-NC code): non-commercial")


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


def vessels():
    qc = json.load(open(os.path.join(DER, "qc/qc.json")))
    ex = json.load(open(os.path.join(DER, "arterial/tsba/va_extract.json")))
    out, info = {}, {}
    full = {}
    for side, v, nm in (("R", "rva", "right"), ("L", "lva", "left")):
        q = qc["sides"][side]
        if not q.get("passes_as_v5_donor") and os.environ.get("PILOT_SEG_ALL", "0") != "1":
            info[nm + "_vertebral"] = "not segmented: the side fails the donor gate (%s)" % q.get("gate_fail_reasons")
            continue
        vj = json.load(open(os.path.join(DER, "arterial/tsba/%s_stageA_vmtk.json" % v)))
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
        step = float(np.median(np.diff(s))) if len(s) > 2 else 0.5
        R = ndimage.median_filter(R, size=max(3, int(round(5.0 / max(step, 1e-3))) | 1), mode="nearest")
        out[nm + "_vertebral"] = {"points": P, "radius_mm": R_SCALE * R}
        info[nm + "_vertebral"] = {"from": os.path.join(DER, "arterial/tsba/%s_stageA_vmtk.json" % v),
                                   "cut_at_dura_mm": cut, "length_mm": round(float(arclen(P)[-1]), 1)}
        Ps, Rs, how = _sa_path(side, P[0])
        out[nm + "_subclavian"] = {"points": Ps, "radius_mm": R_SCALE * Rs}
        info[nm + "_subclavian"] = how
    conf = ex.get("confluence") or {}
    if full.get("R") and full.get("L") and conf.get("method") == "two traces join" and conf.get("basilar_points"):
        B, rb = _ba_path(conf)
        if len(B) >= 8:
            out["basilar"] = {"points": B, "radius_mm": R_SCALE * rb}
            info["basilar"] = "shared stretch of the two traces (confluence.basilar_points), r = Arterial-mask EDT"
    return out, info


def load_ct_crop(lo_ras, hi_ras, margin_mm=20.0):
    import nibabel as nib
    im = nib.load(os.path.join(SRC, "ct.nii.gz"))
    A = im.affine
    assert np.all(np.diag(A)[:3] > 0) and np.abs(A[:3, :3] - np.diag(np.diag(A)[:3])).max() < 1e-6, A
    sp = np.diag(A)[:3].copy()
    sh = np.array(im.shape[:3])
    lo = np.asarray(lo_ras, float) - margin_mm
    hi = np.asarray(hi_ras, float) + margin_mm
    a = np.maximum(np.floor((lo - A[:3, 3]) / sp).astype(int), 0)
    b = np.minimum(np.ceil((hi - A[:3, 3]) / sp).astype(int), sh - 1)
    blk = np.asarray(im.dataobj[a[0]:b[0] + 1, a[1]:b[1] + 1, a[2]:b[2] + 1]).astype(np.int16)
    arr = np.ascontiguousarray(blk.transpose(2, 1, 0))      # [k, j, i]
    org = A[:3, 3] + a * sp
    return {"array": arr, "spacing_mm": sp, "origin_mm": org, "dims_ijk": np.array(arr.shape[::-1]),
            "nifti_block_ijk": [a.tolist(), b.tolist()], "source": os.path.join(SRC, "ct.nii.gz"),
            "frame_check": "RAS NIfTI with a positive diagonal affine: world = origin + ijk * spacing, no flip"}


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
        out["ct"] = load_ct_crop(allp.min(0), allp.max(0))
        # third instance (21:50), env-gated noise-floor test: PILOT_PHANTOM_NPZ = a synthetic CT on EXACTLY this crop grid
        # (pilot_proc_ia172_phantom.py: featureless tubes along the real centrelines, measured PSF and noise); unset = the
        # real CT (the default; nothing changes for the pipeline).
        ph = os.environ.get("PILOT_PHANTOM_NPZ")
        if ph:
            z = np.load(ph)
            assert z["array"].shape == out["ct"]["array"].shape, (z["array"].shape, out["ct"]["array"].shape)
            assert np.allclose(z["origin_mm"], out["ct"]["origin_mm"]) and np.allclose(z["spacing_mm"], out["ct"]["spacing_mm"])
            out["ct"]["array"] = z["array"].astype(np.int16)
            out["ct"]["source"] = ph
            out["phantom"] = {k: str(z[k]) for k in z.files if k not in ("array", "origin_mm", "spacing_mm")}
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
