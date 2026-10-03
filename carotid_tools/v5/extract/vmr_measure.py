#!/usr/bin/env python3
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\vmr_measure.py (= /sp/v5_va/vmr_measure.py in the build containers)
"""VMR 0248 VAs through the shared suite (va_metrics.measure_va, UNCHANGED) and the grader
(va_standard), three representations x two sides, all with VMTK MISR centerlines:

  lofted_vmtk : SimVascular lofted wall face (open, as ref e) + VMTK on the capped wall
  ct_stageA   : CT segmentation (vmr_ct_seg2.py), stage-A surface (binary MC + windowed sinc 20/0.1)
  ct_iso      : CT segmentation, sub-voxel isosurface HU = T_loc, unsmoothed

Landmarks (V1..V4) from the CT (vmr_landmarks.py enclosure profile on the ct_stageA centerline) and
the foramen-magnum plane (basion / opisthion picked on the midsagittal CT, results/vmr/fm_zoom.png);
mapped to the other centerlines by nearest point. Per-segment wall detail through a thin adapter
(union_segments: the union_detail construction, verbatim calls, then vm.measure_union_branch on
landmark ranges; checked against measure_va's own wall_union core) and vm.section_detail on the
segment's centerline. eve-training-fixed. Writes results/vmr0248_metrics.json.
"""
import json
import os
import sys
import time
from multiprocessing import Pool
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, "/sp/v5_va")
import va_metrics as vm  # noqa: E402
import va_standard as vs  # noqa: E402
from vmr_common import DERIVED, RES, WORK, jdump  # noqa: E402

VOXEL = (0.4004, 0.4004, 0.7)
TAG = os.environ.get("VMR_SEG_TAG", "ct_seg2")
OUT = os.path.join(WORK, "vmr0248_metrics_%s.json" % TAG)
# foramen magnum (midsagittal x = -107.9): picked on results/vmr/fm_zoom.png
BASION = np.array([105.5, -140.5])     # (y, z) mm, RAS
OPISTHION = np.array([76.5, -145.0])
C1_C2_Z = -163.0                       # C1 lateral mass on C2 (midsagittal / coronal views)


def z_fm(y):
    return OPISTHION[1] + (BASION[1] - OPISTHION[1]) / (BASION[0] - OPISTHION[0]) * (y - OPISTHION[0])


def runs(mask, s, min_len=1.5):
    out, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            if s[j] - s[i] >= min_len:
                out.append((i, j))
            i = j + 1
        else:
            i += 1
    return out


def find_landmarks(enc):
    s, e = np.asarray(enc["s"]), np.asarray(enc["enclosure_smooth"])
    z, y = np.asarray(enc["z"]), np.asarray(enc["y"])
    rr = runs(e >= 0.5, s)
    first = [r for r in rr if s[r[0]] > 15.0]
    c6 = float(s[first[0][0]])
    below = [r for r in rr if z[r[1]] < C1_C2_Z and s[r[0]] > c6]
    c2 = float(s[below[-1][1]])
    above = [r for r in rr if s[r[0]] > c2 and z[r[0]] >= C1_C2_Z - 5 and z[r[1]] < -140]
    c1 = float(s[above[0][1]]) if above else None
    after = np.nonzero((s > (c1 or c2)) & (z >= z_fm(y)))[0]
    dura = float(s[after[0]]) if len(after) else None
    # V3 horizontal loop: most posterior point between C1 and the FM crossing
    m = (s > (c1 or c2)) & (s < (dura or s[-1]))
    i_post = int(np.nonzero(m)[0][np.argmin(y[m])]) if m.any() else None
    return {"C6": c6, "C2": c2, "C1": c1, "dura": dura,
            "runs_enclosure_ge_0.5": [[round(float(s[a]), 2), round(float(s[b]), 2), round(float(z[a]), 1),
                                       round(float(z[b]), 1)] for a, b in rr],
            "v3_most_posterior_s": None if i_post is None else float(s[i_post]),
            "definitions": {
                "C6": "start of the first bony-enclosure run (>= 0.5 of 48 normal-plane rays meet HU >= 400 within "
                      "r + 0.3..7 mm, smoothed 2.25 mm) after s = 15 mm: entry into the C6 transverse foramen "
                      "(z level checked against the C6 body on the midsagittal CT)",
                "C2": "end of the last enclosure run below z = %.0f (C1/C2 articulation): exit from the C2 "
                      "transverse foramen" % C1_C2_Z,
                "C1": "end of the first enclosure run at the C1 level after C2: exit from the C1 transverse foramen",
                "dura": "first point after C1 above the foramen-magnum plane (basion (y,z) = %s, opisthion %s, "
                        "x-invariant): radiological proxy for the dural entry (V3/V4)" % (BASION.tolist(),
                                                                                         OPISTHION.tolist())}}


def union_segments(P, R, surface, ranges, spacing=None, margin=0.5, taper_start=True, taper_end=False):
    """THIN ADAPTER: the exact construction of va_metrics.union_detail (same calls, same arguments),
    exposing the union so measure_union_branch can be run on landmark ranges."""
    if vm.TOPBRAIN_TOOLS not in sys.path:
        sys.path.insert(0, vm.TOPBRAIN_TOOLS)
    from sdf_mesher import SPACING_MM, iso_surface, tube_field
    from sdf_union_v4 import Samples, Section, add_real_sections_owned
    spacing = spacing or SPACING_MM
    P, R = vm.resample(P, R)[:2]
    va = SimpleNamespace(name="VA", coordinates=P, radii=R)
    pad = vm.CAPSULE_FACTOR * float(R.max()) + vm.CAPSULE_PLUS_MM + 3.0
    lo, hi = P.min(0) - pad, P.max(0) + pad
    dummy = SimpleNamespace(name="pad", coordinates=np.array([lo, hi]), radii=np.zeros(2))
    samp = Samples([va])
    field = tube_field([va, dummy], spacing=spacing)
    surf = vm.prepare_surface(surface)
    sec = Section("va", surf, P, R, taper_start, taper_end, "VA", 0.0)
    add_real_sections_owned(field, [sec], samp, spacing, margin)
    iso = iso_surface(field)
    conn = iso.connectivity()
    rid = np.asarray(conn.cell_data["RegionId"])
    big = int(np.argmax(np.bincount(rid)))
    iso = conn.extract_cells(np.nonzero(rid == big)[0]).extract_surface().triangulate()
    bi = samp.index_of("VA")
    L = float(samp.s[samp.b == 0].max())
    out = {"whole": vm.measure_union_branch(iso, samp, bi, 0.0, L)}
    for nm, (a, b) in ranges.items():
        if a is None or b is None:
            continue
        out[nm] = vm.measure_union_branch(iso, samp, bi, a, b)
        out[nm]["from_mm"], out[nm]["to_mm"] = round(a, 1), round(b, 1)
    return out


def job(args):
    variant, side, cl_path, surf_path, wall_from_loader, landmarks, parent_dir = args
    import pyvista as pv
    t0 = time.time()
    d = json.load(open(cl_path))
    P, R = np.asarray(d["points"], float), np.asarray(d["radii"], float)
    if wall_from_loader:
        from sources import load_vmr0248
        surf = load_vmr0248(with_ct=False, with_model=False, with_walls=True)["vessels"][side + "_vertebral"]["wall"]
    else:
        surf = pv.read(surf_path)
    st = {"lofted_vmtk": "lofted (SimVascular wall face between expert contours)",
          "ct_stageA": "CT segmentation (local half-max), stage-A surface: binary MC 0.5 + windowed sinc 20/0.1",
          "ct_iso": "CT segmentation (local half-max), sub-voxel isosurface, unsmoothed"}[variant]
    res = vm.measure_va(P, R, surf, voxel_mm=VOXEL, source_type=st,
                        radius_kind="VMTK MISR (vmtkCenterlines pointlist, AppendEndPoints=1; neve-build-vmtk)",
                        landmarks=landmarks, starts_at_origin=True, parent_dir=parent_dir, up=np.array([0, 0, 1.0]),
                        name="VMR 0248 %s VA (%s)" % (side, variant),
                        extra={"centerline": cl_path, "surface": surf_path, "variant": variant, "side": side,
                               "vmtk_note": d.get("note")})
    Pr, Rr, sr = vm.resample(P, R)
    L = float(sr[-1])
    seg = {"V1": (0.0, landmarks["C6"]), "V2": (landmarks["C6"], landmarks["C2"]),
           "V3": (landmarks["C2"], landmarks["dura"]), "V4": (landmarks["dura"], L),
           "V3_C2_to_C1": (landmarks["C2"], landmarks["C1"]), "V3_C1_to_dura": (landmarks["C1"], landmarks["dura"]),
           "distal_0_40": (L - 40.0, L), "distal_40_80": (L - 80.0, L - 40.0)}
    us = union_segments(P, R, surf, seg)
    secs = {}
    for nm, (a, b) in seg.items():
        m = (sr >= a) & (sr <= b)
        if m.sum() < 40:
            continue
        secs[nm] = vm.section_detail(Pr[m], Rr[m], sr[m] - sr[m][0], surf)
        secs[nm]["r_misr_median"] = round(float(np.median(Rr[m])), 3)
    res["segments_wall"] = {"union": us, "sections": secs, "ranges_mm": {k: [round(a, 2), round(b, 2)] for k, (a, b) in seg.items()},
                            "adapter_check": {"measure_va_core_0.3": res["wall_union"]["core"]["frac_0.3mm"],
                                              "adapter_core_0.3": us["whole"]["core"]["frac_0.3mm"]}}
    res["radius_profile"] = {"s": np.round(sr[::4], 2).tolist(), "r": np.round(Rr[::4], 3).tolist(),
                             "P": np.round(Pr[::4], 2).tolist()}
    res["seconds"] = round(time.time() - t0, 1)
    wu = res["wall_union"]["core"]
    print("%-11s %-5s L=%.1f r=%.2f DM=%.2f union %.3f/%.3f rel %.3f sec_inout %.3f fitted %.3f genus %s ecc %.3f (%.0fs)" % (
        variant, side, res["gross"]["length_mm"], res["calibre"]["r_median_mm"], res["gross"]["DM"], wu["frac_0.3mm"],
        wu["frac_0.5mm"], wu["frac_rel_0.15"], res["wall_sections"].get("in_out_0.3_mean", -1),
        res["wall_native"].get("fitted_frac_abs_0.3", -1), res["wall_union"]["genus"],
        res["wall_sections"].get("centre_offset_mm_median", -1), res["seconds"]), flush=True)
    return variant, side, res


def main():
    t0 = time.time()
    from sources import load_vmr0248
    vmr = load_vmr0248(with_ct=False, with_model=False, with_walls=False)
    enc = json.load(open(os.path.join(WORK, "enclosure_%s.json" % TAG)))
    side_of = {"rva": "right", "lva": "left"}
    lm, lm_pts, pdirs = {}, {}, {}
    for v, side in side_of.items():
        lm[side] = find_landmarks(enc[v])
        cl = json.load(open(os.path.join(DERIVED, TAG, "%s_stageA_vmtk.json" % v)))
        P, _, s = vm.resample(np.asarray(cl["points"]), None)
        lm_pts[side] = {k: P[int(np.argmin(np.abs(s - lm[side][k])))].tolist() for k in ("C6", "C2", "C1", "dura")}
        # parent direction exactly as ref_vmr0248.py
        sub = vmr["vessels"][side + "_subclavian"]["points"]
        P0 = vmr["vessels"][side + "_vertebral"]["points"][0]
        k = int(np.argmin(np.linalg.norm(sub - P0, axis=1)))
        pd = sub[min(k + 3, len(sub) - 1)] - sub[max(k - 3, 0)]
        pdirs[side] = pd / np.linalg.norm(pd)
        print(side, {k: (round(v, 1) if isinstance(v, float) else v) for k, v in lm[side].items() if k != "definitions"})
    jobs = []
    for v, side in side_of.items():
        for variant, clp, sp, loader in (
                ("lofted_vmtk", os.path.join(DERIVED, "lofted", "%s_vmtk.json" % side), None, True),
                ("ct_stageA", os.path.join(DERIVED, TAG, "%s_stageA_vmtk.json" % v), os.path.join(DERIVED, TAG, "%s_stageA.vtp" % v), False),
                ("ct_iso", os.path.join(DERIVED, TAG, "%s_iso_vmtk.json" % v), os.path.join(DERIVED, TAG, "%s_iso.vtp" % v), False)):
            cl = json.load(open(clp))
            P, _, s = vm.resample(np.asarray(cl["points"]), None)
            lmk = {}
            for k, X in lm_pts[side].items():
                lmk[k] = float(s[int(np.argmin(np.linalg.norm(P - np.asarray(X), axis=1)))])
            jobs.append((variant, side, clp, sp, loader, lmk, pdirs[side]))
    with Pool(6) as pool:
        out = pool.map(job, jobs, chunksize=1)
    res = {"version": vm.VERSION, "source": "VMR 0248_H_AOCERE_CAS", "voxel_mm": VOXEL, "seg_tag": TAG,
           "citation": "Wilson NM, Ortiz AK, Johnson AB. The Vascular Model Repository. J Med Devices 7(4):040923, "
                       "2013; NLM R01LM013120; NHLBI HHSN268201100035C; www.vascularmodel.com (0248_H_AOCERE_CAS)",
           "landmarks_ct": lm, "landmark_points_ras": lm_pts, "foramen_magnum": {"basion_yz": BASION.tolist(),
                                                                              "opisthion_yz": OPISTHION.tolist()},
           "variants": {}, "definitions": vm.METRIC_DEFINITIONS}
    for variant, side, r in out:
        res["variants"].setdefault(variant, {})[side] = r
    res["grades"] = {}
    for variant, d in res["variants"].items():
        res["grades"][variant] = vs.grade_set([d["right"], d["left"]])
        res["grades"][variant]["dominance_L_over_R"] = vm.dominance(d["left"]["calibre"]["r_median_mm"],
                                                                    d["right"]["calibre"]["r_median_mm"])
    res["seconds"] = round(time.time() - t0, 1)
    jdump(res, OUT)
    for variant, g in res["grades"].items():
        print(variant, json.dumps(g["set_rules"]))
        for side, pg in zip(("right", "left"), g["per_va"]):
            print("   ", side, {k: v.get("result") for k, v in pg.items()})
    print("wrote", OUT, "%.0f s" % (time.time() - t0))


if __name__ == "__main__":
    main()
