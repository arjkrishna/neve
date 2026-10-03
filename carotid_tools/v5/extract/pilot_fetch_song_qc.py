#!/usr/bin/env python3
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\pilot_fetch_song_qc.py (= /sp/v5_va/pilot_fetch_song_qc.py in the build containers)
"""pilot_fetch_song_qc.py (pass 1): per Song case, read the shipped ct.nii.gz and write
  - <img>/<case>/qc_pass1.json: header, HU statistics and histogram, clipping analysis, body mask stats, aorta
    tracking from the bottom slice (limbs, arch top), body-area profile, candidate bright blobs on axial levels
    with their HU (for the CCA / subclavian quick look), MIP crop;
  - <img>/<case>/mip_neck_coronal.png, mip_neck_sagittal.png: neck MIPs (arch .. skull base) in RAS;
  - <img>/<case>/qc_axial.png: numbered candidate blobs on the pick levels, plus the HU histogram;
  - <derived>/<case>/ct_bg-100.nii.gz: re-windowed SIDE FILE: every voxel that is exactly 0 AND outside the body
    mask is set to -100 HU; nothing else changes (int16, same affine / sform / qform codes);
  - <derived>/<case>/ct_negzero-100.nii.gz: second, optional side file: every voxel stored as -0.0 (the clip of an
    originally negative HU: air, fat, lung) is set to -100, +0.0 (originally 0..50 HU) stays 0;
  - <derived>/<case>/body_mask.nii.gz (uint8).
The shipped ct.nii.gz is only read, never rewritten.
usage: pilot_fetch_song_qc.py CASE [CASE ...]   (neve-build-stagea; env NO_SIDE=1 skips the side files)
"""
import sys, os, json, time
import numpy as np
import nibabel as nib
from scipy import ndimage as ndi
from skimage import measure
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

IMG = "/repo/carotid_data/va_sources/thin_slice/song"
DER = "/repo/carotid_data/va_sources/derived/thin_slice/song"


def disk(r):
    r = max(1, int(round(r)))
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return x * x + y * y <= r * r


def save_like(arr, ref, hdr, path):
    """save on the shipped grid with the shipped affine AND the shipped qform/sform codes; verify on reload"""
    qc, sc = int(ref.header["qform_code"]), int(ref.header["sform_code"])
    im = nib.Nifti1Image(arr, ref.affine, hdr)
    im.set_sform(ref.affine, code=sc if sc else 0)
    if qc:
        im.set_qform(ref.affine, code=qc)
    else:
        im.set_qform(ref.header.get_qform(), code=0)
    nib.save(im, path)
    chk = nib.load(path)
    assert chk.shape == ref.shape and np.allclose(chk.affine, ref.affine, atol=1e-4), "affine mismatch " + path
    assert int(chk.header["qform_code"]) == qc and int(chk.header["sform_code"]) == sc, "code mismatch " + path


def pct(a, q):
    return [round(float(x), 1) for x in np.percentile(a, q)]


def body_mask_slice(c, se):
    m = ndi.binary_closing(c, structure=se, iterations=1, border_value=0)
    m = ndi.binary_fill_holes(m)
    return m


def comp_stats(mask2d, v2d, sp, origin_xy, min_area=4.0, max_area=4000.0):
    lab = measure.label(mask2d, connectivity=1)
    out = []
    pa = sp[0] * sp[1]
    for r in measure.regionprops(lab):
        area = r.area * pa
        if area < min_area or area > max_area:
            continue
        cm = r.coords
        vals = v2d[cm[:, 0], cm[:, 1]]
        sub = np.zeros(r.image.shape, bool); sub[:] = r.image
        er = ndi.binary_erosion(sub)
        core = vals if er.sum() < 3 else v2d[r.bbox[0]:r.bbox[2], r.bbox[1]:r.bbox[3]][er]
        per = max(r.perimeter, 1e-6) * sp[0]
        circ = float(min(1.0, 4 * np.pi * area / (per * per)))
        cy, cx = r.centroid
        out.append(dict(i=int(cy), j=int(cx), x=round(origin_xy[0] + cy * sp[0], 1), y=round(origin_xy[1] + cx * sp[1], 1),
                        area_mm2=round(area, 1), circ=round(circ, 2), hu_med=round(float(np.median(core)), 0),
                        hu_p90=round(float(np.percentile(core, 90)), 0), hu_mean=round(float(core.mean()), 0),
                        hu_cv=round(float(core.std() / max(1.0, abs(core.mean()))), 3),
                        n=int(r.area), label=int(r.label)))
    return out, lab


def main():
    for case in sys.argv[1:]:
        t0 = time.time()
        d = os.path.join(IMG, case); dd = os.path.join(DER, case); os.makedirs(dd, exist_ok=True)
        img = nib.load(os.path.join(d, "ct.nii.gz"))
        h = img.header
        rec = dict(case=case, date=time.strftime("%Y-%m-%d %H:%M"))
        rec["header"] = dict(shape=[int(s) for s in img.shape], zooms=[round(float(z), 5) for z in h.get_zooms()],
                             on_disk_dtype=str(h.get_data_dtype()),
                             scl_slope=None if not np.isfinite(h["scl_slope"]) else float(h["scl_slope"]),
                             scl_inter=None if not np.isfinite(h["scl_inter"]) else float(h["scl_inter"]),
                             qform_code=int(h["qform_code"]), sform_code=int(h["sform_code"]),
                             affine=np.round(img.affine, 5).tolist(), axcodes="".join(nib.aff2axcodes(img.affine)))
        if img.ndim != 3:
            rec["error"] = "not 3-D"; json.dump(rec, open(os.path.join(d, "qc_pass1.json"), "w"), indent=1); continue
        a = np.asanyarray(img.dataobj)
        v0 = a.astype(np.float32); del a
        # ---- HU statistics on the shipped array (every voxel) ----
        flat = v0.ravel()
        neg0 = int(np.count_nonzero((flat == 0) & np.signbit(flat)))
        n0 = int(np.count_nonzero(flat == 0))
        nz = flat[flat != 0]
        st = dict(n_vox=int(flat.size), min=float(flat.min()), max=float(flat.max()),
                  p=dict(zip(["p0.1", "p1", "p5", "p25", "p50", "p75", "p95", "p99", "p99.9"],
                             pct(flat[::7], [0.1, 1, 5, 25, 50, 75, 95, 99, 99.9]))),
                  frac_eq0=round(n0 / flat.size, 4), frac_negzero=round(neg0 / flat.size, 4),
                  frac_poszero=round((n0 - neg0) / flat.size, 4), frac_lt0=round(float(np.mean(flat[::7] < 0)), 5),
                  min_nonzero=float(nz.min()) if nz.size else None,
                  nonzero_p=dict(zip(["p1", "p50", "p99", "p99.9"], pct(nz[::7], [1, 50, 99, 99.9]))) if nz.size else None,
                  integer_valued=bool(np.all(nz[::97] == np.round(nz[::97]))))
        edges = np.arange(-1024, 3073, 16)
        hc, _ = np.histogram(np.clip(flat[::3], -1024, 3071), bins=edges)
        st["hist"] = dict(note="every 3rd voxel, clipped to [-1024, 3071]; 16 HU bins", start=-1024, step=16,
                          counts=hc.astype(int).tolist())
        hf, _ = np.histogram(nz[(nz > 0) & (nz <= 200)], bins=np.arange(0, 201, 5))
        st["hist_0_200_nonzero"] = dict(start=0, step=5, counts=hf.astype(int).tolist())
        clipped = bool(st["frac_lt0"] < 1e-4 and st["min"] >= 0)
        st["clipped_at_0"] = clipped
        rec["hu"] = st
        del flat, nz
        # ---- RAS view for the analysis ----
        ras = nib.as_closest_canonical(nib.Nifti1Image(v0, img.affine))
        A = ras.affine; v = np.asanyarray(ras.dataobj)
        sp = [float(abs(A[0, 0])), float(abs(A[1, 1])), float(abs(A[2, 2]))]
        org = A[:3, 3]; nx, ny, nzs = v.shape
        zmm = org[2] + np.arange(nzs) * A[2, 2]
        rec["ras"] = dict(spacing=[round(s, 5) for s in sp], origin=np.round(org, 2).tolist(),
                          z_range_mm=[round(float(zmm[0]), 1), round(float(zmm[-1]), 1)],
                          z_extent_mm=round(float(nzs * sp[2]), 1))
        # ---- body mask (2x in-plane subsample, per axial slice) ----
        if clipped:
            cand = (v > 0) | ((v == 0) & ~np.signbit(v))
        else:
            cand = v > -500
        cs = cand[::2, ::2, :]
        se = disk(3.0 / (2 * sp[0]))
        bm_s = np.zeros(cs.shape, bool)
        for k in range(nzs):
            if cs[:, :, k].any():
                bm_s[:, :, k] = body_mask_slice(cs[:, :, k], se)
        bm_s = ndi.binary_opening(bm_s, structure=np.ones((3, 3, 1), bool))
        lab, n = ndi.label(bm_s)
        if n > 1:
            sizes = ndi.sum(bm_s, lab, range(1, n + 1)); bm_s = lab == (1 + int(np.argmax(sizes)))
        del lab, cs
        bm = np.repeat(np.repeat(bm_s, 2, 0), 2, 1)[:nx, :ny, :]
        area_cm2 = bm_s.sum(axis=(0, 1)) * (4 * sp[0] * sp[1]) / 100.0
        inside = v[bm]
        outside = v[~bm]
        rec["body"] = dict(frac_volume=round(float(bm.mean()), 4),
                           inside_frac_negzero=round(float(np.mean((inside == 0) & np.signbit(inside))), 4),
                           inside_frac_poszero=round(float(np.mean((inside == 0) & ~np.signbit(inside))), 4),
                           inside_p=dict(zip(["p1", "p50", "p99", "p99.9"], pct(inside[::5], [1, 50, 99, 99.9]))),
                           outside_frac_eq0=round(float(np.mean(outside == 0)), 4),
                           outside_frac_poszero=round(float(np.mean((outside == 0) & ~np.signbit(outside))), 5),
                           outside_frac_gt0=round(float(np.mean(outside > 0)), 5),
                           outside_max=float(outside.max()) if outside.size else None)
        del inside, outside
        # ---- aorta tracking from the bottom slice ----
        T_LO = 150.0
        def aorta_comps(k, prev=None):
            s = v[:, :, k]
            # bone / dense-venous barrier: HU >= 800 dilated 1 voxel keeps the descending aorta off the spine
            barrier = ndi.binary_dilation(s >= 800, structure=disk(1.0))
            m = (s >= T_LO) & bm[:, :, k] & ~barrier
            m = ndi.binary_opening(m, structure=disk(1.0 / sp[0]))
            cs_, lab_ = comp_stats(m, s, sp, org[:2], min_area=80, max_area=6000)
            keep = []
            for c in cs_:
                if prev is not None:
                    ov = np.logical_and(lab_ == c["label"], prev).sum()
                    if ov < 10 or c["hu_cv"] > 0.35 or c["hu_med"] < 180:
                        continue
                    c["overlap"] = int(ov)
                elif not (c["circ"] >= 0.55 and c["hu_cv"] <= 0.30 and c["hu_med"] >= 200 and c["area_mm2"] >= 150):
                    continue
                keep.append(c)
            return keep, lab_
        k0 = 0
        limbs, lab0 = aorta_comps(k0)
        ao = dict(bottom_k=k0, bottom_z=round(float(zmm[k0]), 1),
                  bottom_limbs=[{kk: c[kk] for kk in ("x", "y", "area_mm2", "circ", "hu_med", "hu_cv")} for c in limbs])
        if limbs:
            prev = np.isin(lab0, [c["label"] for c in limbs])
            hu_ao = [c["hu_med"] for c in limbs]
            top_k = k0; track = []
            a_min = float(max(150.0, 0.6 * np.median([c["area_mm2"] for c in limbs])))
            ao["track_min_area_mm2"] = round(a_min, 0)
            for k in range(1, min(nzs, int(160 / sp[2]))):
                cs_, lab_ = aorta_comps(k, prev)
                big = [c for c in cs_ if c["area_mm2"] >= a_min]  # BCT / CCA / SCA (<= ~180 mm2) end the track
                if not big:
                    break
                prev = np.isin(lab_, [c["label"] for c in big]); top_k = k
                track.append((k, len(big), round(sum(c["area_mm2"] for c in big), 0)))
                hu_ao += [c["hu_med"] for c in big]
            ao["arch_top_k"] = int(top_k); ao["arch_top_z"] = round(float(zmm[top_k]), 1)
            ao["arch_top_above_bottom_mm"] = round(float(top_k * sp[2]), 1)
            ao["n_limbs_track"] = [t[1] for t in track[::max(1, len(track) // 12)]]
            ao["aorta_hu_med"] = float(np.median(hu_ao))
            ao["two_limbs_at_bottom"] = len(limbs) >= 2
        else:
            ao["arch_top_k"] = None; ao["two_limbs_at_bottom"] = False
        # by-eye override of the arch top (read off mip_neck_coronal.png) when the automatic track fails
        ov = os.environ.get("ARCH_TOP_Z")
        if ov is not None:
            ao["arch_top_z_auto"] = ao.get("arch_top_z")
            ao["arch_top_k"] = int(np.argmin(np.abs(zmm - float(ov))))
            ao["arch_top_z"] = round(float(zmm[ao["arch_top_k"]]), 1)
            ao["arch_top_above_bottom_mm"] = round(float(ao["arch_top_k"] * sp[2]), 1)
            ao["arch_top_source"] = "by eye (ARCH_TOP_Z=%s)" % ov
        else:
            ao["arch_top_source"] = "automatic track"
        rec["aorta_auto"] = ao
        # ---- levels ----
        ka = ao["arch_top_k"] if ao["arch_top_k"] is not None else 0
        za = zmm[ka]
        # neck: minimum body area between arch+30 and arch+200 mm
        rng = [k for k in range(nzs) if za + 30 <= zmm[k] <= za + 200]
        kmin = int(rng[int(np.argmin(area_cm2[rng]))]) if rng else nzs // 2
        rec["levels"] = dict(arch_top_z=round(float(za), 1), neck_min_area_z=round(float(zmm[kmin]), 1),
                             neck_min_area_cm2=round(float(area_cm2[kmin]), 1),
                             area_profile_cm2_every10mm=[[round(float(zmm[k]), 0), round(float(area_cm2[k]), 0)]
                                                         for k in range(0, nzs, max(1, int(10 / sp[2])))])
        aoh = ao.get("aorta_hu_med") or 350.0
        T_BONE = float(max(700.0, 1.35 * aoh))
        rec["levels"]["bone_threshold_hu"] = T_BONE
        pick_levels = {"L1_arch+15": za + 15, "L2_arch+35": za + 35, "L3_arch+60": za + 60, "L4_arch+100": za + 100}
        cand_out = {}
        axes_imgs = []
        for name, z in pick_levels.items():
            k = int(np.clip(np.argmin(np.abs(zmm - z)), 0, nzs - 1))
            s = v[:, :, k]
            m = (s >= T_LO) & (s < T_BONE) & bm[:, :, k]
            cs_, lab_ = comp_stats(m, s, sp, org[:2], min_area=4, max_area=600)
            bone = s >= T_BONE
            boned = ndi.binary_dilation(bone, structure=disk(1.0 / sp[0]))
            for c in cs_:
                c["touch_bone"] = bool(np.logical_and(lab_ == c["label"], boned).any())
            cs_.sort(key=lambda c: -c["area_mm2"])
            cs_ = cs_[:40]
            for n_, c in enumerate(cs_):
                c["id"] = n_
            bk = np.nonzero(bm[:, :, k])
            midx = float(np.mean(bk[0])) * sp[0] + org[0] if bk[0].size else org[0] + nx * sp[0] / 2
            midy = float(np.mean(bk[1])) * sp[1] + org[1] if bk[0].size else org[1] + ny * sp[1] / 2
            # spinal canal: the hole of the vertebral ring nearest the body midline (area 60..500 mm2)
            canal = None
            f = ndi.binary_fill_holes(bone)
            hl, hn = ndi.label(f & ~bone)
            best = None
            for r in measure.regionprops(hl):
                ar = r.area * sp[0] * sp[1]
                if 60 <= ar <= 500:
                    cxw = org[0] + r.centroid[0] * sp[0]; cyw = org[1] + r.centroid[1] * sp[1]
                    dist = abs(cxw - midx)
                    if dist < 20 and cyw < midy + 10 and (best is None or dist < best[0]):  # not the mandible ring
                        best = (dist, cxw, cyw, ar)
            if best:
                canal = dict(x=round(best[1], 1), y=round(best[2], 1), area_mm2=round(best[3], 0))
            bxy = np.nonzero(bone & bm[:, :, k])
            spx = round(float(org[0] + np.median(bxy[0]) * sp[0]), 1) if bxy[0].size > 50 else None
            spy = round(float(org[1] + np.median(bxy[1]) * sp[1]), 1) if bxy[0].size > 50 else None
            cand_out[name] = dict(k=k, z=round(float(zmm[k]), 1), body_mid_x=round(midx, 1), body_mid_y=round(midy, 1),
                                  bone_median_x=spx, bone_median_y=spy,
                                  canal=canal,
                                  blobs=[{kk: c[kk] for kk in ("id", "x", "y", "area_mm2", "circ", "hu_med", "hu_p90", "hu_cv", "touch_bone")} for c in cs_])
            axes_imgs.append((name, k, s, cs_, sp))
        sx = [cand_out[n]["bone_median_x"] for n in ("L2_arch+35", "L3_arch+60", "L4_arch+100") if cand_out[n]["bone_median_x"] is not None]
        spine_x = float(np.median(sx)) if sx else cand_out["L3_arch+60"]["body_mid_x"]
        for n in cand_out:
            cand_out[n]["mid_x"] = round(spine_x, 1)
        rec["candidates"] = cand_out
        rec["spine_mid_x"] = round(spine_x, 1)
        # ---- automatic picks (to be confirmed by eye on qc_axial.png) ----
        def auto_pick(level, kind):
            L = cand_out[level]; xm = L["mid_x"]; out = {}
            yc = L["canal"]["y"] if L["canal"] else None
            for side, sgn in (("R", 1), ("L", -1)):
                best = None
                for c in L["blobs"]:
                    dx = sgn * (c["x"] - xm)
                    arterial = 0.75 * aoh <= c["hu_med"] <= 1.5 * aoh
                    if kind == "CCA":  # the LARGEST round arterial blob 10-45 mm off the midline
                        ok = (12 <= c["area_mm2"] <= 90 and c["circ"] >= 0.6 and not c["touch_bone"] and 10 <= dx <= 45
                              and arterial and (yc is None or 8 <= c["y"] - yc <= 50))
                    else:  # SCA (or the BCT on the right at low levels): the most POSTERIOR arterial blob >= 30 mm2
                        ok = (30 <= c["area_mm2"] <= 250 and c["circ"] >= 0.45 and not c["touch_bone"] and 5 <= dx <= 90
                              and arterial
                              and (c["y"] >= yc + 5 if yc is not None else c["y"] >= L["body_mid_y"] - 30))
                    if not ok:
                        continue
                    if kind == "CCA" and (best is None or c["area_mm2"] > best["area_mm2"]):
                        best = c
                    elif kind == "SCA" and (best is None or c["y"] < best["y"]):
                        best = c
                if best:
                    out[side] = dict(level=level, id=best["id"], hu_med=best["hu_med"], x=best["x"], y=best["y"],
                                     area_mm2=best["area_mm2"])
            return out
        rec["auto_picks"] = dict(CCA=auto_pick("L3_arch+60", "CCA"), CCA_hi=auto_pick("L4_arch+100", "CCA"),
                                 SCA=auto_pick("L1_arch+15", "SCA"), SCA_hi=auto_pick("L2_arch+35", "SCA"))
        rec["auto_picks"]["rules"] = ("CCA: L3 (arch+60; CCA_hi = L4, arch+100), area 12-90 mm2, circ >= 0.6, median HU "
                                      "0.75-1.5 x aorta, not touching bone, 10-45 mm off the body midline, 8-50 mm anterior "
                                      "of the canal when found; the LARGEST such blob per side. SCA: L1 (arch+15; SCA_hi = L2, "
                                      "arch+35), area 30-250 mm2, circ >= 0.45, median HU 0.75-1.5 x aorta, not touching bone, "
                                      "5-90 mm off the midline, anterior of the canal; the most POSTERIOR such blob per side. "
                                      "Median HU of the 1-voxel-eroded blob. Every pick is confirmed or overridden by eye "
                                      "(pilot_fetch_song_review.json).")
        # ---- MIP crop and bone suppression ----
        zlo = max(zmm[0], za - 30); zhi = min(zmm[-1], za + 250)
        k_lo = int(np.argmin(np.abs(zmm - zlo))); k_hi = int(np.argmin(np.abs(zmm - zhi)))
        sub = v[:, :, k_lo:k_hi + 1]
        bone = sub >= T_BONE
        bonef = np.zeros_like(bone)
        min_hole = 60.0 / (sp[0] * sp[1])
        for k in range(bone.shape[2]):
            b = bone[:, :, k]
            if not b.any():
                continue
            f = ndi.binary_fill_holes(b)
            holes = f & ~b
            if holes.any():
                hl, hn = ndi.label(holes)
                hs = ndi.sum(holes, hl, range(1, hn + 1))
                big = np.isin(hl, 1 + np.nonzero(hs >= min_hole)[0])
                bonef[:, :, k] = b | big
            else:
                bonef[:, :, k] = b
        T_MIP = float(max(T_LO, 0.5 * aoh))  # drop most cancellous bone (VA / CCA lumen ~ aorta HU)
        lum = sub >= T_MIP
        # geodesic growth of the bone through >= T_MIP voxels for ~3 mm (cortical partial volume, dense trabeculae);
        # vessels touching bone (V2 in the foramina) lose those voxels, so the VA shows as a dashed line there
        n_geo = max(1, int(round(3.0 / sp[0])))
        cross = ndi.generate_binary_structure(3, 1)
        boneg = ndi.binary_dilation(bonef, structure=cross, iterations=n_geo, mask=lum | bonef)
        boneg = ndi.binary_dilation(boneg, structure=np.ones((3, 3, 3), bool))
        bonef = ndi.binary_dilation(bonef, structure=np.ones((3, 3, 3), bool))
        ves = np.where(lum & ~bonef, sub, 0).astype(np.float32)
        vesg = np.where(lum & ~boneg, sub, 0).astype(np.float32)
        del bone, lum, boneg
        midx_i = int(round((spine_x - org[0]) / sp[0]))
        rec["mip"] = dict(z_lo=round(float(zmm[k_lo]), 1), z_hi=round(float(zmm[k_hi]), 1), mid_x_index=midx_i,
                          vessel_window=[round(T_MIP, 0), round(min(T_BONE, max(450.0, 1.1 * aoh)), 0)],
                          bone_rule="simple: HU >= bone threshold, axial holes >= 60 mm2 filled, dilated 1 voxel; "
                                    "geodesic: the same grown %d voxels through HU >= vessel-window low, dilated 1" % n_geo)
        vmax = rec["mip"]["vessel_window"][1]
        T_LO_disp = T_MIP
        ext_c = [org[0], org[0] + nx * sp[0], zmm[k_lo], zmm[k_hi]]
        # coronal (radiological: patient right on image left; RAS x increases to the right of the patient)
        full_c = np.clip(sub.max(axis=1), 0, 1500).T
        ves_c = ves.max(axis=1).T
        cor_va = vesg.max(axis=1).T
        fig, ax = plt.subplots(1, 3, figsize=(21, 9))
        for a_, im, ttl, vr in ((ax[0], full_c, "coronal MIP, all (0..1500 HU)", (0, 1500)),
                                (ax[1], ves_c, "coronal MIP, bone-suppressed (%d..%d HU)" % (T_LO_disp, vmax), (T_LO_disp, vmax)),
                                (ax[2], cor_va, "coronal MIP, geodesic bone removal (VA dashed in foramina)", (T_LO_disp, vmax))):
            a_.imshow(im[:, ::-1], cmap="gray", origin="lower", vmin=vr[0], vmax=vr[1], aspect=sp[2] / sp[0],
                      extent=[ext_c[1], ext_c[0], ext_c[2], ext_c[3]])
            a_.set_title(ttl); a_.set_xlabel("x RAS mm  (patient R on image left)"); a_.set_ylabel("z mm")
            a_.axhline(za, color="c", lw=0.6, ls="--"); a_.axhline(zmm[kmin], color="y", lw=0.6, ls="--")
            a_.text(0.02, 0.97, "R", transform=a_.transAxes, color="r", fontsize=14, va="top")
            a_.text(0.95, 0.97, "L", transform=a_.transAxes, color="r", fontsize=14, va="top")
        fig.suptitle("%s  coronal neck MIP  z %.0f..%.0f mm  (cyan: auto arch top, yellow: neck min-area level)" %
                     (case, zmm[k_lo], zmm[k_hi]))
        plt.tight_layout(); plt.savefig(os.path.join(d, "mip_neck_coronal.png"), dpi=70); plt.close(fig)
        # sagittal: right and left paramedian slabs (|x - mid| 6..40 mm), plus the full width.
        # RAS+ (as_closest_canonical): x index increases toward the patient's RIGHT (A[0,0] > 0).
        xs = org[0] + np.arange(nx) * A[0, 0]
        xmid = spine_x
        right_sel = (xs > xmid + 6) & (xs < xmid + 40)
        left_sel = (xs < xmid - 6) & (xs > xmid - 40)
        ext_s = [org[1], org[1] + ny * sp[1], zmm[k_lo], zmm[k_hi]]
        fig, ax = plt.subplots(1, 3, figsize=(18, 9))
        panels = (("RIGHT slab (x mid+6..+40 mm), bone-suppressed", ves[right_sel].max(axis=0).T, (T_LO_disp, vmax)),
                  ("LEFT slab (x mid-40..-6 mm), bone-suppressed", ves[left_sel].max(axis=0).T, (T_LO_disp, vmax)),
                  ("all, full width (0..1500 HU)", np.clip(sub.max(axis=0), 0, 1500).T, (0, 1500)))
        for a_, (ttl, im, vr) in zip(ax, panels):
            a_.imshow(im, cmap="gray", origin="lower", vmin=vr[0], vmax=vr[1], aspect=sp[2] / sp[1], extent=ext_s)
            a_.set_title(ttl, fontsize=10); a_.set_xlabel("y RAS mm (anterior to the right)"); a_.set_ylabel("z mm")
            a_.axhline(za, color="c", lw=0.6, ls="--"); a_.axhline(zmm[kmin], color="y", lw=0.6, ls="--")
        fig.suptitle("%s  sagittal neck MIPs  (VA runs in the posterior third, beside the vertebral bodies)" % case)
        plt.tight_layout(); plt.savefig(os.path.join(d, "mip_neck_sagittal.png"), dpi=70); plt.close(fig)
        rec["mip"]["slab_note"] = "RAS: +x = patient right; right slab x in mid+6..+40 mm, left slab mid-40..-6 mm"
        del ves, vesg, bonef, sub
        # ---- axial pick panels + histogram ----
        fig, axs = plt.subplots(2, 3, figsize=(24, 16))
        ax = axs.ravel()
        ext = [org[0], org[0] + nx * sp[0], org[1], org[1] + ny * sp[1]]
        pk_col = {"CCA": "red", "CCA_hi": "magenta", "SCA": "cyan", "SCA_hi": "deepskyblue"}
        for a_, (name, k, s, cs_, sp_) in zip(ax, axes_imgs):
            L = cand_out[name]
            a_.imshow(s.T[::-1, ::-1], cmap="gray", vmin=0, vmax=max(500, vmax), extent=[ext[1], ext[0], ext[2], ext[3]])
            for c in cs_:
                a_.text(c["x"], c["y"], str(c["id"]), color="yellow" if not c["touch_bone"] else "orange", fontsize=9,
                        ha="center", va="center", fontweight="bold", clip_on=True)
            for kind, col in pk_col.items():
                for side, pk in rec["auto_picks"][kind].items():
                    if not isinstance(pk, dict):
                        continue
                    if pk["level"] == name:
                        a_.add_patch(plt.Circle((pk["x"], pk["y"]), 5, fill=False, color=col, lw=1.2))
                        a_.text(pk["x"], pk["y"] + 7, "%s %s %d" % (kind, side, pk["hu_med"]), color=col, fontsize=9, ha="center", clip_on=True)
            if L["canal"]:
                a_.plot(L["canal"]["x"], L["canal"]["y"], "g+", ms=10)
            cy0 = L["canal"]["y"] + 15 if L["canal"] else L["body_mid_y"]
            a_.set_xlim(L["mid_x"] + 60, L["mid_x"] - 60)
            a_.set_ylim(cy0 - 45, cy0 + 60)
            a_.set_title("%s z=%.0f (k %d); yellow ids = candidate blobs (orange: touch bone)" % (name, zmm[k], k), fontsize=9)
            a_.set_xlabel("x RAS mm (patient R on image left)"); a_.set_ylabel("y RAS mm (anterior up)")
        s = v[:, :, 0]
        ax[4].imshow(s.T[::-1, ::-1], cmap="gray", vmin=0, vmax=max(500, vmax), extent=[ext[1], ext[0], ext[2], ext[3]])
        for c in ao["bottom_limbs"]:
            ax[4].plot(c["x"], c["y"], "c+", ms=14)
        ax[4].set_title("bottom slice z=%.0f; auto aorta limbs (cyan +): %d" % (zmm[0], len(ao["bottom_limbs"])), fontsize=9)
        e = np.array(rec["hu"]["hist"]["counts"], float)
        xe = -1024 + 16 * np.arange(len(e))
        ax[5].semilogy(xe, np.maximum(e, 0.5), drawstyle="steps-post")
        ax[5].set_title("HU histogram (every 3rd voxel); frac==0 %.2f (-0.0 %.2f), min nonzero %s" %
                        (rec["hu"]["frac_eq0"], rec["hu"]["frac_negzero"], rec["hu"]["min_nonzero"]), fontsize=9)
        ax[5].set_xlabel("HU"); ax[5].set_xlim(-1100, 3100)
        fig.suptitle("%s  axial quick look (window 0..%d HU); circles = automatic CCA / SCA picks with median HU" %
                     (case, max(500, vmax)))
        plt.tight_layout(); plt.savefig(os.path.join(d, "qc_axial.png"), dpi=65); plt.close(fig)
        # ---- side files (shipped grid, shipped affine) ----
        if not os.environ.get("NO_SIDE"):
            hdr = img.header.copy(); hdr.set_data_dtype(np.int16); hdr["scl_slope"] = 1; hdr["scl_inter"] = 0
            # body mask back to the shipped axis order
            bm_img = nib.Nifti1Image(bm.astype(np.uint8), A)
            ornt = nib.orientations.ornt_transform(nib.orientations.io_orientation(A), nib.orientations.io_orientation(img.affine))
            bm0 = nib.orientations.apply_orientation(bm.astype(np.uint8), ornt).astype(bool)
            assert bm0.shape == v0.shape
            out = np.rint(v0).astype(np.int16)
            out[(v0 == 0) & ~bm0] = -100
            hm = img.header.copy(); hm.set_data_dtype(np.uint8); hm["scl_slope"] = 1; hm["scl_inter"] = 0
            save_like(bm0.astype(np.uint8), img, hm, os.path.join(dd, "body_mask.nii.gz"))
            save_like(out, img, hdr, os.path.join(dd, "ct_bg-100.nii.gz"))
            n_bg = int(np.count_nonzero(out == -100))
            sides = dict(ct_bg_minus100=dict(path="carotid_data/va_sources/derived/thin_slice/song/%s/ct_bg-100.nii.gz" % case,
                                             rule="voxel == 0 and outside the body mask -> -100; all else unchanged; int16",
                                             n_set=n_bg, frac_set=round(n_bg / out.size, 4)),
                         body_mask="carotid_data/va_sources/derived/thin_slice/song/%s/body_mask.nii.gz" % case)
            if clipped:
                out = np.rint(v0).astype(np.int16)
                negz = (v0 == 0) & np.signbit(v0)
                out[negz] = -100
                save_like(out, img, hdr, os.path.join(dd, "ct_negzero-100.nii.gz"))
                sides["ct_negzero_minus100"] = dict(
                    path="carotid_data/va_sources/derived/thin_slice/song/%s/ct_negzero-100.nii.gz" % case,
                    rule="voxel stored as -0.0 (clipped from a negative HU) -> -100; +0.0 and all else unchanged; int16",
                    n_set=int(negz.sum()), frac_set=round(float(negz.mean()), 4))
            del out
            rec["side_files"] = sides
        elif os.path.exists(os.path.join(d, "qc_pass1.json")):  # NO_SIDE re-run: keep the record of the side files
            old = json.load(open(os.path.join(d, "qc_pass1.json")))
            if old.get("side_files"):
                rec["side_files"] = old["side_files"]
        rec["seconds"] = round(time.time() - t0, 1)
        json.dump(rec, open(os.path.join(d, "qc_pass1.json"), "w"), indent=1)
        print(case, "clipped", clipped, "limbs", len(ao["bottom_limbs"]), "arch_top_above_bottom",
              ao.get("arch_top_above_bottom_mm"), "aorta_hu", ao.get("aorta_hu_med"), "%.0fs" % rec["seconds"], flush=True)
        del v0, v, bm, ras


if __name__ == "__main__":
    main()
