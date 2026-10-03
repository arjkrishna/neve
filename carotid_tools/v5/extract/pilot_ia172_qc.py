# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\pilot_ia172_qc.py (= /sp/v5_va/pilot_ia172_qc.py in the build containers)
"""pilot_ia172_qc: quick-look QC, neck MIP PNG and meta.json for fetched 172-series cases.

usage: python3 pilot_ia172_qc.py <case> [<case> ...]     (case = ia172_<sid>)
reads : thin_slice/ia172/<case>/ct.nii.gz (native RAS HU); derived/thin_slice/ia172/<case>/{fetch,convert}.json,
        ts_fast_1p5.nii.gz (TotalSegmentator total --fast on the 1.5 mm copy, pilot_ia172_ts.sh);
        /sp/v5_va/results/pilot_ia172_candidates.json (headers + Metadata.csv);
        /sp/v5_va/results/pilot_ia172_visual.json (my visual reading of the MIPs, if present).
writes: thin_slice/ia172/<case>/meta.json and mip_neck.png; derived/.../qc.json (same content as meta.json).

Quick-look HU: median (p10, p90) of the native CT sampled at the 1-voxel-eroded (1.5 mm) TotalSegmentator fast masks.
Injection side = the brachiocephalic vein (BCV) with the higher median. Phase call:
  non-contrast  CCA median < 150 HU;
  arterial      CCA median >= 250 HU and contralateral BCV < 200 HU (the polytrauma guard of donor_pool_plan Stage 2);
  check         anything else (then the visual reading decides).
Coverage: LSA touches the aorta and RSA touches the brachiocephalic trunk (or the aorta) inside the image, with the
origin >= 5 mm above the bottom slice; C1..C7 and T1 present; skull present and the image top >= 30 mm above the C1 top.
Motion proxy: largest slice-to-slice jump of the body (HU > -300) centroid and area over the neck stretch.
"""
import json
import os
import sys
import time

import nibabel as nib
import numpy as np
from scipy import ndimage

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

IMG = "/repo/carotid_data/va_sources/thin_slice/ia172"
DER = "/repo/carotid_data/va_sources/derived/thin_slice/ia172"
LAB = json.load(open(os.path.join(DER, "_ts_total_labels.json")))
ID = {v: int(k) for k, v in LAB.items()}
CAND = json.load(open("/sp/v5_va/results/pilot_ia172_candidates.json"))["candidates"]
VIS_P = "/sp/v5_va/results/pilot_ia172_visual.json"
VIS = json.load(open(VIS_P)) if os.path.exists(VIS_P) else {}
VESSELS = ["aorta", "brachiocephalic_trunk", "subclavian_artery_right", "subclavian_artery_left",
           "common_carotid_artery_right", "common_carotid_artery_left", "brachiocephalic_vein_right",
           "brachiocephalic_vein_left", "superior_vena_cava"]
VERT = ["vertebrae_C1", "vertebrae_C2", "vertebrae_C3", "vertebrae_C4", "vertebrae_C5", "vertebrae_C6",
        "vertebrae_C7", "vertebrae_T1"]


def r1(x):
    return None if x is None else round(float(x), 1)


def bone_suppressed(v, sp, hole_mm2=120.0):
    """display-only bone removal: cortical bone (> 600 HU), closed by ~1.5 mm, plus every axial hole in it larger than
    hole_mm2 (vertebral-body trabecular interior, spinal canal); small holes such as the transverse foramina (VA,
    ~20-35 mm^2) are kept. Removed voxels are set to -1000."""
    b = v > 600
    it = max(1, int(round(1.5 / sp[0])))
    st = ndimage.generate_binary_structure(3, 1)
    b = ndimage.binary_closing(b, structure=st, iterations=it)
    px = sp[0] * sp[1]
    for k in range(v.shape[2]):
        s = b[:, :, k]
        if not s.any():
            continue
        holes = ndimage.binary_fill_holes(s) & ~s
        if holes.any():
            lab, n = ndimage.label(holes)
            area = np.bincount(lab.ravel()) * px
            big = area > hole_mm2
            big[0] = False
            b[:, :, k] = s | big[lab]
    b = ndimage.binary_dilation(b, iterations=2)
    return np.where(b, -1000, v).astype(np.int16)


def run(case):
    t0 = time.time()
    sid = case.split("_", 1)[1]
    cdir, ddir = os.path.join(IMG, case), os.path.join(DER, case)
    conv = json.load(open(os.path.join(ddir, "convert.json")))
    fetch = json.load(open(os.path.join(ddir, "fetch.json")))
    c = CAND[sid]
    ctimg = nib.load(os.path.join(cdir, "ct.nii.gz"))
    ct = np.asarray(ctimg.dataobj)
    A = ctimg.affine
    sp = np.array(ctimg.header.get_zooms()[:3], float)
    tsimg = nib.load(os.path.join(ddir, "ts_fast_1p5.nii.gz"))
    ts = np.asarray(tsimg.dataobj).astype(np.int16)
    A15 = tsimg.affine
    M = np.linalg.inv(A) @ A15  # 1.5 mm voxel -> native voxel

    def to_native(ijk):
        x = M[:3, :3] @ ijk.T + M[:3, 3:4]
        x = np.round(x).astype(int)
        for d in range(3):
            np.clip(x[d], 0, ct.shape[d] - 1, out=x[d])
        return x

    def zworld(k15):
        return A15[2, 2] * k15 + A15[2, 3]

    z0 = A[2, 3]; z1 = A[2, 3] + A[2, 2] * (ct.shape[2] - 1)
    rec = {"image_z_range_mm": [r1(z0), r1(z1)]}

    # --- vessel HU
    hu = {}
    masks = {}
    for name in VESSELS:
        m = ts == ID[name]
        masks[name] = m
        if m.sum() == 0:
            hu[name] = None
            continue
        core = ndimage.binary_erosion(m)
        if core.sum() < 5:
            core = m
        x = to_native(np.argwhere(core))
        v = ct[x[0], x[1], x[2]].astype(float)
        kk = np.argwhere(m)[:, 2]
        hu[name] = {"median": r1(np.median(v)), "p10": r1(np.percentile(v, 10)), "p90": r1(np.percentile(v, 90)),
                    "n": int(core.sum()), "z_mm": [r1(zworld(kk.min())), r1(zworld(kk.max()))]}
    rec["vessel_hu"] = hu

    def med(n):
        return hu[n]["median"] if hu.get(n) else None
    cca = [med("common_carotid_artery_right"), med("common_carotid_artery_left")]
    cca_v = [x for x in cca if x is not None]
    cca_med = float(np.mean(cca_v)) if cca_v else None
    bcv = {"right": med("brachiocephalic_vein_right"), "left": med("brachiocephalic_vein_left")}
    inj = None; contra = None
    if bcv["right"] is not None and bcv["left"] is not None:
        inj = "right" if bcv["right"] > bcv["left"] else "left"
        contra = bcv["left" if inj == "right" else "right"]
    if cca_med is None:
        phase = "check"
    elif cca_med < 150:
        phase = "non-contrast"
    elif cca_med >= 250 and contra is not None and contra < 200:
        phase = "arterial"
    else:
        phase = "check"
    rec["phase_auto"] = {"cca_mean_of_medians": r1(cca_med), "injection_side_by_bcv": inj,
                         "contralateral_bcv_median": contra, "call": phase,
                         "sca_medians_R_L": [med("subclavian_artery_right"), med("subclavian_artery_left")]}

    # --- coverage
    gaps = {}

    def touch(a, b, tol_mm=10.0):
        """z of the voxel of mask a nearest to mask b, if that gap is <= tol_mm (fast 3 mm masks leave gaps)."""
        if masks[a].sum() == 0 or masks[b].sum() == 0:
            gaps[a + "->" + b] = None
            return None
        dist = ndimage.distance_transform_edt(~masks[b], sampling=np.abs(np.diag(A15)[:3]))
        j = np.argwhere(masks[a])
        d = dist[j[:, 0], j[:, 1], j[:, 2]]
        gaps[a + "->" + b] = r1(d.min())
        if d.min() > tol_mm:
            return None
        return r1(zworld(j[np.argmin(d), 2]))
    lsa_o = touch("subclavian_artery_left", "aorta")
    rsa_o = touch("subclavian_artery_right", "brachiocephalic_trunk")
    rsa_from = "brachiocephalic_trunk"
    if rsa_o is None:
        rsa_o = touch("subclavian_artery_right", "aorta"); rsa_from = "aorta" if rsa_o is not None else None
    bct_o = touch("brachiocephalic_trunk", "aorta")
    verts = {}
    for v in VERT:
        kk = np.argwhere(ts == ID[v])[:, 2]
        verts[v] = None if len(kk) < 30 else [r1(zworld(kk.min())), r1(zworld(kk.max()))]
    sk = np.argwhere(ts == ID["skull"])
    skull = None if len(sk) < 100 else [r1(zworld(sk[:, 2].min())), r1(zworld(sk[:, 2].max()))]
    c1top = verts["vertebrae_C1"][1] if verts["vertebrae_C1"] else None
    cov = {"lsa_origin_z": lsa_o, "rsa_origin_z": rsa_o, "rsa_origin_from": rsa_from, "bct_origin_z": bct_o,
           "mask_gaps_mm": gaps,
           "aorta_top_z": hu["aorta"]["z_mm"][1] if hu.get("aorta") else None, "vertebrae_z": verts,
           "skull_z": skull, "C1_top_z": c1top,
           "image_top_above_C1_top_mm": r1(z1 - c1top) if c1top is not None else None}
    cov["subclavian_origins_in_image"] = bool(lsa_o is not None and rsa_o is not None
                                              and min(lsa_o, rsa_o) - z0 >= 5)
    cov["vertebrae_C1_T1_present"] = all(verts[v] is not None for v in VERT)
    cov["skull_base_in_image"] = bool(skull is not None and c1top is not None and z1 - c1top >= 30
                                      and skull[0] <= c1top + 15)
    rec["coverage"] = cov

    # --- neck stretch, midline
    lo_cands = [x for x in (lsa_o, rsa_o, bct_o, cov["aorta_top_z"]) if x is not None]
    zlo = max(z0, (min(lo_cands) if lo_cands else z0) - 25)
    zhi = min(z1, (c1top + 25) if c1top is not None else z1)
    k_lo = int(round((zlo - z0) / sp[2])); k_hi = int(round((zhi - z0) / sp[2]))
    cord = np.argwhere(ts == ID["spinal_cord"])
    cw = (A15 @ np.c_[cord, np.ones(len(cord))].T).T if len(cord) else None
    if cw is not None:
        sel = (cw[:, 2] >= zlo) & (cw[:, 2] <= zhi)
        xm, ym = (float(np.median(cw[sel, 0])), float(np.median(cw[sel, 1]))) if sel.any() else \
            (float(np.median(cw[:, 0])), float(np.median(cw[:, 1])))
    else:
        xm = A[0, 3] + A[0, 0] * ct.shape[0] / 2; ym = A[1, 3] + A[1, 1] * ct.shape[1] / 2
    rec["neck_stretch_z_mm"] = [r1(zlo), r1(zhi)]
    rec["cord_midline_xy_mm"] = [r1(xm), r1(ym)]

    # --- motion proxy over the neck stretch
    sub = ct[:, :, k_lo:k_hi + 1]
    body = sub > -300
    area = body.sum(axis=(0, 1)).astype(float)
    cx = (body.sum(axis=1) * np.arange(sub.shape[0])[:, None]).sum(axis=0) / np.maximum(area, 1) * sp[0]
    cy = (body.sum(axis=0) * np.arange(sub.shape[1])[:, None]).sum(axis=0) / np.maximum(area, 1) * sp[1]
    jump = np.hypot(np.diff(cx), np.diff(cy))
    darea = np.abs(np.diff(area)) / np.maximum(area[:-1], 1)
    rec["motion_proxy"] = {"max_slice_centroid_jump_mm": r1(jump.max()) if len(jump) else None,
                           "p99_slice_centroid_jump_mm": round(float(np.percentile(jump, 99)), 3) if len(jump) else None,
                           "max_slice_area_change_frac": round(float(darea.max()), 4) if len(darea) else None}
    del body

    # --- MIP figure (neck: arch to skull base)
    def ix(xw):
        return int(np.clip(round((xw - A[0, 3]) / A[0, 0]), 0, ct.shape[0] - 1))

    def iy(yw):
        return int(np.clip(round((yw - A[1, 3]) / A[1, 1]), 0, ct.shape[1] - 1))
    xa, xb = sorted([ix(xm - 75), ix(xm + 75)])
    ya, yb = sorted([iy(ym - 45), iy(ym + 75)])
    sl = bone_suppressed(sub[xa:xb + 1, ya:yb + 1], sp)
    yy = A[1, 3] + A[1, 1] * np.arange(ya, yb + 1)
    xx = A[0, 3] + A[0, 0] * np.arange(xa, xb + 1)
    band_ap = (yy >= ym + 3) & (yy <= ym + 45)
    band_va = (yy >= ym - 8) & (yy <= ym + 22)
    cor_ap = sl[:, band_ap].max(axis=1).T
    cor_va = np.where((np.abs(xx - xm) < 9)[:, None], -1000, sl[:, band_va].max(axis=1)).T  # mask vertebral bodies
    sag_r = sl[(xx >= xm - 32) & (xx <= xm - 8)].max(axis=0).T
    sag_l = sl[(xx >= xm + 8) & (xx <= xm + 32)].max(axis=0).T
    del sl
    full = ct.max(axis=1).T
    ext_full = [A[0, 3], A[0, 3] + A[0, 0] * ct.shape[0], z0, z1]
    ext_cor = [xx[0], xx[-1], zlo, zhi]
    ext_sag = [yy[0], yy[-1], zlo, zhi]
    fig = plt.figure(figsize=(26, 19))
    gs = fig.add_gridspec(2, 20, height_ratios=[1.15, 1])
    ax = [fig.add_subplot(gs[0, 0:4]), fig.add_subplot(gs[0, 4:9]), fig.add_subplot(gs[0, 9:14]),
          fig.add_subplot(gs[0, 14:17]), fig.add_subplot(gs[0, 17:20])]
    axb = [fig.add_subplot(gs[1, 5 * i:5 * i + 5]) for i in range(4)]
    ax_levels = []
    for lab, vname in (("C2", "vertebrae_C2"), ("C4", "vertebrae_C4"), ("C6", "vertebrae_C6"), ("C7", "vertebrae_C7")):
        if verts[vname] is not None:
            ax_levels.append((lab, 0.5 * (verts[vname][0] + verts[vname][1])))
    for a, (lab, zz) in zip(axb, ax_levels):
        k = int(np.clip(round((zz - z0) / sp[2]), 0, ct.shape[2] - 1))
        k0, k1 = max(0, k - int(round(0.5 / sp[2]))), min(ct.shape[2], k + int(round(0.5 / sp[2])) + 1)
        im = ct[xa:xb + 1, ya:yb + 1, k0:k1].max(axis=2).T
        a.imshow(im, cmap="gray", origin="lower", vmin=-100, vmax=900, extent=[xx[0], xx[-1], yy[0], yy[-1]])
        a.set_xlim(xm + 38, xm - 38); a.set_ylim(ym - 20, ym + 38)
        a.set_title("axial %s mid-body, z %.0f mm (1 mm MIP, -100..900 HU)\npatient right on the left, anterior up"
                    % (lab, zz), fontsize=10)
    ax[0].imshow(np.clip(full, -200, 1200), cmap="gray", origin="lower", extent=ext_full)
    ax[0].axhline(zlo, color="y", lw=1); ax[0].axhline(zhi, color="y", lw=1)
    ax[0].set_title("coronal MIP, whole volume\n(-200..1200 HU); yellow = neck stretch", fontsize=10)
    kw = dict(cmap="gray", origin="lower", vmin=300, vmax=650)
    ax[1].imshow(cor_ap, extent=ext_cor, **kw)
    ax[1].set_title("coronal MIP, AP slab cord +3..+45 mm\n300-650 HU, bone removed (display only)", fontsize=10)
    ax[2].imshow(cor_va, extent=ext_cor, **kw)
    ax[2].set_title("coronal MIP, VA corridor: y = cord -8..+22 mm, |x - cord| > 9 mm\n300-650 HU, bone removed",
                    fontsize=10)
    ax[3].imshow(sag_r, extent=ext_sag, **kw)
    ax[3].set_title("sagittal MIP RIGHT, x = cord -32..-8 mm\n300-650 HU, bone removed; anterior right", fontsize=10)
    ax[4].imshow(sag_l, extent=ext_sag, **kw)
    ax[4].set_title("sagittal MIP LEFT, x = cord +8..+32 mm\n300-650 HU, bone removed; anterior right", fontsize=10)
    for a in ax[:3]:
        a.invert_xaxis()  # radiological view: patient right on the image left
        a.set_xlabel("x (mm, RAS; patient right on the left)")
    for a in ax[3:]:
        a.set_xlabel("y (mm, RAS)")
    marks = [("C1", c1top), ("C6", verts["vertebrae_C6"][1] if verts["vertebrae_C6"] else None),
             ("T1", verts["vertebrae_T1"][1] if verts["vertebrae_T1"] else None),
             ("LSA o", lsa_o), ("RSA o", rsa_o)]
    for a in ax[1:]:
        for lab, zz in marks:
            if zz is not None and zlo <= zz <= zhi:
                a.axhline(zz, color="c", lw=0.6, ls=":")
                a.text(a.get_xlim()[0], zz, " " + lab, color="c", fontsize=8, va="bottom")
    hdr = c
    fig.suptitle("%s  (%s %s, IA %s)  %s %s %s  grid %.3fx%.3fx%.2f mm (thickness %.2f)  z %.0f mm  |  CCA %s HU, "
                 "SCA R/L %s/%s HU, contralateral BCV %s HU  ->  phase(auto) %s"
                 % (case, hdr["sex"], hdr["age"], hdr["ia"], hdr["manu"], hdr["model"], hdr["kernel"], sp[0], sp[1],
                    sp[2], hdr["thick"], conv["z_extent_mm"], r1(cca_med), med("subclavian_artery_right"),
                    med("subclavian_artery_left"), contra, phase), fontsize=12)
    plt.tight_layout()
    png = os.path.join(cdir, "mip_neck.png")
    plt.savefig(png, dpi=72); plt.close(fig)

    # --- meta.json
    vis = VIS.get(case, {})
    reasons = []
    if not cov["subclavian_origins_in_image"]:
        reasons.append("subclavian origins not found in the image (auto)")
    if not cov["skull_base_in_image"]:
        reasons.append("skull base not covered (auto)")
    if phase == "non-contrast":
        reasons.append("non-contrast (auto)")
    if conv.get("geometry_problem"):
        reasons.append("slice spacing not uniform / gaps")
    auto_ok = not reasons and phase == "arterial"
    if vis.get("verdict"):
        verdict = vis["verdict"]
    else:
        verdict = "pending visual check" + ("" if auto_ok else " (auto flags: %s)" % ("; ".join(reasons) or phase))
    meta = {
        "case": case, "source": "Head-CTA 172-series set (Blagosklonova et al.), Zenodo record 15697196",
        "record_url": "https://zenodo.org/records/15697196", "doi": "10.5281/zenodo.15697196",
        "paper_doi": "10.3390/data11040074", "licence": "CC BY 4.0 (image); derivatives via Arterial are non-commercial",
        "file_url": fetch["url"], "zip_members": fetch["zip_members"], "zip_byte_range": fetch.get("byte_range"),
        "dataset_case_id": sid, "patient_id": None,
        "patient_id_note": "DICOM PatientID/Name/Sex/Age are blank (anonymised); the dataset CaseID is the only "
                           "identifier. Sex and age from the dataset's Metadata.csv.",
        "sex": c["sex"], "age_years": c["age"], "aneurysm_status": "control" if c["ia"] == "0" else "IA " + str(c["ia"]),
        "aneurysms": c.get("arteries"), "head_neck_flag_csv": c["head_neck"],
        "scanner": {"manufacturer": c["manu"], "model": c["model"], "software": c["sw"], "kernel": c["kernel"],
                    "kvp": c["kvp"], "tube_current_mA": c["ma"], "table_height": c["table"],
                    "reconstruction_diameter_mm": conv["unique_values"]["ReconstructionDiameter"],
                    "transfer_syntax": c["ts"], "patient_position": c["pos"]},
        "dicom": {"n_files": conv["n_files"], "slice_thickness_mm": conv["unique_values"]["SliceThickness"],
                  "pixel_spacing_mm": conv["unique_values"]["PixelSpacing"],
                  "iop": conv["unique_values"]["ImageOrientationPatient"],
                  "gantry_tilt": conv["unique_values"]["GantryDetectorTilt"],
                  "rescale_slope_intercept": conv["unique_values"]["RescaleSlope_Intercept"],
                  "acquisition_number": conv["unique_values"]["AcquisitionNumber"]},
        "acquisition_voxel_mm": [round(conv["unique_values"]["PixelSpacing"][0][1], 4),
                                 round(conv["unique_values"]["PixelSpacing"][0][0], 4),
                                 conv["unique_values"]["SliceThickness"][0]],
        "acquisition_voxel_note": "in-plane pixel x reconstructed slice THICKNESS (the grader's acquisition voxel); "
                                  "the slice INTERVAL (grid z spacing) is smaller when slices overlap",
        "grid_spacing_mm": conv["spacing_mm"], "shape": conv["shape"], "z_extent_mm": conv["z_extent_mm"],
        "orientation": conv["axcodes"], "affine": conv["affine"], "dtype": conv["dtype"],
        "resampled": conv["resampled"], "geometry_checks": conv["geometry"],
        "hu_p1_p50_p999": conv["hu_p1_p50_p999_all_voxels"],
        "hu_p1_p50_p999_voxels_above_m500": conv["hu_p1_p50_p999_above_m500"],
        "padding_value": conv["min_value"], "padding_fraction": conv["frac_at_min_value"],
        "quick_look": rec, "quick_look_method": "TotalSegmentator 2.18.0 total --fast (3 mm model) on a 1.5 mm RAS "
                                                 "copy; native HU sampled at 1-voxel-eroded masks",
        "mip_png": png, "visual_check": vis or None,
        "qc_auto_flags": reasons, "phase_auto": phase, "verdict": verdict,
        "date": time.strftime("%Y-%m-%d"), "qc_seconds": round(time.time() - t0, 1),
    }
    json.dump(meta, open(os.path.join(cdir, "meta.json"), "w"), indent=1)
    json.dump(meta, open(os.path.join(ddir, "qc.json"), "w"), indent=1)
    print(case, "phase", phase, "CCA", r1(cca_med), "SCA", meta["quick_look"]["phase_auto"]["sca_medians_R_L"],
          "contraBCV", contra, "inj", inj, "origins", cov["subclavian_origins_in_image"], lsa_o, rsa_o, rsa_from,
          "skullbase", cov["skull_base_in_image"], "verts", cov["vertebrae_C1_T1_present"],
          "motion", rec["motion_proxy"], "verdict", verdict, "%.0fs" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    for cs in sys.argv[1:]:
        run(cs)
