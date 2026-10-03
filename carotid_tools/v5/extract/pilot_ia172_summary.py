# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\pilot_ia172_summary.py (= /sp/v5_va/pilot_ia172_summary.py in the build containers)
"""pilot_ia172_summary: collect the fetch / convert / QC records of the 172-series pilot into
/sp/v5_va/results/pilot_ia172_fetch.json and print a table.  usage: python3 pilot_ia172_summary.py <case> ..."""
import json
import os
import sys
import time

IMG = "/repo/carotid_data/va_sources/thin_slice/ia172"
DER = "/repo/carotid_data/va_sources/derived/thin_slice/ia172"
cand = json.load(open("/sp/v5_va/results/pilot_ia172_candidates.json"))
C, dup = cand["candidates"], cand["duplicate_screen"]
VIS_P = "/sp/v5_va/results/pilot_ia172_visual.json"
VIS = json.load(open(VIS_P)) if os.path.exists(VIS_P) else {}


def du(p):
    t = 0
    for r, _, fs in os.walk(p):
        t += sum(os.path.getsize(os.path.join(r, f)) for f in fs)
    return round(t / 1e6, 1)


hn = [s for s, c in C.items() if c["head_neck"] == "1"]
hn300 = [s for s in hn if (C[s]["z"] or 0) >= 300]
thin = [s for s in hn300 if (C[s]["thick"] or 9) <= 0.5]
thin_ovl = [s for s in thin if (C[s]["dz"] or 9) < 0.45]
pool = {"series_in_zip": len(C), "head_neck_flag_1": len(hn), "head_neck_z_ge_300mm": len(hn300),
        "head_neck_z300_thickness_le_0p5": len(thin), "of_which_overlapping_interval_lt_0p45": len(thin_ovl),
        "thickness_counts_head_neck_z300": {}}
for s in hn300:
    k = str(C[s]["thick"])
    pool["thickness_counts_head_neck_z300"][k] = pool["thickness_counts_head_neck_z300"].get(k, 0) + 1

def arch_limbs(case):
    """aorta components in the bottom slices of the quick-look mask (2 = the bottom slice cuts the arch into
    ascending + descending limbs, which Arterial's start point needs)."""
    import nibabel as nib
    import numpy as np
    from scipy import ndimage
    ts = np.asarray(nib.load(os.path.join(DER, case, "ts_fast_1p5.nii.gz")).dataobj)
    a = ts == 52  # aorta in the "total" class map
    ks = np.where(a.any(axis=(0, 1)))[0]
    if len(ks) == 0:
        return {"aorta_in_image": False}
    res = {"aorta_in_image": True, "aorta_lowest_k_1p5": int(ks.min())}
    for k in (0, 2, 4):
        res["components_at_k%d" % k] = int(ndimage.label(a[:, :, k])[1])
    return res


SC_P = "/sp/v5_va/results/pilot_ia172_slicecheck.json"
SC = json.load(open(SC_P)) if os.path.exists(SC_P) else {}


def slice_summary(case, m):
    """slice-check spikes and odd-padding slices in world z, and which of them fall in the neck stretch."""
    s = SC.get(case)
    if not s:
        return None
    a = m["affine"]
    z = lambda k: round(a[2][3] + a[2][2] * k, 1)  # noqa: E731
    lo, hi = m["quick_look"]["neck_stretch_z_mm"]
    c1 = m["quick_look"]["coverage"]["C1_top_z"]
    odd = s["slices_with_other_min"]; sp = s["spike_pairs_gt_3x_median"]
    return {"padding_value": s["padding_mode"], "outside_circle_frac": s["outside_circle_frac_median"],
            "sub_padding_slices_z_mm": [z(min(odd)), z(max(odd))] if odd else None,
            "sub_padding_min_HU": min(s["other_min_values"]) if s["other_min_values"] else None,
            "sub_padding_rel_C1_top_mm": [round(z(min(odd)) - c1, 1), round(z(max(odd)) - c1, 1)] if odd and c1 else None,
            "neighbour_absdiff_median_HU": s["neighbour_absdiff_median_HU"],
            "neighbour_absdiff_max_HU": s["neighbour_absdiff_max_HU"],
            "spike_pairs_z_mm": [z(min(sp)), z(max(sp))] if sp else None, "n_spike_pairs_listed": len(sp),
            "spike_pairs_in_neck_stretch": [z(k) for k in sp if lo <= z(k) <= hi]}


ALTERNATES = {"ia172_190", "ia172_015"}
AC_P = "/sp/v5_va/results/pilot_ia172_archcut.json"
AC = json.load(open(AC_P)) if os.path.exists(AC_P) else {}
cases = []
for case in sys.argv[1:]:
    m = json.load(open(os.path.join(IMG, case, "meta.json")))
    f = json.load(open(os.path.join(DER, case, "fetch.json")))
    cv = json.load(open(os.path.join(DER, case, "convert.json")))
    q = m["quick_look"]
    hu = q["vessel_hu"]

    def med(n):
        return hu[n]["median"] if hu.get(n) else None
    cases.append({
        "case": case, "role": "alternate" if case in ALTERNATES else "primary",
        "dataset_case_id": m["dataset_case_id"], "sex": m["sex"], "age": m["age_years"],
        "status": m["aneurysm_status"], "aneurysms": m["aneurysms"],
        "scanner": "%s %s %s" % (m["scanner"]["manufacturer"], m["scanner"]["model"], m["scanner"]["software"]),
        "kernel": m["scanner"]["kernel"], "kvp": m["scanner"]["kvp"], "mA": m["scanner"]["tube_current_mA"],
        "recon_diameter_mm": m["scanner"]["reconstruction_diameter_mm"],
        "acquisition_voxel_mm": m["acquisition_voxel_mm"], "grid_spacing_mm": m["grid_spacing_mm"],
        "shape": m["shape"], "z_extent_mm": m["z_extent_mm"], "orientation": m["orientation"],
        "resampled": m["resampled"], "uniform_spacing": m["geometry_checks"]["uniform"],
        "gaps": m["geometry_checks"]["n_gaps"], "duplicates": m["geometry_checks"]["n_duplicates"],
        "hu_p1_p50_p999": m["hu_p1_p50_p999"], "hu_p1_p50_p999_above_m500": m["hu_p1_p50_p999_voxels_above_m500"],
        "padding_value": m["padding_value"], "padding_fraction": m["padding_fraction"],
        "cca_R_L": [med("common_carotid_artery_right"), med("common_carotid_artery_left")],
        "sca_R_L": [med("subclavian_artery_right"), med("subclavian_artery_left")],
        "aorta": med("aorta"), "bcv_R_L": [med("brachiocephalic_vein_right"), med("brachiocephalic_vein_left")],
        "svc": med("superior_vena_cava"), "injection_side": q["phase_auto"]["injection_side_by_bcv"],
        "phase_auto": q["phase_auto"]["call"], "coverage": {k: q["coverage"][k] for k in (
            "subclavian_origins_in_image", "lsa_origin_z", "rsa_origin_z", "rsa_origin_from", "aorta_top_z",
            "skull_base_in_image", "C1_top_z", "image_top_above_C1_top_mm", "vertebrae_C1_T1_present")},
        "origin_above_bottom_mm": round(min(q["coverage"]["lsa_origin_z"], q["coverage"]["rsa_origin_z"])
                                        - q["image_z_range_mm"][0], 1)
        if q["coverage"]["lsa_origin_z"] is not None and q["coverage"]["rsa_origin_z"] is not None else None,
        "image_z_range_mm": q["image_z_range_mm"], "motion_proxy": q["motion_proxy"],
        "arch_at_bottom_slice": arch_limbs(case),
        "arch_cut_native": {k: AC.get(case, {}).get(k) for k in ("n_aortic_components_bottom", "bottom_two_largest_centre_dist_mm", "0mm", "2mm", "5mm")},
        "slice_check": slice_summary(case, m),
        "visual": VIS.get(case), "verdict": m["verdict"],
        "fetch": {"seconds": f.get("seconds"), "MB_per_s": f.get("MB_per_s"), "span_MB": f.get("span_MB"),
                  "byte_range": f.get("byte_range"), "copied_from": f.get("copied_from")},
        "convert_seconds": cv["seconds"],
        "disk_MB": {"dicom": du(os.path.join(IMG, case, "dicom")),
                    "ct_nii_gz": round(os.path.getsize(os.path.join(IMG, case, "ct.nii.gz")) / 1e6, 1),
                    "derived": du(os.path.join(DER, case))},
        "paths": {"ct": os.path.join(IMG, case, "ct.nii.gz"), "meta": os.path.join(IMG, case, "meta.json"),
                  "mip": os.path.join(IMG, case, "mip_neck.png"), "dicom": os.path.join(IMG, case, "dicom")},
    })

rej_p = "/sp/v5_va/results/pilot_ia172_rejections.json"
out = {"task": "pilot_fetch_ia172", "date": time.strftime("%Y-%m-%d"),
       "source": {"name": "Head-CTA 172-series set (Blagosklonova et al. 2025)", "record": "https://zenodo.org/records/15697196",
                  "doi": "10.5281/zenodo.15697196", "paper_doi": "10.3390/data11040074", "licence": "CC BY 4.0",
                  "file": "Dataset.zip, 25,578,845,008 bytes, md5 264ff9ee868c022d108b7c7aa7396d32"},
       "pool": pool, "duplicate_screen": dup,
       "selected": cases, "rejections": json.load(open(rej_p)) if os.path.exists(rej_p) else [],
       "n_primary_accepted": sum(1 for c in cases if c["role"] == "primary" and str(c["verdict"]).startswith("accept")),
       "n_alternates": sum(1 for c in cases if c["role"] == "alternate"),
       "flags": {
           "arch_apex_only_at_bottom_slice": {"cases": ["ia172_090", "ia172_118"],
                                              "meaning": "one aortic component at the bottom slice (native check); subclavian origins in view (LSA 13.5 / 10.5 mm above the bottom) but Arterial's two-limb arch start may fail",
                                              "alternates": ["ia172_190", "ia172_015"]},
           "forward_head_posture": ["ia172_073"], "left_VA_origin_near_LSA_root_check_arch_origin": ["ia172_141"],
           "weakest_enhancement": ["ia172_039"],
           "all_cases": "padding -2048 HU outside the reconstruction circle (21-31% of voxels); defaced; dental-metal streaks 14-45 mm below the C1 top stay in the oral cavity; grade with acquisition_voxel_mm (thickness 0.5 mm), not the 0.25-0.4 mm grid interval"},
       "reserves_not_fetched": ["062 F53", "155 M42", "160 M42", "067 M49", "060 M52", "119 M60", "052 F78", "131 F93"],
       "notes": "/sp/v5_va/notes/pilot_fetch_ia172.md",
       "provenance": "D:/Arjun/neve/carotid_data/va_sources/thin_slice/ia172/PROVENANCE.md"}
json.dump(out, open("/sp/v5_va/results/pilot_ia172_fetch.json", "w"), indent=1)
print(json.dumps(pool))
for c in cases:
    print(c["case"], c["sex"], c["age"], c["status"], c["acquisition_voxel_mm"], c["grid_spacing_mm"], c["shape"],
          c["z_extent_mm"], "CCA", c["cca_R_L"], "SCA", c["sca_R_L"], "BCV", c["bcv_R_L"], c["phase_auto"],
          "orig>bot", c["origin_above_bottom_mm"], "top>C1", c["coverage"]["image_top_above_C1_top_mm"],
          "HU", c["hu_p1_p50_p999"], c["hu_p1_p50_p999_above_m500"], "disk", c["disk_MB"], c["verdict"])

# markdown rows for PROVENANCE.md and the notes
print("\n| case | CaseID | sex/age | status | scanner, software | kernel | acq. voxel (mm) | grid (mm) | shape | z (mm) | zip byte range |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for c in cases:
    av = c["acquisition_voxel_mm"]; g = c["grid_spacing_mm"]
    print("| %s | %s | %s/%s | %s | %s | %s | %.3f x %.3f x %.2f | %.3f x %.3f x %.2f | %s | %.0f | %s-%s |" % (
        c["case"], c["dataset_case_id"], c["sex"], c["age"], c["status"], c["scanner"], c["kernel"], av[0], av[1],
        av[2], g[0], g[1], g[2], "x".join(map(str, c["shape"])), c["z_extent_mm"], c["fetch"]["byte_range"][0],
        c["fetch"]["byte_range"][1]))
print("\n| case | HU p1/p50/p99.9 (all) | HU p1/p50/p99.9 (> -500) | CCA R/L | SCA R/L | aorta | BCV R/L | inj. | phase | LSA/RSA origin above bottom (mm) | top above C1 (mm) | arch comps at bottom | motion max/p99 (mm) | verdict |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for c in cases:
    cv = c["coverage"]; z0 = c["image_z_range_mm"][0]
    o = "%.0f/%.0f" % (cv["lsa_origin_z"] - z0, cv["rsa_origin_z"] - z0) if cv["lsa_origin_z"] is not None and cv["rsa_origin_z"] is not None else "-"
    ab = c["arch_at_bottom_slice"]
    print("| %s | %s | %s | %s/%s | %s/%s | %s | %s/%s | %s | %s | %s | %s | %s | %s/%s | %s |" % (
        c["case"], "/".join("%.0f" % x for x in c["hu_p1_p50_p999"]), "/".join("%.0f" % x for x in c["hu_p1_p50_p999_above_m500"]),
        c["cca_R_L"][0], c["cca_R_L"][1], c["sca_R_L"][0], c["sca_R_L"][1], c["aorta"], c["bcv_R_L"][0], c["bcv_R_L"][1],
        c["injection_side"], c["phase_auto"], o, cv["image_top_above_C1_top_mm"], ab.get("components_at_k0"),
        c["motion_proxy"]["max_slice_centroid_jump_mm"], c["motion_proxy"]["p99_slice_centroid_jump_mm"], c["verdict"]))
