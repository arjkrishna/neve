#!/usr/bin/env python3
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\pilot_fetch_song_meta2.py (= /sp/v5_va/pilot_fetch_song_meta2.py in the build containers)
"""pilot_fetch_song_meta2.py (pilot_fetch_song, instance 3, 2026-10-01 ~20:45 EDT): instance 1's
pilot_fetch_song_meta.py (never run; kept unchanged) with three changes: the CCA / subclavian HU come from
results/pilot_fetch_song_picks.json (column-frame rules + by-eye overrides; qc_pass1's own auto picks were wrong or
missing on several sides), the column-frame VA check and side-file integrity check (<case>/qc_va_check.json) are
recorded, and the summary carries the selection rule, pool counts and known issues from the review file.
Assembles <img>/<case>/meta.json for every fetched Song case from
  - results/pilot_fetch_song_index.json   (archive walk: member, offsets, CRC32, NIfTI header)
  - results/pilot_fetch_song_table.json   (clinical_all.csv sex / age / aneurysm; sidecar kernel / scanner)
  - <case>/fetch.json                      (range-read record, CRC32 check, md5)
  - <case>/qc_pass1.json                   (HU statistics, clipping, aorta tracking, candidate blobs)
  - <case>/qc_va_check.json                (vertebral-column track, side-file integrity check)
  - results/pilot_fetch_song_picks.json    (CCA / SCA picks per side and level)
  - results/pilot_fetch_song_review.json   (the by-eye review: verdict, phase, VA visibility, FOV, motion, notes)
and write the pilot summary results/pilot_fetch_song.json.
usage: pilot_fetch_song_meta2.py CASE [CASE ...]   (neve-build-stagea)
"""
import sys, os, json, time, zlib

IMG = "/repo/carotid_data/va_sources/thin_slice/song"
RES = "/sp/v5_va/results"
REC = dict(dataset="CMHA: Intracranial Aneurysm CTA Image & 3D Model Dataset with Clinical, Morphological, "
                   "Hemodynamic Data (Song et al. 2024)",
           record="figshare 26965450 v1", doi="10.6084/m9.figshare.26965450.v1",
           url="https://doi.org/10.6084/m9.figshare.26965450",
           article="Song M. et al., Sci Data 2024, doi:10.1038/s41597-024-04056-8 (article CC BY-NC-ND)",
           licence="CC BY 4.0 (figshare record); anything derived through Arterial is non-commercial "
                   "(CC BY-NC weights, PolyForm-NC code)")
PICK_RULES = ("dx/dy in the vertebral-column frame of qc_va_check.json. CCA at L3 (arch+60; L4 = arch+100 as the "
              "fallback/second level): circularity >= 0.8, area 20-100 mm2, median HU 0.8-1.6 x aorta, not touching bone, "
              "dy +3..+70 mm, |dx| 8-60 mm, the largest per side. SCA at L1 (arch+15; L2 = arch+35 fallback): circularity "
              ">= 0.45, area 30-250 mm2, median HU 0.75-1.6 x aorta, not touching bone, |dx| 5-90 mm, dy >= -15 mm, the "
              "most posterior per side. Median HU of the 1-voxel-eroded blob (qc_pass1). Every pick checked by eye on "
              "qc_axial.png; overrides are listed in pilot_fetch_song_picks.py OVERRIDES.")


def picks(case, pj):
    pk = pj[case]["picks"]
    out = {}
    for kind, lv in (("CCA", ("L3", "L4")), ("SCA", ("L1", "L2"))):
        res = {}
        for side in ("R", "L"):
            allv = {l: pk.get("%s_%s_%s" % (kind, l, side)) for l in lv}
            prim = next(((l, v) for l, v in allv.items() if v), (None, None))
            if prim[1] is None:
                res[side] = dict(hu_median=None, method="none identifiable at %s" % "/".join(lv), all_levels=allv)
            else:
                v = prim[1]
                res[side] = dict(hu_median=v["hu_med"], level=prim[0], blob_id=v["id"], area_mm2=v["area_mm2"],
                                 method=("by-eye override: " + v["override"]) if v.get("override") else
                                 ("column-frame rule, confirmed on qc_axial.png by eye" if kind == "CCA" else
                                  "column-frame rule; arterial reference at the subclavian level -- identity (SCA vs "
                                  "brachiocephalic / CCA / VA origin) not verified"), all_levels=allv)
        out[kind] = res
    return out


def main():
    idx = json.load(open(os.path.join(RES, "pilot_fetch_song_index.json")))
    table = json.load(open(os.path.join(RES, "pilot_fetch_song_table.json")))
    review = json.load(open(os.path.join(RES, "pilot_fetch_song_review.json")))
    pj = json.load(open(os.path.join(RES, "pilot_fetch_song_picks.json")))
    summary = dict(task="pilot_fetch_song", date=time.strftime("%Y-%m-%d %H:%M"), source=REC, cases={},
                   selection_rule=review.get("_selection"), pool=review.get("_pool"),
                   rejections=review.get("_rejections", []), known_issues=review.get("_known_issues"),
                   acceptance_rules=review.get("_acceptance_rules"))
    for case in sys.argv[1:]:
        d = os.path.join(IMG, case)
        fe = json.load(open(os.path.join(d, "fetch.json")))
        qc = json.load(open(os.path.join(d, "qc_pass1.json")))
        tb = table[case]; rv = review.get(case, {})
        sc_path = os.path.join(IMG, "_source_meta", "sidecars", case + ".json")
        sc = json.load(open(sc_path))
        nii = idx["cases"][case]["nii"]
        hu = qc["hu"]; ao = qc["aorta_auto"]
        pk = picks(case, pj)
        vc = json.load(open(os.path.join(d, "qc_va_check.json")))
        vcol = {k: v for k, v in vc["column"].items() if k != "track_every_5mm"}
        meta = dict(
            case=case, group="control" if tb["group"] == "controls" else "aneurysm patient",
            source=dict(REC, archive_file=tb["group"] + ".rar", archive_url=fe["url"], member=fe["member"],
                        member_data_offset=fe["data_offset"], member_bytes=fe["size"], member_stored="RAR5 method 0 "
                        "(stored): ct.nii.gz is the shipped .nii.gz byte for byte", crc32_rar5=fe["crc32_expected"],
                        crc32_ok=fe["crc_ok"], md5=fe["md5"], fetched=fe["date"], fetched_how=fe["source"],
                        sidecar=dict(member=fe["member"].replace(".nii.gz", ".json"), path="_source_meta/sidecars/%s.json" % case)),
            patient_id=None,
            patient_id_note="no PatientID in the release: the dcm2niix sidecar carries no patient tags; the case id "
                            "is the dataset's own (AHMU1218001-099 patients, 100-143 controls)",
            sex=tb["sex"], age=tb["age"],
            clinical=dict(source="statistical results.rar / clinical_all.csv", aneurysm_locations=tb["aneurysm_locations"]),
            image=dict(path="carotid_data/va_sources/thin_slice/song/%s/ct.nii.gz" % case, kept_as_shipped=True,
                       shape=qc["header"]["shape"], spacing_mm=qc["header"]["zooms"],
                       acquisition_voxel_mm=qc["header"]["zooms"],
                       acquisition_voxel_note="native grid as shipped (dcm2niix); grade on this voxel, never on "
                                              "Arterial's resampled grid",
                       slice_class_mm=tb["slice_class"], dtype_on_disk=qc["header"]["on_disk_dtype"],
                       scl_slope=qc["header"]["scl_slope"], scl_inter=qc["header"]["scl_inter"],
                       qform_code=qc["header"]["qform_code"], sform_code=qc["header"]["sform_code"],
                       affine=qc["header"]["affine"], orientation=qc["header"]["axcodes"],
                       orientation_note="voxel axes run to patient Left, Anterior, Superior (LAS); sform only "
                                        "(qform_code 0); head-first supine (sidecar PatientPosition %s)" % sc.get("PatientPosition"),
                       z_extent_mm=qc["ras"]["z_extent_mm"], z_range_ras_mm=qc["ras"]["z_range_mm"],
                       fov_mm=tb["fov_mm"], world_bbox_ras_mm=nii.get("world_bbox_mm")),
            hu=dict(p1=hu["p"]["p1"], p50=hu["p"]["p50"], p99_9=hu["p"]["p99.9"], percentiles=hu["p"],
                    min=hu["min"], max=hu["max"], frac_eq0=hu["frac_eq0"], frac_negzero=hu["frac_negzero"],
                    frac_poszero=hu["frac_poszero"], min_nonzero=hu["min_nonzero"], nonzero_percentiles=hu["nonzero_p"],
                    clipped_at_0=hu["clipped_at_0"], integer_valued=hu["integer_valued"],
                    clipping_note=("every voxel <= 50 HU is stored as 0 (no value in 1..50; the smallest non-zero "
                                   "value is %s). The sign of the zero survives: -0.0 = originally negative (air, fat, "
                                   "lung), +0.0 = originally 0..50 HU (soft tissue, CSF)" % hu["min_nonzero"])
                    if hu["clipped_at_0"] else "not clipped",
                    in_body=qc["body"], histogram=hu["hist"], histogram_nonzero_0_200=hu["hist_0_200_nonzero"],
                    sampling="percentiles on every 7th voxel, histogram on every 3rd; min / max / fractions on all"),
            contrast=dict(aorta_hu_median_auto=ao.get("aorta_hu_med"), CCA=pk["CCA"], subclavian=pk["SCA"],
                          pick_rules=PICK_RULES, phase=rv.get("phase"), venous=rv.get("venous")),
            landmarks=dict(bottom_slice_z_ras=ao["bottom_z"], aorta_limbs_in_bottom_slice=len(ao["bottom_limbs"]),
                           bottom_limbs=ao["bottom_limbs"], two_limbs_at_bottom=ao["two_limbs_at_bottom"],
                           arch_top_z_ras=ao.get("arch_top_z"), arch_top_above_bottom_mm=ao.get("arch_top_above_bottom_mm"),
                           neck_min_area_z_ras=qc["levels"]["neck_min_area_z"],
                           neck_min_area_cm2=qc["levels"]["neck_min_area_cm2"],
                           note="automatic; arch top = top of the aorta tracked up from the bottom slice"),
            scanner=dict(family=tb["scanner_family"], basis=tb["scanner_basis"], kernel=sc.get("ConvolutionKernel"),
                         software=sc.get("SoftwareVersions"), protocol=sc.get("ProtocolName"),
                         series_description=sc.get("SeriesDescription"), tube_current_mA=sc.get("XRayTubeCurrent"),
                         exposure_time_s=sc.get("ExposureTime"), device_serial=sc.get("DeviceSerialNumber"),
                         scan_options=sc.get("ScanOptions"), converter="%s %s" % (sc.get("ConversionSoftware"), sc.get("ConversionSoftwareVersion"))),
            visual_qc=dict({k: v for k, v in rv.items() if k != "picks"}, reviewed_on=["mip_neck_coronal.png",
                           "mip_neck_sagittal.png", "qc_axial.png", "qc_va_check.png", "qc_va_zoom.png"]),
            va_check=dict(script=vc.get("script"), column=vcol, column_seed=vc.get("column_seed"),
                          band_brightness_indicative=vc.get("band_brightness_indicative"), side_file_check=vc["side_check"]),
            files=dict(ct="ct.nii.gz", mip_coronal="mip_neck_coronal.png", mip_sagittal="mip_neck_sagittal.png",
                       qc_axial="qc_axial.png", qc_va_axial="qc_va_axial.png (superseded: crops centred on the mandible above arch+90)",
                       qc_va_check="qc_va_check.png", qc_va_check_json="qc_va_check.json", qc_va_zoom="qc_va_zoom.png",
                       qc_pass1="qc_pass1.json", fetch="fetch.json",
                       side_files=qc.get("side_files"),
                       side_file_use="only if Arterial fails on the shipped ct.nii.gz (run that first)"),
            mip=qc["mip"],
            date=time.strftime("%Y-%m-%d %H:%M"), written_by="pilot_fetch_song (pilot_fetch_song_meta2.py)")
        json.dump(meta, open(os.path.join(d, "meta.json"), "w"), indent=1)
        summary["cases"][case] = dict(
            group=meta["group"], sex=meta["sex"], age=meta["age"], slice_class_mm=tb["slice_class"],
            spacing_mm=meta["image"]["spacing_mm"], shape=meta["image"]["shape"], z_extent_mm=meta["image"]["z_extent_mm"],
            fov_mm=tb["fov_mm"], scanner=tb["scanner_family"], kernel=sc.get("ConvolutionKernel"),
            hu_p1_p50_p999=[hu["p"]["p1"], hu["p"]["p50"], hu["p"]["p99.9"]], min_nonzero=hu["min_nonzero"],
            frac_eq0=hu["frac_eq0"], aorta_hu=ao.get("aorta_hu_med"),
            cca_hu={s: pk["CCA"][s].get("hu_median") for s in "RL"}, sca_hu={s: pk["SCA"][s].get("hu_median") for s in "RL"},
            two_limbs_at_bottom=ao["two_limbs_at_bottom"], arch_top_above_bottom_mm=ao.get("arch_top_above_bottom_mm"),
            verdict=rv.get("verdict"), phase=rv.get("phase"), va_visible=rv.get("va_visible"),
            venous=rv.get("venous"), notes=rv.get("notes"), crc32_ok=fe["crc_ok"], md5=fe["md5"],
            side_files_ok=vc["side_check"]["ok"], image_top_above_arch_mm=vcol.get("image_top_above_arch_mm"),
            meta="carotid_data/va_sources/thin_slice/song/%s/meta.json" % case,
            side_bg_minus100=(qc.get("side_files") or {}).get("ct_bg_minus100", {}).get("path"),
            side_negzero_minus100=(qc.get("side_files") or {}).get("ct_negzero_minus100", {}).get("path"))
        print(case, meta["sex"], meta["age"], tb["slice_class"], "CCA", summary["cases"][case]["cca_hu"],
              "SCA", summary["cases"][case]["sca_hu"], rv.get("verdict"))
    acc = [c for c, v in summary["cases"].items() if v["verdict"] == "accept"]
    summary["accepted"] = acc
    summary["n_accepted"] = len(acc)
    json.dump(summary, open(os.path.join(RES, "pilot_fetch_song.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
