#!/usr/bin/env python3
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\pilot_fetch_song_meta.py (= /sp/v5_va/pilot_fetch_song_meta.py in the build containers)
"""pilot_fetch_song_meta.py (pass 2): assemble <img>/<case>/meta.json for every fetched Song case from
  - results/pilot_fetch_song_index.json   (archive walk: member, offsets, CRC32, NIfTI header)
  - results/pilot_fetch_song_table.json   (clinical_all.csv sex / age / aneurysm; sidecar kernel / scanner)
  - <case>/fetch.json                      (range-read record, CRC32 check, md5)
  - <case>/qc_pass1.json                   (HU statistics, clipping, aorta tracking, candidate blobs, auto picks)
  - results/pilot_fetch_song_review.json   (the by-eye review: verdict, phase, VA visibility, FOV, motion, and
                                            any CCA / SCA pick overrides by blob id)
and write the pilot summary results/pilot_fetch_song.json.
usage: pilot_fetch_song_meta.py CASE [CASE ...]
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


def blob(qc, level, bid):
    for b in qc["candidates"][level]["blobs"]:
        if b["id"] == bid:
            return b
    raise KeyError("%s blob %s" % (level, bid))


def picks(qc, rv):
    out = {}
    for kind in ("CCA", "SCA"):
        ov = (rv.get("picks") or {}).get(kind)
        auto = qc["auto_picks"].get(kind, {})
        res = {}
        for side in ("R", "L"):
            if ov and side in ov and ov[side] is not None:
                lvl = ov.get("level_" + side, ov.get("level"))
                b = blob(qc, lvl, ov[side])
                res[side] = dict(hu_median=b["hu_med"], hu_p90=b["hu_p90"], level=lvl, z_mm=qc["candidates"][lvl]["z"],
                                 blob_id=b["id"], area_mm2=b["area_mm2"], xy_ras_mm=[b["x"], b["y"]], method="by eye")
            elif ov and side in ov and ov[side] is None:
                res[side] = dict(hu_median=None, method="by eye: not identifiable on the pick levels")
            elif side in auto:
                a = auto[side]; b = blob(qc, a["level"], a["id"])
                res[side] = dict(hu_median=b["hu_med"], hu_p90=b["hu_p90"], level=a["level"],
                                 z_mm=qc["candidates"][a["level"]]["z"], blob_id=b["id"], area_mm2=b["area_mm2"],
                                 xy_ras_mm=[b["x"], b["y"]], method="automatic, confirmed by eye")
            else:
                hi = qc["auto_picks"].get(kind + "_hi", {})
                if side in hi:
                    a = hi[side]; b = blob(qc, a["level"], a["id"])
                    res[side] = dict(hu_median=b["hu_med"], hu_p90=b["hu_p90"], level=a["level"],
                                     z_mm=qc["candidates"][a["level"]]["z"], blob_id=b["id"], area_mm2=b["area_mm2"],
                                     xy_ras_mm=[b["x"], b["y"]], method="automatic (upper level), confirmed by eye")
                else:
                    res[side] = dict(hu_median=None, method="no automatic pick")
        out[kind] = res
    return out


def main():
    idx = json.load(open(os.path.join(RES, "pilot_fetch_song_index.json")))
    table = json.load(open(os.path.join(RES, "pilot_fetch_song_table.json")))
    review = json.load(open(os.path.join(RES, "pilot_fetch_song_review.json")))
    summary = dict(task="pilot_fetch_song", date=time.strftime("%Y-%m-%d %H:%M"), source=REC, cases={},
                   selection_rule=review.get("_selection"), rejections=review.get("_rejections", []))
    for case in sys.argv[1:]:
        d = os.path.join(IMG, case)
        fe = json.load(open(os.path.join(d, "fetch.json")))
        qc = json.load(open(os.path.join(d, "qc_pass1.json")))
        tb = table[case]; rv = review.get(case, {})
        sc_path = os.path.join(IMG, "_source_meta", "sidecars", case + ".json")
        sc = json.load(open(sc_path))
        nii = idx["cases"][case]["nii"]
        hu = qc["hu"]; ao = qc["aorta_auto"]
        pk = picks(qc, rv)
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
                          pick_rules=qc["auto_picks"]["rules"], phase=rv.get("phase")),
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
                           "mip_neck_sagittal.png", "qc_axial.png"]),
            files=dict(ct="ct.nii.gz", mip_coronal="mip_neck_coronal.png", mip_sagittal="mip_neck_sagittal.png",
                       qc_axial="qc_axial.png", qc_pass1="qc_pass1.json", fetch="fetch.json",
                       side_files=qc.get("side_files"),
                       side_file_use="only if Arterial fails on the shipped ct.nii.gz (run that first)"),
            mip=qc["mip"],
            date=time.strftime("%Y-%m-%d %H:%M"), written_by="pilot_fetch_song (pilot_fetch_song_meta.py)")
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
            notes=rv.get("notes"), crc32_ok=fe["crc_ok"], md5=fe["md5"],
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
