#!/usr/bin/env python3
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\v5_eligible_final.py (= /sp/v5_va/v5_eligible_final.py in the build containers)
"""v5_eligible_final.py -- (v5_eye172, 2026-10-02) the FINAL eligible v5 VA donor list, from the re-run tally.

Reads (read-only): results/scaleup_tally.json (scaleup_tally.py, re-run after the v5_eye172 by-eye verdicts),
results/scaleup_eye_rejections.json, results/pilot_donor_ranking.json (VMR), per case meta.json (licence, scanner,
acquisition voxel) and the wall files on disk. Writes results/v5_eligible_final.json.

Classes (V5_BUILD_LOG 14:25 decision, confirmed by eye by v5_eye172): RESERVE = ia172_172 (L VA of aortic-arch origin;
the trace's SCA junction is a ~2 mm tissue link) and Song AHMU1218138 (smoother than the host mesh on both sides; R real
wall only origin -> C2); used only when a TopBrain patient has no other gate-passing donor, in that order. Every other
eligible pair is PRIMARY. Side use (follow rule): native anatomies take the donor's R on the RVA and L on the LVA;
mirrored ("_L") anatomies take the donor's L mirrored on the RVA and its R on the LVA. eve-training-fixed."""
import hashlib
import json
import os
import time

SP = "/sp/v5_va"
RES = SP + "/results"
REPO = "/repo"
DERT = REPO + "/carotid_data/va_sources/derived/thin_slice"
SRCT = REPO + "/carotid_data/va_sources/thin_slice"
HOST_DENSE = 0.027
VPRE = {"R": "rva", "L": "lva"}
RESERVES = {
    ("ia172", "ia172_172"): {
        "reserve_order": 1,
        "why": "L VA arises from the AORTIC ARCH between the LCCA and the LSA (z ~ -731); the trace's subclavian junction is "
               "a ~2 mm link through tissue from the adjacent LSA at z -715 (native HU 297, 154, 79, 125, 218 over s 0-2 mm; "
               "verify_scaleup: no enhanced SCA->VA link within 6 mm); it passes QC only through the provisional V1 <= 3 mm "
               "limit. From s ~3 mm the L trace is a genuine VA (V1 behind the CCA, C6..C1 foramina, V3 loop, HU/CCA 1.0); "
               "R side clean (bridge only at the basilar confluence).",
        "condition_if_used": "advance the L ostium >= 3 mm past the tissue link (graft_v5_reg ostium-advance repair) and flag "
                             "'donor L VA aortic-arch origin; subclavian junction synthetic'",
        "evidence": ["results/v5_eye172/origin_ia172_172_L.png", "results/v5_eye172/origin_ia172_172_L_low.png",
                     "results/v5_eye172/look_ia172_172_L.png", "notes/verify_scaleup.md item 6"]},
    ("song", "AHMU1218138"): {
        "reserve_order": 2,
        "why": "not coarse: all-bins donor-stretch union 0.0154 R / 0.0065 L and contact-free 0.0146 / 0.0078, all below the "
               "dense host RVA mesh (0.027) -- no relief gain over v4; R wall stops at C2 (s ~131 of 209 mm) at a 3 mm "
               "low-HU gap (221 HU = 0.28 x CCA) and the R trace crosses the V3 loop above C1 by a 9.5 mm straight bridge "
               "through tissue (10 HU). Both traces are genuine VAs origin -> C1 (C6..C1 foramina; L V3 loop and wall complete).",
        "condition_if_used": "last resort (after ia172_172); R real wall covers origin -> C2 only: a V3 splice above C2 needs "
                             "the tube (trace) above C2, flag 'R wall short of C1'; L usable as is",
        "evidence": ["results/v5_eye172/look_AHMU1218138_R.png", "results/v5_eye172/look_AHMU1218138_L.png",
                     "derived/thin_slice/song/AHMU1218138/qc/AHMU1218138_wall3d_{R,L}.png", "notes/verify_scaleup.md item 3"]},
}
LIC_DEFAULT = {
    "ia172": "Head-CTA 172-series (Blagosklonova et al., Zenodo 15697196, doi 10.5281/zenodo.15697196; paper doi "
             "10.3390/data11040074), CC BY 4.0 image -- attribution required",
    "song": "CMHA intracranial aneurysm CTA dataset (Song et al. 2024, figshare 26965450, doi 10.6084/m9.figshare.26965450), "
            "CC BY 4.0 image record -- attribution required (article Sci Data 2024 doi 10.1038/s41597-024-04056-8)",
    "vmr0248": "Vascular Model Repository 0248_H_AOCERE_CAS: research and development use; keep LICENSE_VMR_0248.txt; "
               "acknowledge Wilson et al. 2013 (doi 10.1115/1.4025983), NLM R01LM013120 / NHLBI HHSN268201100035C",
}
ARTERIAL_NC = ("VA trace derived through Arterial (CC BY-NC 4.0 weights, PolyForm-NC code): the derived anatomy is "
               "NON-COMMERCIAL")
TS_ATTR = ("TotalSegmentator (CC BY 4.0, Wasserthal et al. 2023, doi 10.1148/ryai.230024) used for vertebra landmarks / "
           "QC references: attribution")


def L(p):
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return None


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def win(p):
    return p.replace(REPO + "/", "D:\\Arjun\\neve\\").replace("/", "\\") if p and p.startswith(REPO) else p


def mtime(p):
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(p))) if p and os.path.exists(p) else None


T = L(RES + "/scaleup_tally.json")
E = L(RES + "/scaleup_eye_rejections.json") or {}
eye = {}
for e in E.get("rejections") or []:
    eye.setdefault((e.get("source"), e.get("case"), e.get("side")), []).append(e)
rk = L(RES + "/pilot_donor_ranking.json") or {}


def side_block(src, case, s, v):
    out = {"qc_status": v.get("status"), "trace_set": v.get("trace_set"), "wall_tag": v.get("wall_tag"),
           "donor_u03_contact_free": v.get("donor_u03_contact_free"), "donor_u03_all_bins": v.get("donor_u03_all_bins"),
           "donor_u03_all_bins_source": v.get("donor_u03_all_bins_source"), "contact_free_share": v.get("contact_free_share"),
           "genus_whole": v.get("genus_whole"), "wall_valid": v.get("wall_valid"), "wall_truncated": v.get("wall_truncated"),
           "partial_truncated_wall": v.get("partial_truncated_wall"), "v1_rescue_used": v.get("v1_rescue_used"),
           "pinch_fix_triggered": v.get("pinch_fix_triggered"), "review": v.get("review"),
           "provisional_only": v.get("provisional_only")}
    der = "%s/%s/%s" % (DERT, src, case)
    ts = v.get("trace_set") or "tsba"
    qf = "qc.json" if ts == "tsba" else "qc_%s.json" % ts
    out["qc_file"] = "%s/qc/%s" % (der, qf) if os.path.isfile("%s/qc/%s" % (der, qf)) else None
    tr = "%s/arterial/%s/va_extract.json" % (der, ts)
    out["trace_file"] = tr if os.path.isfile(tr) else None
    tag = v.get("wall_tag") or "ct_seg4"
    sd = "%s/ct_seg/%s" % (der, tag)
    surf, vm, me = "%s/%s_stageA.vtp" % (sd, VPRE[s]), "%s/%s_stageA_vmtk.json" % (sd, VPRE[s]), "%s/measure.json" % sd
    out["wall_files"] = {"surface_vtp": surf if os.path.isfile(surf) else None,
                         "surface_vtp_windows": win(surf) if os.path.isfile(surf) else None,
                         "surface_md5": md5(surf) if os.path.isfile(surf) else None, "surface_mtime": mtime(surf),
                         "vmtk_centerline_json": vm if os.path.isfile(vm) else None,
                         "measure_json": me if os.path.isfile(me) else None}
    if tag != "ct_seg4" or ts != "tsba":
        out["donor_tool_caveat"] = ("graft_v5_reg_donor.py build_thin() reads ct_seg/ct_seg4 (else ct_seg4_nofix) and "
                                    "qc/qc.json; this side's graded wall is ct_seg/%s on trace set %s (qc %s): point the "
                                    "donor record at these files" % (tag, ts, qf))
    ev = eye.get((src, case, s)) or []
    out["eye"] = [{"verdict": e.get("verdict", "reject"), "by": e.get("by"), "when": e.get("when"),
                   "why": (e.get("why") or "")[:400]} for e in ev]
    return out


def meta_block(src, case):
    if src == "vmr0248":
        return {"dataset_ref": "VMR 0248_H_AOCERE_CAS", "licence": LIC_DEFAULT["vmr0248"], "arterial_derived": False,
                "non_commercial": False, "attributions": [LIC_DEFAULT["vmr0248"]], "scanner": None,
                "acquisition_voxel_mm": [0.40, 0.40, 0.70]}
    m = L("%s/%s/%s/meta.json" % (SRCT, src, case)) or {}
    srcm = m.get("source") if isinstance(m.get("source"), dict) else {}
    lic = m.get("licence") or srcm.get("licence")
    sc = m.get("scanner") or {}

    def find(o, key):
        if isinstance(o, dict):
            if key in o:
                return o[key]
            for v in o.values():
                r = find(v, key)
                if r is not None:
                    return r
        return None
    vox = find(m, "acquisition_voxel_mm")
    slc = find(m, "slice_class_mm")
    flags = []
    man = (sc.get("manufacturer") or sc.get("family") or "") if isinstance(sc, dict) else ""
    sdesc = (sc.get("series_description") or "") if isinstance(sc, dict) else ""
    if src == "song" and "SIEMENS" in man.upper():
        flags.append("Siemens Song case (series '%s': 1.0 mm slice thickness, %s mm grid): accepted at native bars "
                     "(decision 03:50 item 6)" % (sdesc.strip(), slc))
    return {"dataset_ref": LIC_DEFAULT[src], "licence_in_meta": lic, "licence": LIC_DEFAULT[src] + "; " + ARTERIAL_NC,
            "arterial_derived": True, "non_commercial": True, "attributions": [LIC_DEFAULT[src], TS_ATTR],
            "scanner": ({"manufacturer": sc.get("manufacturer") or sc.get("family"), "model": sc.get("model"),
                         "kernel": sc.get("kernel"), "series_description": sc.get("series_description")}
                        if isinstance(sc, dict) else sc),
            "acquisition_voxel_mm": vox, "slice_class_mm": slc, "provenance_flags": flags,
            "meta_file": "%s/%s/%s/meta.json" % (SRCT, src, case)}


pairs = []
for p in T["eligible_pairs"]:
    src, case = p["source"], p["case"]
    rec = {"tally_rank": p["rank"], "source": src, "case": case, "patient_key": "%s:%s" % (src, case),
           "tally_origin": p.get("origin")}
    res = RESERVES.get((src, case))
    rec["class"] = "reserve" if res else "primary"
    if res:
        rec["reserve"] = res
    rec.update(meta_block(src, case))
    rec["coarseness"] = {"score_mean_contact_free": p.get("score_mean_contact_free"),
                         "score_min_side_contact_free": p.get("score_min_side_contact_free"),
                         "R_L_contact_free": p.get("R_L_contact_free"), "R_L_all_bins": p.get("R_L_all_bins"),
                         "all_bins_gt_dense_host_both": p.get("all_bins_gt_dense_host_both"),
                         "contact_free_ge_dense_host_both": p.get("contact_free_ge_dense_host_both"),
                         "dense_host_RVA_mesh": HOST_DENSE}
    rec["flags"] = list(p.get("flags") or [])
    rec["eye_check_pending_sides"] = p.get("eye_check_pending_sides")
    if src == "vmr0248":
        vroot = REPO + "/carotid_data/va_sources/derived/vmr0248/ct_seg4"
        rec["sides"] = {}
        for s in ("R", "L"):
            surf = "%s/%s_stageA.vtp" % (vroot, VPRE[s])
            rec["sides"][s] = {"wall_tag": "ct_seg4", "wall_files": {
                "surface_vtp": surf if os.path.isfile(surf) else None,
                "surface_vtp_windows": win(surf) if os.path.isfile(surf) else None,
                "surface_md5": md5(surf) if os.path.isfile(surf) else None, "surface_mtime": mtime(surf)},
                "donor_u03_contact_free": p["R_L_contact_free"][0 if s == "R" else 1],
                "donor_u03_all_bins": p["R_L_all_bins"][0 if s == "R" else 1],
                "note": "expert VMR path; left V1 s 10-40 mm vein contact -> tube below 40 mm (prototype rule)" if s == "L" else
                        "expert VMR path"}
    else:
        rec["sides"] = {s: side_block(src, case, s, p[s]) for s in ("R", "L")}
    pairs.append(rec)

prim = [r for r in pairs if r["class"] == "primary"]
resv = sorted([r for r in pairs if r["class"] == "reserve"], key=lambda r: r["reserve"]["reserve_order"])
for i, r in enumerate(prim):
    r["primary_rank"] = i + 1
by_src = {}
for r in prim:
    by_src[r["source"]] = by_src.get(r["source"], 0) + 1
missing = [(r["case"], s) for r in pairs for s in ("R", "L") if not (r["sides"][s].get("wall_files") or {}).get("surface_vtp")]
caveat = [(r["case"], s) for r in pairs for s in ("R", "L") if r["sides"][s].get("donor_tool_caveat")]
out = {
    "what": "v5 FINAL eligible VA donor list (v5_eye172, %s): eligible pairs of the re-run tally (scaleup_tally.py) after "
            "the by-eye checks of the flagged 172-series pilot sides, split PRIMARY / RESERVE." % time.strftime("%Y-%m-%d %H:%M"),
    "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
    "inputs": {"tally": RES + "/scaleup_tally.json", "tally_generated": T.get("generated"),
               "eye_verdicts": RES + "/scaleup_eye_rejections.json", "eye_verdicts_mtime": mtime(RES + "/scaleup_eye_rejections.json")},
    "rules": {"eligibility": T["rules"]["eligibility"], "coarseness": T["rules"]["coarseness"],
              "classes": "RESERVE = ia172_172 then Song AHMU1218138 (decision V5_BUILD_LOG 14:25, confirmed by eye by "
                         "v5_eye172): used only if a TopBrain patient has no other gate-passing donor; else keep the v4 VA "
                         "(flagged). PRIMARY = every other eligible pair, ranked by score_mean_contact_free (tally order).",
              "side_use": "native anatomy: donor R -> RVA, donor L -> LVA; mirrored ('_L') anatomy: donor L mirrored -> RVA, "
                          "donor R mirrored -> LVA. The RVA is the training-reachable side.",
              "licences": "ia172 / Song: CC BY 4.0 image (attribution) + Arterial-derived = NON-COMMERCIAL; VMR 0248: "
                          "research use + acknowledgement; TotalSegmentator CC BY 4.0 attribution; TaG-Net never a donor."},
    "counts": {"eligible_pairs": len(pairs), "primary": len(prim), "reserve": len(resv), "primary_by_source": by_src,
               "need_pairs": 25,
               "primary_all_bins_gt_dense_host_both": sum(bool(r["coarseness"]["all_bins_gt_dense_host_both"]) for r in prim),
               "primary_contact_free_ge_dense_host_both": sum(bool(r["coarseness"]["contact_free_ge_dense_host_both"]) for r in prim),
               "eye_check_pending_pairs": sum(bool(r.get("eye_check_pending_sides")) for r in pairs),
               "tally_counts": T.get("counts"),
               "sides_missing_wall_file": missing, "sides_with_donor_tool_caveat": caveat},
    "open_issues": [
        "graft_v5_reg_donor.py build_thin() (written 01:25, before pipeline v2) reads the wall from ct_seg/ct_seg4 (else "
        "ct_seg4_nofix), the QC from results/pilot_proc_<src>.json else qc/qc.json, and the tsba / ts traces; for the sides in "
        "counts.sides_with_donor_tool_caveat the graded wall is ct_seg4v2 and/or the trace set is ts / tsba_v1r (all 14 v2 "
        "pilot-case surfaces differ from their ct_seg4 by md5; Song 141 R has no ct_seg4 rva surface). The assign agent must "
        "build those donor records from sides.<R|L>.wall_files / qc_file / trace_file here (stated extension of the tool).",
        "Song cases 113, 094, 109 (primary) and 138 (reserve) are Siemens 1.0 mm-slice series (provenance flag).",
        "Reserve conditions: ia172_172 L ostium advanced >= 3 mm (arch-origin VA); Song 138 R real wall origin -> C2 only."],
    "primary": prim, "reserve": resv}
json.dump(out, open(RES + "/v5_eligible_final.json", "w"), indent=1, default=str)
print("eligible %d: primary %d %s, reserve %d; primary coarse all-bins both %d, cf both %d; eye pending pairs %d; "
      "missing wall files %s; donor-tool caveat sides %d" % (
          len(pairs), len(prim), by_src, len(resv), out["counts"]["primary_all_bins_gt_dense_host_both"],
          out["counts"]["primary_contact_free_ge_dense_host_both"], out["counts"]["eye_check_pending_pairs"], missing, len(caveat)))
for r in prim + resv:
    print("%-7s #%-2s %-7s %-12s cf %s all %s %s" % (r["class"], r.get("primary_rank", "-"), r["source"], r["case"],
          r["coarseness"]["R_L_contact_free"], r["coarseness"]["R_L_all_bins"],
          ",".join("%s:%s/%s" % (s, r["sides"][s].get("wall_tag"), r["sides"][s].get("trace_set", "-")) for s in ("R", "L"))))
print("caveat sides:", caveat)
