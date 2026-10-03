#!/usr/bin/env python3
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\pilot_fetch_song_table.py (= /sp/v5_va/pilot_fetch_song_table.py in the build containers)
"""pilot_fetch_song_table.py: one row per Song case = archive index (header geometry) + clinical_all.csv (sex, age,
aneurysm location) + the dcm2niix sidecar (kernel, software, protocol; scanner family inferred).
Writes /sp/v5_va/results/pilot_fetch_song_table.json and prints a compact table.
"""
import json, csv, zlib, os, io, collections

IDX = "/sp/v5_va/results/pilot_fetch_song_index.json"
M = "/repo/carotid_data/va_sources/thin_slice/song/_source_meta"
OUT = "/sp/v5_va/results/pilot_fetch_song_table.json"


def scanner_family(sc):
    sw = str(sc.get("SoftwareVersions", "")); k = str(sc.get("ConvolutionKernel", ""))
    if sc.get("Manufacturer"):
        return sc["Manufacturer"] + " " + sc.get("ManufacturersModelName", ""), "Manufacturer tag"
    if sw.startswith("syngo CT"):
        return "Siemens (inferred)", "SoftwareVersions syngo CT, kernel %s" % k
    if "IMR" in k:
        return "Philips (inferred; one case of this group tags iCT 256)", "kernel IMR (Philips Iterative Model Reconstruction), SoftwareVersions %s" % sw
    if k == "STANDARD":
        return "GE (inferred)", "kernel STANDARD (GE kernel name), SoftwareVersions %s" % sw
    return "unknown", "SoftwareVersions %s, kernel %s" % (sw, k)


def main():
    idx = json.load(open(IDX))
    clin = {}
    with io.open(os.path.join(M, "statistical results", "clinical_all.csv"), encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    hdr = rows[0]
    for r in rows[2:]:
        if not r or not r[0].startswith("AHMU"):
            continue
        d = dict(zip(hdr, r))
        c = clin.setdefault(r[0], dict(sex=d["Gender"], age=int(d["Age"]), has_aneurysm=d["Has aneurysm"],
                                       locations=[], n_rows=0))
        c["n_rows"] += 1
        if d.get("location"):
            c["locations"].append(d["location"])
    table = {}
    for case, r in sorted(idx["cases"].items()):
        n = r.get("nii", {})
        mem = idx["members"][r["ct_member"].replace(".nii.gz", ".json")]
        b = open(os.path.join(M, "sidecars", case + ".json"), "rb").read()
        sc = json.loads(b) if zlib.crc32(b) & 0xFFFFFFFF == mem["crc32"] else {}
        fam, basis = scanner_family(sc)
        ct = idx["members"][r["ct_member"]]
        sh = n.get("shape", []); zo = n.get("zooms", [])
        row = dict(case=case, group=r["group"], ndim=len(sh), shape=sh, zooms=zo,
                   z_mm=round(sh[2] * zo[2], 1) if len(sh) >= 3 else None,
                   fov_mm=round(sh[0] * zo[0], 1) if sh else None, slice_mm=zo[2] if len(zo) >= 3 else None,
                   datatype=n.get("datatype"), axcodes=n.get("axcodes"), ct_bytes=ct["packed"],
                   sex=clin.get(case, {}).get("sex"), age=clin.get(case, {}).get("age"),
                   aneurysm_locations=clin.get(case, {}).get("locations"),
                   kernel=sc.get("ConvolutionKernel"), software=sc.get("SoftwareVersions"), protocol=sc.get("ProtocolName"),
                   series_description=sc.get("SeriesDescription"), device_serial=sc.get("DeviceSerialNumber"),
                   scanner_family=fam, scanner_basis=basis, sidecar_crc_ok=bool(sc))
        table[case] = row
    json.dump(table, open(OUT, "w"), indent=1)
    cnt = collections.Counter()
    for c, r in table.items():
        sl = r["slice_mm"]
        sl_cls = None if sl is None else ("0.45" if abs(sl - 0.45) < 0.02 else "0.625" if abs(sl - 0.625) < 0.01 else
                                          "0.7" if abs(sl - 0.7) < 0.01 else str(sl))
        r["slice_class"] = sl_cls
        ok = r["ndim"] == 3 and r["z_mm"] and r["z_mm"] >= 300
        cnt[(r["group"], sl_cls, ok)] += 1
        print("%s %-8s %s/%-3s %5s z%6s fov%6s %-28s %-18s %-40s %s" % (
            c, r["group"], r["sex"], r["age"], sl_cls, r["z_mm"], r["fov_mm"], r["scanner_family"][:28],
            str(r["kernel"])[:18], str(r["protocol"])[:40], r["device_serial"]))
    json.dump(table, open(OUT, "w"), indent=1)
    for k, v in sorted(cnt.items(), key=str):
        print(k, v)


if __name__ == "__main__":
    main()
