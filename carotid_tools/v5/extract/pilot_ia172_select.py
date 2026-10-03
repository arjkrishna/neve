# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\pilot_ia172_select.py (= /sp/v5_va/pilot_ia172_select.py in the build containers)
"""pilot_ia172_select: candidate table + duplicate screen for the 172-series set, from the remote headers
(results/pilot_ia172_headers.json) and the dataset's Metadata.csv (va_pool/data/ia172/Metadata.csv).

Duplicate screen (PatientID is blank in every DICOM, so it cannot be used):
  * exact keys shared between series: StudyInstanceUID, FrameOfReferenceUID, SeriesInstanceUID, StackID;
  * identical acquisition: same scanner model + software + table height + first IPP (x, y, z) + n files + kVp + mA;
  * same patient demographics (sex, age within 1 y) AND same scanner model and table height within 5 mm.
-> results/pilot_ia172_candidates.json; prints the table.
usage: python3 pilot_ia172_select.py
"""
import collections
import csv
import json

H = json.load(open("/sp/v5_va/results/pilot_ia172_headers.json"))
rows = list(csv.reader(open("/sp/va_pool/data/ia172/Metadata.csv", encoding="utf-8", errors="replace"), delimiter=";"))
hdr = rows[0]
meta = {}
for r in rows[1:]:
    if not r or not r[0]:
        continue
    sid = r[0]
    if sid not in meta:  # the first row of a case carries sex, age and the series description
        meta[sid] = {"ia": r[1], "sex": r[2], "age": int(r[3]) if r[3].isdigit() else None,
                     "artery": r[4], "size": r[8], "images": r[18], "thick_csv": r[19].replace(",", "."),
                     "head_neck": r[20], "arteries": []}
    if r[4] != "NA":
        meta[sid]["arteries"].append(r[4] + ":" + r[8])

cand = {}
for sid, h in H.items():
    f = h.get("first", {})
    m = meta.get(sid, {})
    ps = f.get("PixelSpacing", [None])[0]
    c = {"sid": sid, "n": h["n_files"], "z": h.get("z_extent_mm"), "dz": h.get("z_step_mm"),
         "thick": f.get("SliceThickness"), "ps": round(ps, 3) if ps else None,
         "rdiam": f.get("ReconstructionDiameter"), "model": f.get("ManufacturerModelName"),
         "manu": f.get("Manufacturer"), "sw": f.get("SoftwareVersions"), "kernel": f.get("ConvolutionKernel"),
         "kvp": f.get("KVP"), "ma": f.get("XRayTubeCurrent"), "table": f.get("TableHeight"),
         "ipp0": f.get("ImagePositionPatient"), "iop": f.get("ImageOrientationPatient"),
         "tilt": f.get("GantryDetectorTilt"), "study": f.get("StudyInstanceUID"), "for": f.get("FrameOfReferenceUID"),
         "series": f.get("SeriesInstanceUID"), "stack": f.get("StackID"), "acq": f.get("AcquisitionNumber"),
         "ser_no": f.get("SeriesNumber"), "pos": f.get("PatientPosition"), "body": f.get("BodyPartExamined"),
         "proc": f.get("procedure"), "pid": f.get("PatientID"), "psex": f.get("PatientSex"), "page": f.get("PatientAge"),
         "ts": f.get("TransferSyntaxUID"), "slope": f.get("RescaleSlope"), "inter": f.get("RescaleIntercept"),
         "monotonic": h.get("file_order_monotonic_z"), "z_fml": h.get("z_first_mid_last"),
         "ia": m.get("ia"), "sex": m.get("sex"), "age": m.get("age"), "artery": m.get("artery"),
         "arteries": m.get("arteries"), "ia_size": m.get("size"), "head_neck": m.get("head_neck")}
    cand[sid] = c

# duplicate screen
dups = collections.defaultdict(list)
for key in ("study", "for", "series", "stack"):
    g = collections.defaultdict(list)
    for sid, c in cand.items():
        if c[key]:
            g[c[key]].append(sid)
    for v, s in g.items():
        if len(s) > 1:
            dups["shared_" + key].append(s)
g = collections.defaultdict(list)
for sid, c in cand.items():
    k = (c["model"], c["sw"], c["table"], json.dumps(c["ipp0"]), c["n"], c["kvp"], c["ma"])
    g[k].append(sid)
dups["identical_acquisition"] = [s for s in g.values() if len(s) > 1]
pairs = []
ids = sorted(cand)
for i, a in enumerate(ids):
    for b in ids[i + 1:]:
        A, B = cand[a], cand[b]
        if A["sex"] and A["sex"] == B["sex"] and A["age"] is not None and B["age"] is not None \
                and abs(A["age"] - B["age"]) <= 1 and A["model"] == B["model"] and A["table"] is not None \
                and B["table"] is not None and abs(A["table"] - B["table"]) <= 5:
            pairs.append([a, b])
dups["same_demographics_scanner_table"] = pairs
out = {"candidates": cand, "duplicate_screen": dups}
json.dump(out, open("/sp/v5_va/results/pilot_ia172_candidates.json", "w"), indent=1)

print("dup screen:", {k: (len(v), v[:30]) for k, v in dups.items()})
print("unique values: pid", collections.Counter(c["pid"] for c in cand.values()).most_common(3),
      "psex", collections.Counter(c["psex"] for c in cand.values()).most_common(3))
print("models", collections.Counter((c["manu"], c["model"]) for c in cand.values()).most_common())
print("ts", collections.Counter(c["ts"] for c in cand.values()).most_common())
print("iop", collections.Counter(json.dumps(c["iop"]) for c in cand.values()).most_common(5))
print("tilt", collections.Counter(c["tilt"] for c in cand.values()).most_common(5))
print("pos", collections.Counter(c["pos"] for c in cand.values()).most_common(5))
print("proc", collections.Counter(c["proc"] for c in cand.values()).most_common(8))
print("%4s %2s %3s %5s %4s %5s %5s %5s %6s %-22s %-12s %-7s %5s %4s %6s %s" % (
    "sid", "sx", "age", "ia", "n", "z", "dz", "thk", "ps", "model", "sw", "kern", "kvp", "mA", "table", "art"))
for sid in ids:
    c = cand[sid]
    if c["head_neck"] != "1" or (c["z"] or 0) < 300:
        continue
    print("%4s %2s %3s %5s %4d %5.0f %5.3f %5s %6s %-22s %-12s %-7s %5s %4s %6s %s %s" % (
        sid, c["sex"], c["age"], c["ia"], c["n"], c["z"], c["dz"], c["thick"], c["ps"], (c["model"] or "")[:22],
        (c["sw"] or "")[:12], c["kernel"], c["kvp"], c["ma"], c["table"], c["artery"], "" if c["monotonic"] else "NONMONO"))
