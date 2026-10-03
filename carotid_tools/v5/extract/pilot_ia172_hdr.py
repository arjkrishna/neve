# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\pilot_ia172_hdr.py (= /sp/v5_va/pilot_ia172_hdr.py in the build containers)
"""pilot_ia172_hdr: full DICOM headers of every series of the 172-series set, read remotely, for selection and
de-duplication. For each Studies/<sid>/ it reads the first, middle and last file (by file number) and keeps the
acquisition/geometry tags of the first plus the IPP of all three. -> /sp/v5_va/results/pilot_ia172_headers.json

Container: neve-build-stagea, PYTHONPATH=/sp/va_pool/pylib.  usage: python3 pilot_ia172_hdr.py
"""
import collections
import concurrent.futures as cf
import io
import json
import os
import sys

import pydicom

sys.path.insert(0, "/sp/v5_va")
from pilot_ia172_zip import build_index, get_member  # noqa: E402

OUT = "/sp/v5_va/results/pilot_ia172_headers.json"
TAGS = ["StudyInstanceUID", "SeriesInstanceUID", "FrameOfReferenceUID", "StackID", "SeriesNumber",
        "AcquisitionNumber", "ImageType", "Manufacturer", "ManufacturerModelName", "SoftwareVersions", "StationName",
        "InstitutionName", "DeviceSerialNumber", "BodyPartExamined", "PatientPosition", "PatientSex", "PatientAge",
        "PatientID", "PatientName", "PatientBirthDate", "StudyDate", "StudyTime", "AcquisitionTime", "ContentTime",
        "SeriesTime", "SeriesDescription", "ProtocolName", "ContrastBolusAgent", "ContrastBolusRoute",
        "ContrastBolusVolume", "ContrastBolusStartTime", "KVP", "XRayTubeCurrent", "Exposure", "ExposureTime",
        "CTDIvol", "SpiralPitchFactor", "ExposureModulationType", "EstimatedDoseSaving", "ReconstructionDiameter",
        "DataCollectionDiameter", "TableHeight", "GantryDetectorTilt", "ConvolutionKernel", "FilterType",
        "FocalSpots", "SliceThickness", "SpacingBetweenSlices", "PixelSpacing", "Rows", "Columns",
        "ImagePositionPatient", "ImageOrientationPatient", "SliceLocation", "RescaleSlope", "RescaleIntercept",
        "RescaleType", "WindowCenter", "WindowWidth", "PixelRepresentation", "BitsStored", "PatientOrientation",
        "InStackPositionNumber", "InstanceNumber", "ImageComments"]


def val(v):
    if isinstance(v, pydicom.multival.MultiValue):
        return [val(x) for x in v]
    if isinstance(v, (pydicom.valuerep.DSfloat, pydicom.valuerep.DSdecimal, float)):
        return float(v)
    if isinstance(v, (pydicom.valuerep.IS, int)):
        return int(v)
    return str(v)


def head(idx, name):
    raw = get_member(idx, name, head_only=24576)
    ds = pydicom.dcmread(io.BytesIO(raw), stop_before_pixels=True, force=True)
    r = {t: val(getattr(ds, t)) for t in TAGS if getattr(ds, t, None) not in (None, "")}
    try:
        r["TransferSyntaxUID"] = str(ds.file_meta.TransferSyntaxUID)
    except Exception:  # noqa: BLE001
        pass
    for sq, key in (("ProcedureCodeSequence", "procedure"), ("PerformedProtocolCodeSequence", "performed_protocol")):
        try:
            r[key] = str(getattr(ds, sq)[0].CodeMeaning)
        except Exception:  # noqa: BLE001
            pass
    return r


def one(idx, sid, names):
    files = sorted(names, key=lambda n: int(os.path.basename(n).split(".")[0]))
    pick = {"first": files[0], "mid": files[len(files) // 2], "last": files[-1]}
    rec = {"n_files": len(files), "files": {k: os.path.basename(v) for k, v in pick.items()}}
    try:
        h = head(idx, pick["first"])
        rec["first"] = h
        for k in ("mid", "last"):
            hk = head(idx, pick[k])
            rec[k] = {t: hk.get(t) for t in ("ImagePositionPatient", "InstanceNumber", "InStackPositionNumber",
                                              "SliceLocation", "SeriesInstanceUID", "AcquisitionNumber",
                                              "ContentTime", "ImageOrientationPatient")}
        zs = [rec[k]["ImagePositionPatient"][2] if k != "first" else h["ImagePositionPatient"][2]
              for k in ("first", "mid", "last")]
        rec["z_first_mid_last"] = zs
        rec["z_extent_mm"] = round(abs(zs[2] - zs[0]), 2)
        rec["z_step_mm"] = round(abs(zs[2] - zs[0]) / max(1, len(files) - 1), 4)
        rec["file_order_monotonic_z"] = bool((zs[1] - zs[0]) * (zs[2] - zs[1]) > 0)
    except Exception as e:  # noqa: BLE001
        rec["error"] = repr(e)[:300]
    return sid, rec


def main():
    idx = build_index()
    print("index", idx["n_members"], "members", idx.get("read"), flush=True)
    by = collections.defaultdict(list)
    for n in idx["members"]:
        p = n.split("/")
        if len(p) == 3 and p[0] == "Studies" and n.lower().endswith(".dcm"):
            by[p[1]].append(n)
    print("series", len(by), flush=True)
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    todo = [s for s in sorted(by) if s not in res or "error" in res[s]]
    with cf.ThreadPoolExecutor(4) as ex:
        for f in cf.as_completed([ex.submit(one, idx, s, by[s]) for s in todo]):
            sid, rec = f.result()
            res[sid] = rec
            h = rec.get("first", {})
            print(sid, rec["n_files"], rec.get("z_extent_mm"), rec.get("z_step_mm"), h.get("SliceThickness"),
                  h.get("PixelSpacing"), h.get("ManufacturerModelName"), h.get("ConvolutionKernel"),
                  rec.get("error", ""), flush=True)
    json.dump(dict(sorted(res.items())), open(OUT, "w"), indent=1)
    print("wrote", OUT, len(res))


if __name__ == "__main__":
    main()
