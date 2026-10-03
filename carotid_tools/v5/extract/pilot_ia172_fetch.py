# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\pilot_ia172_fetch.py (= /sp/v5_va/pilot_ia172_fetch.py in the build containers)
"""pilot_ia172_fetch: download whole series of the 172-series set by one byte-span read each, then convert.

usage: python3 pilot_ia172_fetch.py <sid> [<sid> ...]       (sids as in the zip: 001, 090, ...)
  images : /repo/carotid_data/va_sources/thin_slice/ia172/ia172_<sid>/dicom/*.dcm   (as stored in the zip)
           /repo/carotid_data/va_sources/thin_slice/ia172/ia172_<sid>/ct.nii.gz    (pilot_ia172_convert.py)
  derived: /repo/carotid_data/va_sources/derived/thin_slice/ia172/ia172_<sid>/fetch.json, convert.json, ct_1p5.nii.gz
Series already on disk in /sp/va_pool/data/ia172/<sid>/ (001, 003) are copied and every file is checked against
the zip index (size and CRC-32) instead of being downloaded again.
Container: neve-build-stagea, PYTHONPATH=/sp/va_pool/pylib.
"""
import json
import os
import shutil
import subprocess
import sys
import time
import zlib

sys.path.insert(0, "/sp/v5_va")
from pilot_ia172_zip import build_index, fetch_series, series_members, URL  # noqa: E402

IMG = "/repo/carotid_data/va_sources/thin_slice/ia172"
DER = "/repo/carotid_data/va_sources/derived/thin_slice/ia172"
LOCAL = "/sp/va_pool/data/ia172"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def main():
    idx = build_index()
    for sid in sys.argv[1:]:
        case = "ia172_" + sid
        cdir, ddir = os.path.join(IMG, case), os.path.join(DER, case)
        dcm = os.path.join(cdir, "dicom")
        os.makedirs(ddir, exist_ok=True)
        fj = os.path.join(ddir, "fetch.json")
        if not os.path.exists(fj):
            names, inside = series_members(idx, sid)
            if os.path.isdir(os.path.join(LOCAL, sid)):
                log(case, "copy from", os.path.join(LOCAL, sid))
                os.makedirs(dcm, exist_ok=True)
                bad = 0
                for n in names:
                    r = idx["members"][n]
                    src = os.path.join(LOCAL, sid, os.path.basename(n))
                    data = open(src, "rb").read()
                    if len(data) != r[2] or (zlib.crc32(data) & 0xFFFFFFFF) != r[4]:
                        bad += 1
                    shutil.copyfile(src, os.path.join(dcm, os.path.basename(n)))
                rec = {"n_files": len(names), "copied_from": os.path.join(LOCAL, sid), "crc_or_size_mismatch": bad,
                       "byte_range": [idx["members"][names[0]][0], None], "other_members_in_span": inside}
            else:
                log(case, "fetch", len(names), "files")
                rec = fetch_series(idx, sid, dcm, workers=4, log=lambda m: log(case, m))
            rec.update({"url": URL, "zip_members": "Studies/%s/*.dcm" % sid, "date": time.strftime("%Y-%m-%d %H:%M")})
            json.dump(rec, open(fj, "w"), indent=1)
            log(case, "fetched", json.dumps(rec))
        ct = os.path.join(cdir, "ct.nii.gz")
        if not os.path.exists(os.path.join(ddir, "convert.json")):
            log(case, "convert")
            rc = subprocess.call([sys.executable, "/sp/v5_va/pilot_ia172_convert.py", dcm, ct, ddir])
            log(case, "convert rc", rc)


if __name__ == "__main__":
    main()
