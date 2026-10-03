#!/usr/bin/env python3
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\pilot_song_rar.py (= /sp/v5_va/pilot_song_rar.py in the build containers)
"""pilot_song_rar.py (pilot_fetch_song): full RAR5 header walk of the two Song et al. 2024 archives
(figshare 26965450: controls.rar = file 49199083, patients.rar = file 49201807) over HTTP range requests.

Extends va_pool/rar5_walk.py (unchanged there). Diff against rar5_walk.py:
  1. walks ALL members (no subject cap), and stores every file record (rar5_walk kept only the first 50);
  2. records the header offset and the RAR5 data CRC32 of every member (needed to verify a range-read member);
  3. reads the FULL NIfTI-1 header of every stored (method 0) .nii.gz member with nibabel (rar5_walk parsed only
     shape / spacing / srow): datatype, scl_slope / scl_inter, qform / sform codes and matrices, axis codes,
     cal_min / cal_max, descrip.
Output: /sp/v5_va/results/pilot_fetch_song_index.json
Run:  docker run --rm -v "D:\\Arjun\\neve:/repo" -v "<scratchpad>:/sp" neve-build-stagea python3 /sp/v5_va/pilot_song_rar.py
"""
import sys, json, time, zlib, io, struct, urllib.request
import numpy as np
import nibabel as nib

ARCHIVES = {"controls": "https://ndownloader.figshare.com/files/49199083",
            "patients": "https://ndownloader.figshare.com/files/49201807"}
OUT = "/sp/v5_va/results/pilot_fetch_song_index.json"
NREQ = 0


def get(url, a, n):
    global NREQ
    for k in range(8):
        try:
            req = urllib.request.Request(url, headers={"Range": "bytes=%d-%d" % (a, a + n - 1)})
            with urllib.request.urlopen(req, timeout=90) as r:
                NREQ += 1
                return r.read()
        except Exception as e:
            print("retry", k, a, repr(e), flush=True)
            time.sleep(5 * (k + 1))
    raise RuntimeError("range fail %s %d" % (url, a))


def vint(b, i):
    v = 0; s = 0
    while True:
        c = b[i]; i += 1
        v |= (c & 0x7F) << s; s += 7
        if not c & 0x80:
            return v, i


def walk(url):
    pos = 8
    files = []
    while True:
        b = get(url, pos, 4096)
        if len(b) < 8:
            break
        i = 4
        hsize, i = vint(b, i); hstart = i
        htype, i = vint(b, i); hflags, i = vint(b, i)
        data = 0
        if hflags & 1:
            _, i = vint(b, i)
        if hflags & 2:
            data, i = vint(b, i)
        if htype == 2:
            fflags, i = vint(b, i); usize, i = vint(b, i); attr, i = vint(b, i)
            mtime = crc = None
            if fflags & 2:
                mtime = struct.unpack("<I", b[i:i + 4])[0]; i += 4
            if fflags & 4:
                crc = struct.unpack("<I", b[i:i + 4])[0]; i += 4
            comp, i = vint(b, i); hostos, i = vint(b, i); nlen, i = vint(b, i)
            name = b[i:i + nlen].decode("utf8", "replace")
            files.append(dict(name=name, is_dir=bool(fflags & 1), usize=usize, packed=data,
                              method=(comp >> 7) & 7, solid=bool(comp & 0x40), crc32=crc, mtime=mtime,
                              header_offset=pos, data_offset=pos + hstart + hsize))
        elif htype == 5:
            break
        pos = pos + hstart + hsize + data
    return files


def nifti_header(url, f):
    raw = get(url, f["data_offset"], 65536)
    dec = zlib.decompressobj(31).decompress(raw, 4096)
    h = nib.Nifti1Header.from_fileobj(io.BytesIO(dec[:348]))
    shape = [int(x) for x in h.get_data_shape()]
    zooms = [float(x) for x in h.get_zooms()]
    out = dict(shape=shape, zooms=[round(z, 5) for z in zooms], datatype=str(h.get_data_dtype()),
               scl_slope=None if not np.isfinite(h["scl_slope"]) else float(h["scl_slope"]),
               scl_inter=None if not np.isfinite(h["scl_inter"]) else float(h["scl_inter"]),
               qform_code=int(h["qform_code"]), sform_code=int(h["sform_code"]),
               cal_min=float(h["cal_min"]), cal_max=float(h["cal_max"]),
               descrip=h["descrip"].tobytes().split(b"\0")[0].decode("latin1"),
               xyzt_units=int(h["xyzt_units"]), dim0=int(h["dim"][0]), pixdim=[float(x) for x in h["pixdim"]])
    try:
        A = h.get_best_affine()
        out["affine"] = np.round(A, 5).tolist()
        out["axcodes"] = "".join(nib.aff2axcodes(A))
        out["qform"] = np.round(h.get_qform(), 5).tolist()
        out["sform"] = np.round(h.get_sform(), 5).tolist()
        if len(shape) >= 3:
            out["extent_mm"] = [round(shape[k] * zooms[k], 1) for k in range(3)]
            corners = np.array([[i, j, k, 1] for i in (0, shape[0] - 1) for j in (0, shape[1] - 1)
                                for k in (0, shape[2] - 1)], float)
            w = (A @ corners.T).T[:, :3]
            out["world_bbox_mm"] = [np.round(w.min(0), 1).tolist(), np.round(w.max(0), 1).tolist()]
    except Exception as e:
        out["affine_error"] = repr(e)
    return out


def main():
    res = dict(record="figshare 26965450 (doi:10.6084/m9.figshare.26965450.v1)", archives=ARCHIVES,
               date=time.strftime("%Y-%m-%d %H:%M"), members={}, cases={})
    for grp, url in ARCHIVES.items():
        t0 = time.time()
        files = walk(url)
        print(grp, "members", len(files), "requests", NREQ, "%.0fs" % (time.time() - t0), flush=True)
        for f in files:
            f["archive"] = grp; f["url"] = url
            res["members"][f["name"]] = f
        for f in files:
            parts = f["name"].split("/")
            if f["is_dir"] or len(parts) < 3:
                continue
            case = parts[1]
            c = res["cases"].setdefault(case, dict(group=grp, url=url, files={}))
            c["files"][parts[-1]] = f["name"]
            if f["name"].endswith(".nii.gz") and f["method"] == 0 and "cta_images" in f["name"]:
                try:
                    c["nii"] = nifti_header(url, f)
                except Exception as e:
                    c["nii"] = {"error": repr(e)}
                c["ct_member"] = f["name"]
                print(case, c["nii"].get("shape"), c["nii"].get("zooms"), c["nii"].get("axcodes"),
                      c["nii"].get("datatype"), c["nii"].get("scl_slope"), c["nii"].get("scl_inter"), flush=True)
    res["requests"] = NREQ
    json.dump(res, open(OUT, "w"), indent=1)
    print("cases", len(res["cases"]), "requests", NREQ)


if __name__ == "__main__":
    main()
