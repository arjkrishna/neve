# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\pilot_ia172_zip.py (= /sp/v5_va/pilot_ia172_zip.py in the build containers)
"""pilot_ia172_zip: range-read helpers for the Head-CTA 172-series set (Zenodo 15697196, CC BY 4.0).

The record is one 25.6 GB zip of DICOM (Dataset.zip). Zenodo accepts HTTP byte ranges, so a member or a whole series
can be read without the rest. Extends va_pool/rzip.py (HttpFile, used once to read the central directory) with:
  * build_index(): the central directory as {name: [header_offset, compress_size, file_size, compress_type, crc]},
    cached as JSON on D: (the zip has 149,452 members);
  * get_member(): one member in one range request (local header + data), decompressed and CRC-checked;
  * fetch_span(): a contiguous byte span in parallel chunks with retry (series are stored contiguously);
  * series_members(): the members of Studies/<sid>/ sorted by offset, with a contiguity check.
Container: neve-build-stagea, PYTHONPATH=/sp/va_pool/pylib (pydicom 2.4.4 + pylibjpeg for JPEG-lossless DICOM).
"""
import concurrent.futures as cf
import io
import json
import os
import struct
import sys
import time
import urllib.request
import zipfile
import zlib

URL = "https://zenodo.org/records/15697196/files/Dataset.zip?download=1"
RECORD = "https://zenodo.org/records/15697196"
DOI = "10.5281/zenodo.15697196"
PAPER_DOI = "10.3390/data11040074"
IDX_PATH = "/repo/carotid_data/va_sources/derived/thin_slice/ia172/_zip_index.json"


def http_range(a, b, url=URL, tries=8):
    """bytes [a, b] inclusive, with retry/backoff."""
    last = None
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"Range": "bytes=%d-%d" % (a, b)})
            with urllib.request.urlopen(req, timeout=300) as r:
                if r.status != 206:
                    raise IOError("HTTP status %s for a range request" % r.status)
                data = r.read()
            if len(data) != b - a + 1:
                raise IOError("short read %d of %d" % (len(data), b - a + 1))
            return data
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(10 * (k + 1))
    raise last


def build_index(force=False):
    if os.path.exists(IDX_PATH) and not force:
        return json.load(open(IDX_PATH))
    sys.path.insert(0, "/sp/va_pool")
    from rzip import HttpFile
    hf = HttpFile(URL)
    z = zipfile.ZipFile(io.BufferedReader(hf, buffer_size=1 << 22))
    mem = {x.filename: [x.header_offset, x.compress_size, x.file_size, x.compress_type, x.CRC]
           for x in z.infolist() if not x.is_dir()}
    res = {"url": URL, "zip_size": hf.size, "n_members": len(mem), "members": mem,
           "read": {"requests": hf.nreq, "bytes": hf.nbytes, "date": time.strftime("%Y-%m-%d %H:%M")}}
    os.makedirs(os.path.dirname(IDX_PATH), exist_ok=True)
    json.dump(res, open(IDX_PATH, "w"))
    return res


def _parse_local(buf, pos, name, rec):
    """parse the local file header at buf[pos:]; return (data_start, data_end) offsets inside buf."""
    sig, = struct.unpack("<I", buf[pos:pos + 4])
    if sig != 0x04034B50:
        raise IOError("bad local header signature for %s" % name)
    nlen, xlen = struct.unpack("<HH", buf[pos + 26:pos + 30])
    lname = buf[pos + 30:pos + 30 + nlen].decode("utf-8", "replace")
    if lname != name:
        raise IOError("local name %r != %r" % (lname, name))
    s = pos + 30 + nlen + xlen
    return s, s + rec[1]


def _inflate(raw, rec, name):
    ctype = rec[3]
    if ctype == 0:
        data = raw
    elif ctype == 8:
        data = zlib.decompressobj(-15).decompress(raw)
    else:
        raise IOError("compress type %d unsupported (%s)" % (ctype, name))
    if len(data) != rec[2] or (zlib.crc32(data) & 0xFFFFFFFF) != rec[4]:
        raise IOError("size/CRC mismatch for %s" % name)
    return data


def get_member(idx, name, head_only=None):
    """whole member (CRC-checked), or with head_only=N the first ~N decompressed bytes (no CRC)."""
    rec = idx["members"][name]
    off, csize = rec[0], rec[1]
    want = csize if head_only is None else min(csize, head_only)
    buf = http_range(off, off + 30 + len(name.encode()) + 512 + want - 1)
    s, e = _parse_local(buf, 0, name, rec)
    if head_only is None:
        if e > len(buf):
            buf += http_range(off + len(buf), off + e - 1)
        return _inflate(buf[s:e], rec, name)
    raw = buf[s:min(e, len(buf))]
    return raw if rec[3] == 0 else zlib.decompressobj(-15).decompress(raw)


def series_members(idx, sid):
    pre = "Studies/%s/" % sid
    names = [n for n in idx["members"] if n.startswith(pre) and n.lower().endswith(".dcm")]
    names.sort(key=lambda n: idx["members"][n][0])
    offs = [idx["members"][n][0] for n in names]
    # contiguity: any other member stored inside the span?
    lo, hi = offs[0], offs[-1]
    inside = sum(1 for n, r in idx["members"].items() if lo <= r[0] <= hi and not n.startswith(pre))
    return names, inside


def fetch_span(a, b, chunk=8 << 20, workers=4, log=None):
    """bytes [a, b] inclusive in parallel chunks; returns bytes."""
    parts = [(s, min(s + chunk - 1, b)) for s in range(a, b + 1, chunk)]
    out = [None] * len(parts)
    t0 = time.time(); done = [0]

    def one(i):
        s, e = parts[i]
        out[i] = http_range(s, e)
        done[0] += e - s + 1
        if log and (i % 8 == 0):
            dt = time.time() - t0
            log("  %.0f/%.0f MB  %.2f MB/s" % (done[0] / 1e6, (b - a + 1) / 1e6, done[0] / 1e6 / max(dt, 1e-3)))
    with cf.ThreadPoolExecutor(workers) as ex:
        list(ex.map(one, range(len(parts))))
    return b"".join(out)


def fetch_series(idx, sid, dest, workers=4, log=print):
    """download every DICOM of Studies/<sid>/ into dest/ (one span read), CRC-checked. Returns a record."""
    names, inside = series_members(idx, sid)
    recs = [idx["members"][n] for n in names]
    a = recs[0][0]
    last = names[-1]
    b = recs[-1][0] + 30 + len(last.encode()) + 1024 + recs[-1][1] - 1
    b = min(b, idx["zip_size"] - 1)
    t = time.time()
    blob = fetch_span(a, b, workers=workers, log=log)
    dt = time.time() - t
    os.makedirs(dest, exist_ok=True)
    nbytes = 0
    for n, r in zip(names, recs):
        s, e = _parse_local(blob, r[0] - a, n, r)
        data = _inflate(blob[s:e], r, n)
        open(os.path.join(dest, os.path.basename(n)), "wb").write(data)
        nbytes += len(data)
    return {"n_files": len(names), "byte_range": [a, b], "span_MB": round((b - a + 1) / 1e6, 1),
            "other_members_in_span": inside, "seconds": round(dt, 1),
            "MB_per_s": round((b - a + 1) / 1e6 / max(dt, 1e-3), 2), "dicom_MB": round(nbytes / 1e6, 1)}
