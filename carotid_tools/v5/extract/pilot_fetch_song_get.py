#!/usr/bin/env python3
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\pilot_fetch_song_get.py (= /sp/v5_va/pilot_fetch_song_get.py in the build containers)
"""pilot_fetch_song_get.py: range-read one Song et al. 2024 CTA member out of its RAR5 archive (stored, method 0,
so the member bytes ARE the shipped .nii.gz) and write it byte-for-byte to
/repo/carotid_data/va_sources/thin_slice/song/<case>/ct.nii.gz, verifying the RAR5 data CRC32.

Uses the index written by pilot_song_rar.py (/sp/v5_va/results/pilot_fetch_song_index.json).
If a byte-identical local copy exists (va_pool/data/samples/song_cta_images_head_<case>.nii.gz, same size and CRC32)
it is copied instead of downloaded.
usage: pilot_fetch_song_get.py CASE [CASE ...]
Writes <case>/fetch.json (url, member, byte range, size, crc32 expected/got, md5, seconds, source).
"""
import sys, os, json, time, zlib, hashlib, urllib.request, shutil

IDX = "/sp/v5_va/results/pilot_fetch_song_index.json"
ROOT = "/repo/carotid_data/va_sources/thin_slice/song"
CHUNK = 16 << 20


def crc_md5_file(p):
    c = 0; m = hashlib.md5()
    with open(p, "rb") as f:
        while True:
            b = f.read(CHUNK)
            if not b:
                break
            c = zlib.crc32(b, c); m.update(b)
    return c & 0xFFFFFFFF, m.hexdigest()


def fetch(url, a, n, dest):
    c = 0; m = hashlib.md5(); got = 0
    with open(dest + ".part", "wb") as o:
        while got < n:
            k = min(CHUNK, n - got)
            for t in range(10):
                try:
                    req = urllib.request.Request(url, headers={"Range": "bytes=%d-%d" % (a + got, a + got + k - 1)})
                    with urllib.request.urlopen(req, timeout=180) as r:
                        b = r.read()
                    if len(b) != k:
                        raise IOError("short read %d of %d" % (len(b), k))
                    break
                except Exception as e:
                    print("  retry", t, repr(e), flush=True)
                    time.sleep(5 * (t + 1))
            else:
                raise RuntimeError("download failed at %d" % (a + got))
            o.write(b); c = zlib.crc32(b, c); m.update(b); got += len(b)
    return c & 0xFFFFFFFF, m.hexdigest()


def main():
    idx = json.load(open(IDX))
    for case in sys.argv[1:]:
        c = idx["cases"][case]
        mem = idx["members"][c["ct_member"]]
        assert mem["method"] == 0 and mem["packed"] == mem["usize"], "member not stored"
        d = os.path.join(ROOT, case); os.makedirs(d, exist_ok=True)
        dest = os.path.join(d, "ct.nii.gz")
        rec = dict(case=case, group=c["group"], url=c["url"], member=c["ct_member"], data_offset=mem["data_offset"],
                   size=mem["packed"], crc32_expected="%08x" % mem["crc32"])
        t0 = time.time()
        local = "/sp/va_pool/data/samples/song_cta_images_head_%s.nii.gz" % case
        if os.path.exists(dest) and os.path.getsize(dest) == mem["packed"]:
            crc, md5 = crc_md5_file(dest); rec["source"] = "existing file re-verified"
        elif os.path.exists(local) and os.path.getsize(local) == mem["packed"]:
            crc, md5 = crc_md5_file(local)
            if crc == mem["crc32"]:
                shutil.copyfile(local, dest + ".part"); os.replace(dest + ".part", dest)
                rec["source"] = "copied from va_pool/data/samples (earlier range-read of the same member)"
            else:
                crc, md5 = fetch(c["url"], mem["data_offset"], mem["packed"], dest)
                rec["source"] = "range-read"
        else:
            crc, md5 = fetch(c["url"], mem["data_offset"], mem["packed"], dest)
            rec["source"] = "range-read"
        rec["crc32_got"] = "%08x" % crc; rec["md5"] = md5; rec["seconds"] = round(time.time() - t0, 1)
        rec["crc_ok"] = crc == mem["crc32"]
        if os.path.exists(dest + ".part"):
            if rec["crc_ok"]:
                os.replace(dest + ".part", dest)
            else:
                os.replace(dest + ".part", dest + ".bad")
        rec["date"] = time.strftime("%Y-%m-%d %H:%M")
        json.dump(rec, open(os.path.join(d, "fetch.json"), "w"), indent=1)
        print(case, rec["source"], rec["size"], "crc_ok", rec["crc_ok"], "%.0fs" % rec["seconds"], flush=True)


if __name__ == "__main__":
    main()
