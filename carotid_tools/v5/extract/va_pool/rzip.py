# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\va_pool\rzip.py (= /sp/va_pool/rzip.py in the build containers)
"""Remote ZIP probe: list entries and read NIfTI headers via HTTP range requests.

usage: python3 rzip.py URL OUTJSON [--hdr PATTERN] [--max N] [--get NAME:DEST ...]
"""
import io, sys, json, zipfile, urllib.request, zlib, struct, re

class HttpFile(io.RawIOBase):
    def __init__(self, url):
        self.url = url
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=60) as r:
            self.size = int(r.headers["Content-Length"])
        self.pos = 0; self.nreq = 0; self.nbytes = 0
    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos
    def seek(self, off, whence=0):
        if whence == 0: self.pos = off
        elif whence == 1: self.pos += off
        else: self.pos = self.size + off
        return self.pos
    def read(self, n=-1):
        if n is None or n < 0: n = self.size - self.pos
        if n == 0 or self.pos >= self.size: return b""
        end = min(self.pos + n, self.size) - 1
        import time
        for attempt in range(8):
            try:
                req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
                with urllib.request.urlopen(req, timeout=120) as r:
                    data = r.read()
                break
            except Exception as e:
                if attempt == 7: raise
                time.sleep(15 * (attempt + 1))
        self.nreq += 1; self.nbytes += len(data)
        self.pos += len(data)
        return data
    def readinto(self, b):
        d = self.read(len(b)); b[:len(d)] = d; return len(d)

def nifti_hdr(raw):
    """parse a NIfTI-1 header from the first 352 bytes"""
    sz = struct.unpack("<i", raw[:4])[0]
    e = "<" if sz == 348 else ">"
    dim = struct.unpack(e + "8h", raw[40:56])
    datatype = struct.unpack(e + "h", raw[70:72])[0]
    pixdim = struct.unpack(e + "8f", raw[76:108])
    qform_code, sform_code = struct.unpack(e + "2h", raw[252:256])
    srow = struct.unpack(e + "12f", raw[280:328])
    descrip = raw[148:228].split(b"\0")[0].decode("latin1", "replace")
    n = dim[0]
    shape = list(dim[1:1 + n])
    sp = [round(float(p), 4) for p in pixdim[1:1 + n]]
    ext = [round(shape[i] * sp[i], 1) for i in range(min(3, n))]
    return dict(shape=shape, spacing=sp, extent_mm=ext, datatype=datatype,
                qform=qform_code, sform=sform_code,
                srow=[round(x, 3) for x in srow], descrip=descrip)

def read_member_head(z, name, nbytes=4096):
    with z.open(name) as m:
        raw = m.read(nbytes)
    if name.endswith(".gz"):
        d = zlib.decompressobj(31)
        raw = d.decompress(raw, 2048)
    return raw

if __name__ == "__main__":
    url, out = sys.argv[1], sys.argv[2]
    args = sys.argv[3:]
    pat = None; mx = 10 ** 9; gets = []
    i = 0
    while i < len(args):
        if args[i] == "--hdr": pat = re.compile(args[i + 1]); i += 2
        elif args[i] == "--max": mx = int(args[i + 1]); i += 2
        elif args[i] == "--get": gets.append(args[i + 1]); i += 2
        else: i += 1
    hf = HttpFile(url)
    f = io.BufferedReader(hf, buffer_size=1 << 16)
    z = zipfile.ZipFile(f)
    infos = z.infolist()
    res = dict(url=url, zip_size=hf.size, n_entries=len(infos),
               entries=[[x.filename, x.file_size, x.compress_size] for x in infos])
    hdrs = {}
    if pat:
        k = 0
        for x in infos:
            if pat.search(x.filename) and k < mx:
                try:
                    raw = read_member_head(z, x.filename)
                    hdrs[x.filename] = nifti_hdr(raw)
                except Exception as e:
                    hdrs[x.filename] = {"error": repr(e)}
                k += 1
                print(x.filename, hdrs[x.filename], flush=True)
    res["headers"] = hdrs
    for g in gets:
        name, dest = g.split("::")
        with z.open(name) as m, open(dest, "wb") as o:
            while True:
                b = m.read(1 << 24)
                if not b: break
                o.write(b)
        print("got", name, "->", dest, flush=True)
    res["http_requests"] = hf.nreq; res["http_bytes"] = hf.nbytes
    json.dump(res, open(out, "w"), indent=1)
    print("entries", len(infos), "requests", hf.nreq, "bytes", hf.nbytes)
