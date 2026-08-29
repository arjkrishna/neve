#!/usr/bin/env python3
"""Read selected members out of a remote ZIP without downloading the whole file.

The TopCoW release is a 10.5 GB archive of paired CTA and MRA images plus their
label masks. This pipeline only ever consumes the LABEL MASKS, which are a few
hundred kilobytes each, so pulling the whole archive to reach roughly 50 MB of
them would be absurd. Zenodo serves HTTP range requests, and a ZIP's central
directory lives at the END of the file, so the archive can be indexed remotely
and individual members fetched by byte range.

    python topbrain_tools/remote_zip.py <url> --list
    python topbrain_tools/remote_zip.py <url> --extract 'pattern' --out DIR
"""
import argparse
import fnmatch
import io
import os
import sys
import urllib.request
import zipfile


class HttpFile(io.RawIOBase):
    """Minimal seekable file over HTTP range requests, with a small cache.

    zipfile only needs seek/read/tell. Reads are served from a rolling block
    cache so that walking the central directory does not issue one request per
    struct field.
    """

    BLOCK = 1 << 20          # 1 MB

    def __init__(self, url, timeout=120):
        self.url = url
        self.timeout = timeout
        self._pos = 0
        self._cache = {}
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            self.size = int(r.headers["Content-Length"])
        self.requests = 0
        self.bytes_fetched = 0

    # -- io plumbing -----------------------------------------------------
    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self._pos

    def seek(self, off, whence=0):
        if whence == 0:
            self._pos = off
        elif whence == 1:
            self._pos += off
        else:
            self._pos = self.size + off
        return self._pos

    def _fetch(self, start, end):
        """Inclusive byte range."""
        req = urllib.request.Request(
            self.url, headers={"Range": "bytes=%d-%d" % (start, end)})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            data = r.read()
        self.requests += 1
        self.bytes_fetched += len(data)
        return data

    def _block(self, idx):
        if idx not in self._cache:
            start = idx * self.BLOCK
            end = min(start + self.BLOCK, self.size) - 1
            self._cache[idx] = self._fetch(start, end)
            if len(self._cache) > 64:            # bound the cache
                self._cache.pop(next(iter(self._cache)))
        return self._cache[idx]

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self._pos
        n = min(n, self.size - self._pos)
        if n <= 0:
            return b""
        # A large read (a whole member) goes straight out, uncached: caching
        # a 300 MB image would defeat the point.
        if n > 4 * self.BLOCK:
            data = self._fetch(self._pos, self._pos + n - 1)
            self._pos += len(data)
            return data
        out = bytearray()
        while len(out) < n:
            idx = self._pos // self.BLOCK
            off = self._pos % self.BLOCK
            blk = self._block(idx)
            take = min(n - len(out), len(blk) - off)
            if take <= 0:
                break
            out += blk[off:off + take]
            self._pos += take
        return bytes(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--extract", default=None,
                    help="fnmatch pattern of members to pull out")
    ap.add_argument("--out", default=".")
    ap.add_argument("--max-mb", type=float, default=500.0,
                    help="refuse to extract more than this in total")
    a = ap.parse_args()

    f = HttpFile(a.url)
    print("remote size %.1f GB" % (f.size / 1e9), flush=True)
    z = zipfile.ZipFile(f)
    names = z.namelist()
    print("members: %d (indexed with %d requests, %.1f MB fetched)"
          % (len(names), f.requests, f.bytes_fetched / 1e6), flush=True)

    if a.list:
        from collections import Counter
        tops = Counter(n.split("/")[0] for n in names)
        print("\ntop-level entries:")
        for k, v in sorted(tops.items()):
            print("   %-50s %d" % (k[:50], v))
        print("\nsample paths:")
        for n in names[:15]:
            print("   ", n)

    if a.extract:
        hits = [n for n in names if fnmatch.fnmatch(n, a.extract)]
        tot = sum(z.getinfo(n).file_size for n in hits)
        print("\nmatched %d members, %.1f MB uncompressed"
              % (len(hits), tot / 1e6), flush=True)
        if tot / 1e6 > a.max_mb:
            print("ABORT: exceeds --max-mb %.0f" % a.max_mb)
            return 1
        os.makedirs(a.out, exist_ok=True)
        for i, n in enumerate(hits, 1):
            dest = os.path.join(a.out, os.path.basename(n))
            if os.path.exists(dest):
                continue
            with z.open(n) as src, open(dest, "wb") as dst:
                dst.write(src.read())
            if i % 10 == 0 or i == len(hits):
                print("  %d/%d  (%.1f MB fetched)"
                      % (i, len(hits), f.bytes_fetched / 1e6), flush=True)
        print("extracted %d files to %s" % (len(hits), a.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
