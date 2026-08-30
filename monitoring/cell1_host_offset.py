"""CELL 1: derive the HOST s_RCCA offset independently (do NOT assume 33.31).
Method: cur_branch is the centerline segment nearest the guidewire tip. The RCCA
ostium arclength = the proj_s at which cur_branch switches off the pre-ostium
segment 'Centerline curve (11).mrk' onto 'Centerline curve - RCCA.mrk'.
Cross-check with the same procedure on a TOPBRAIN cohort run, where the answer
is known to be 33.31.
"""
import glob, json, os, re, statistics as stx, sys
PROJ = re.compile(r"proj_s=([-0-9.]+)")
CB   = re.compile(r"cur_branch=([^|]*)")
PL   = re.compile(r"path_len=([0-9.]+)")

def scan(L):
    out = []
    for p in sorted(glob.glob(os.path.join(L, "worker_*.log"))):
        live = {}
        def fin(st):
            if st["rcca"] and st["non"]:
                out.append((min(st["rcca"]), max(st["non"]), st["pl"]))
        for line in open(p, errors="replace"):
            i = line.find("pid="); pid = line[i+4:].split(" ")[0].strip() if i >= 0 else "?"
            if "EPISODE_START" in line:
                if pid in live: fin(live.pop(pid))
                live[pid] = {"rcca": [], "non": [], "pl": None}
                continue
            st = live.get(pid)
            if st is None: continue
            if "EPISODE_OUTCOME" in line:
                fin(live.pop(pid)); continue
            if " STEP |" not in line: continue
            m = PROJ.search(line); c = CB.search(line)
            if not (m and c): continue
            proj = float(m.group(1)); br = c.group(1).strip()
            if st["pl"] is None:
                q = PL.search(line)
                if q: st["pl"] = float(q.group(1))
            if ("RCCA" in br) or ("RVA" in br): st["rcca"].append(proj)
            else: st["non"].append(proj)
        for st in live.values(): fin(st)
    return out

def rep(tag, L):
    v = scan(L)
    if not v:
        print("%s : no episodes with both labels" % tag); return
    a = sorted(x[0] for x in v); b = sorted(x[1] for x in v)
    mid = sorted((x[0]+x[1])/2.0 for x in v)
    def q(s, p): return s[min(len(s)-1, int(p*len(s)))]
    print("%s  n=%d" % (tag, len(v)))
    print("   first proj_s ON RCCA branch : med=%.2f p10=%.2f p90=%.2f min=%.2f" % (stx.median(a), q(a,.1), q(a,.9), a[0]))
    print("   last  proj_s ON pre-ostium  : med=%.2f p10=%.2f p90=%.2f max=%.2f" % (stx.median(b), q(b,.1), q(b,.9), b[-1]))
    print("   OFFSET estimate (midpoint)  : med=%.2f p10=%.2f p90=%.2f" % (stx.median(mid), q(mid,.1), q(mid,.9)))

if __name__ == "__main__":
    for tag, L in [tuple(x.split("=", 1)) for x in sys.argv[1:]]:
        rep(tag, L)
