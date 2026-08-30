import json, statistics as stx
from collections import Counter
import sys
sys.path.insert(0,"D:/Arjun/workspace/neve/monitoring")
from cell2_report import load, pct, q

A,_ = load("A_all22"); H,_ = load("H_all22")
a = {e["seed"]: e for e in A}; h = {e["seed"]: e for e in H}
sh = set(a) & set(h)
print("seeds A=%d H=%d shared=%d ; anatomy match=%d/%d ; path_len match=%d" % (
  len(a),len(h),len(sh),
  sum(1 for s in sh if a[s]["anat"]==h[s]["anat"]), len(sh),
  sum(1 for s in sh if abs(a[s]["pl"]-h[s]["pl"])<1e-6)))

def st(e): return bool(e["ev"]["cmd12"])
def unrec(e): return any(k["k"]=="unrec" for k in e["ev"]["cmd12"])

for label, sel in (("ALL", lambda s: True),
                   ("SHARED", lambda s: a[s]["pl"]<=166.91),
                   ("GRAFTED", lambda s: a[s]["pl"]>166.91)):
    ss = [s for s in sorted(sh) if sel(s)]
    t = Counter()
    for s in ss:
        t[(h[s]["succ"], a[s]["succ"])] += 1
    print("\n== %s n=%d  H=%d/%d (%s)  A=%d/%d (%s)" % (label, len(ss),
        sum(h[s]["succ"] for s in ss), len(ss), pct(sum(h[s]["succ"] for s in ss),len(ss)),
        sum(a[s]["succ"] for s in ss), len(ss), pct(sum(a[s]["succ"] for s in ss),len(ss))))
    print("   transitions H->A: fail->succ=%d  succ->fail=%d  succ->succ=%d  fail->fail=%d" % (
        t[(False,True)], t[(True,False)], t[(True,True)], t[(False,False)]))
    # cross-tab transition x stall status
    print("   %-14s | H_stall A_stall | H_unrec A_unrec | medlen H/A | maxp H/A | gwmax H/A | slack H/A" % "transition")
    for nm,(hv,av) in (("fail->succ",(False,True)),("succ->fail",(True,False)),
                       ("succ->succ",(True,True)),("fail->fail",(False,False))):
        g = [s for s in ss if h[s]["succ"]==hv and a[s]["succ"]==av]
        if not g: continue
        print("   %-14s n=%3d |  %5.0f%%  %5.0f%%  |  %5.0f%%  %5.0f%%  | %4.0f/%4.0f | %5.1f/%5.1f | %5.1f/%5.1f | %4.1f/%4.1f" % (
          nm, len(g),
          100*sum(st(h[s]) for s in g)/len(g), 100*sum(st(a[s]) for s in g)/len(g),
          100*sum(unrec(h[s]) for s in g)/len(g), 100*sum(unrec(a[s]) for s in g)/len(g),
          stx.median([h[s]["steps"] for s in g]), stx.median([a[s]["steps"] for s in g]),
          stx.median([h[s]["maxp"] for s in g]), stx.median([a[s]["maxp"] for s in g]),
          stx.median([h[s]["gw_max"] for s in g]), stx.median([a[s]["gw_max"] for s in g]),
          stx.median([h[s]["maxslack"] for s in g]), stx.median([a[s]["maxslack"] for s in g])))
    # paired stall counts
    dh = sum(len(h[s]["ev"]["cmd12"]) for s in ss); da = sum(len(a[s]["ev"]["cmd12"]) for s in ss)
    print("   paired stalls: H=%d A=%d ; eps stalled H=%d A=%d ; both=%d neither=%d Honly=%d Aonly=%d" % (
      dh, da, sum(st(h[s]) for s in ss), sum(st(a[s]) for s in ss),
      sum(1 for s in ss if st(h[s]) and st(a[s])), sum(1 for s in ss if not st(h[s]) and not st(a[s])),
      sum(1 for s in ss if st(h[s]) and not st(a[s])), sum(1 for s in ss if not st(h[s]) and st(a[s]))))
    # progress reached
    print("   max proj_s reached  H med=%.1f A med=%.1f ; frac reaching >=path_len-33.31: H=%s A=%s" % (
      stx.median([h[s]["maxp"] for s in ss]), stx.median([a[s]["maxp"] for s in ss]),
      pct(sum(1 for s in ss if h[s]["maxp"]>=h[s]["pl"]-33.31), len(ss)),
      pct(sum(1 for s in ss if a[s]["maxp"]>=a[s]["pl"]-33.31), len(ss))))
    # failure reasons
    print("   A fail reasons: %s" % Counter(a[s]["reason"] for s in ss if not a[s]["succ"]).most_common())
    print("   H fail reasons: %s" % Counter(h[s]["reason"] for s in ss if not h[s]["succ"]).most_common())

# per-anatomy grafted breakdown
print("\n== grafted per-mesh (A succ / n, stalls/ep A, unrec-rate A)")
gg = [s for s in sorted(sh) if a[s]["pl"]>166.91]
by = {}
for s in gg: by.setdefault(a[s]["mfp"], []).append(s)
for m, ss in sorted(by.items(), key=lambda kv: -len(kv[1])):
    print("   %-14s n=%2d A=%s H=%s  A_st/ep=%.2f A_unrec=%s" % (m, len(ss),
      pct(sum(a[s]["succ"] for s in ss),len(ss)), pct(sum(h[s]["succ"] for s in ss),len(ss)),
      sum(len(a[s]["ev"]["cmd12"]) for s in ss)/len(ss),
      pct(sum(unrec(a[s]) for s in ss),len(ss))))
