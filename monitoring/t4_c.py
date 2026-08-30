import pickle, numpy as np, collections
D=pickle.load(open("_t4_rows.pkl","rb")); T,H=D["T"],D["H"]
mh={r["seed"]:r for r in H}
gt=[r for r in T if r["grafted"]]
f=[r for r in gt if not r["succ"]]
SEAM=133.6
def band(s):
    if s<SEAM-5: return "pre-seam"
    if s<=SEAM+5: return "SEAM(128.6-138.6)"
    if s<167: return "138.6-167"
    if s<=200: return "167-200"
    return ">200"
print("=== PER FAILED GRAFTED EPISODE (teacher) ===")
print("anat    seed   tgt_s  maxS  short  tailmedS  fold  slk  push/pull/hold  band            Hsucc HmaxS")
for r in sorted(f, key=lambda x:(x["anat"], x["max_s"])):
    h=mh[r["seed"]]
    print(f"{r['anat']} {r['seed']} {r['tgt_s']:6.1f} {r['max_s']:6.1f} {r['shortfall']:6.1f} {r['tail_med_s']:7.1f} "
          f"{r['fold_max']:4d} {r['slack_max']:5.1f}  {r['push_frac']:.2f}/{r['pull_frac']:.2f}/{r['hold_frac']:.2f} "
          f"{band(r['max_s']):17s} {int(h['succ'])} {h['max_s']:6.1f}")
print()
ms=np.array([r["max_s"] for r in f])
print("arrest depth (max_s) of 55 failures: min %.1f p25 %.1f med %.1f p75 %.1f max %.1f"%(
    ms.min(),np.percentile(ms,25),np.median(ms),np.percentile(ms,75),ms.max()))
print("shortfall: med %.1f  p25 %.1f p75 %.1f"%(np.median([r['shortfall'] for r in f]),
      np.percentile([r['shortfall'] for r in f],25),np.percentile([r['shortfall'] for r in f],75)))
print(collections.Counter(band(r["max_s"]) for r in f))
# 10mm histogram
hist=collections.Counter(int(r["max_s"]//10)*10 for r in f)
print("arrest hist (10mm bins):", sorted(hist.items()))
