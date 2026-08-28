import json, numpy as np
D = json.load(open(r"D:\Arjun\workspace\neve\monitoring\out_rf\refute_dilate.json"))
NQ = json.load(open(r"D:\Arjun\workspace\neve\monitoring\out_nq\nav_quality.json"))
COH = [k for k in D if k.startswith("topcow")]
print("=== D. IS THE COHORT DEFLATION AN OFFSET OR A SCALE? ===")
print("pool all cohort stations 20<s<end-8, bin by stated_r; if OFFSET, tube_dev flat vs r; if SCALE, tube_dev ~ -c*r")
R, TD = [], []
for t in COH:
    o = D[t]; S = np.array(o["S"]); m = (S > 20) & (S < S[-1]-8)
    R.append(np.array(o["R"])[m]); TD.append(np.array(o["tube_dev"])[m])
R = np.concatenate(R); TD = np.concatenate(TD); ok = np.isfinite(TD)
R, TD = R[ok], TD[ok]
print(" %8s %6s %9s %9s" % ("r bin", "n", "tube_dev", "dev/r"))
for lo, hi in [(1.0,1.6),(1.6,1.9),(1.9,2.2),(2.2,2.5),(2.5,2.8),(2.8,3.6)]:
    m = (R>=lo)&(R<hi)
    if m.sum()<30: continue
    print(" %3.1f-%3.1f %6d %+9.3f %+9.3f" % (lo,hi,m.sum(),np.median(TD[m]),np.median(TD[m]/R[m])))
A = np.vstack([np.ones_like(R), R]).T
c = np.linalg.lstsq(A, TD, rcond=None)[0]
print(" least squares tube_dev = %+.4f %+.4f * r   (pure offset => slope 0; pure scale => intercept 0)" % (c[0], c[1]))
print(" pure-offset fit: %+.4f mm, residual sd %.4f | pure-scale fit: %+.4f * r, residual sd %.4f" %
      (TD.mean(), TD.std(), (TD*R).sum()/(R*R).sum(), (TD-((TD*R).sum()/(R*R).sum())*R).std()))
print(" voxel reference: mesher grid 0.6/0.9 mm -> half-voxel 0.30/0.45 mm")
print()
print("=== E. OPERATIONAL VERDICT AFTER CORRECTING THE MESHER OFFSET (+0.30 mm radius) ===")
DELTA = 0.30
rows=[]
for t in ["HOST_COLLISION","HOST_VISUAL"]+COH:
    if t not in NQ: continue
    p = NQ[t]["profile"]; S=np.array(p["s"]); cl=np.array(p["clear"])
    m = S < S[-1]-8.0     # 8 mm distal trim, applied to every surface incl. host
    d = cl[m] + (DELTA if t.startswith("topcow") else 0.0)
    rows.append((t, np.median(cl[m]), np.median(d), d.min(), int((d<0.30).sum()), int((cl[m]<0.30).sum())))
hc=[r for r in rows if r[0]=="HOST_COLLISION"][0]; hv=[r for r in rows if r[0]=="HOST_VISUAL"][0]
co=[r for r in rows if r[0].startswith("topcow")]
print(" 8 mm distal trim on ALL surfaces; cohort clearance corrected by +%.2f mm (measured mesher offset)"%DELTA)
print(" %-16s %9s %9s %9s %7s"%("surface","clr_med","corrected","corr_min","n<0.30"))
print(" %-16s %9.3f %9.3f %9.3f %7d"%("HOST_COLLISION",hc[1],hc[2],hc[3],hc[4]))
print(" %-16s %9.3f %9.3f %9.3f %7d"%("HOST_VISUAL",hv[1],hv[2],hv[3],hv[4]))
print(" %-16s %9.3f %9.3f %9.3f %7d   (median of 22; raw n<0.30 total=%d, corrected total=%d)"%(
    "COHORT med",np.median([r[1] for r in co]),np.median([r[2] for r in co]),
    np.median([r[3] for r in co]),int(np.median([r[4] for r in co])),
    sum(r[5] for r in co),sum(r[4] for r in co)))
print(" cohort corrected min-clearance worst 4:", ", ".join("%s %.3f"%(r[0][-6:],r[3]) for r in sorted(co,key=lambda x:x[3])[:4]))
print(" cohort corrected median clearance vs HOST_VISUAL: %+.3f mm ; vs HOST_COLLISION: %+.3f mm"%(
    np.median([r[2] for r in co])-hv[2], np.median([r[2] for r in co])-hc[2]))
