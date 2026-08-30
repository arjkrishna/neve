import pickle, numpy as np, collections
D=pickle.load(open("_t4_rows2.pkl","rb")); T,H=D["T"],D["H"]
mh={r["seed"]:r for r in H}
GEO=pickle.load(open("_t4_geom.pkl","rb"))
G2=pickle.load(open("_t2_geo.pkl","rb"))["G"]
def key(a): return "topcow_"+a.replace("mr_mr","mr_")
def key2(a): return "topcow"+a.replace("mr_mr","mr")
def at(a,s,f):
    g=GEO[key(a)]; return float(np.interp(s,g["q"],g[f]))
def clr(a,s):
    v=G2[key2(a)]; return float(np.interp(s,np.array(v["g"]),np.array(v["d"])))
gt=[r for r in T if r["grafted"]]; fai=[r for r in gt if not r["succ"]]
print("=== geometry AT arrest station, per failure ===")
print("anat     seed    arrestS   r_decl clear  Rc(2mm) Rc(5mm) bend5 bend10   local rank of Rc5 in own 20-200 band")
for r in sorted(fai,key=lambda x:x["max_s"]):
    a=r["anat"]; s=r["max_s"]
    g=GEO[key(a)]; m=(g["q"]>=20)&(g["q"]<=min(200,g["L"]-6))
    rc5=at(a,s,"Rc5"); pct=100*float(np.nanmean(np.array(g["Rc5"])[m]<rc5))
    print(f"{a} {r['seed']} {s:8.1f} {at(a,s,'r'):7.2f} {clr(a,s):6.2f} {at(a,s,'Rc2'):7.1f} {rc5:7.1f} "
          f"{at(a,s,'b5'):6.1f} {at(a,s,'b10'):6.1f}   Rc5 pct={pct:5.1f}")
print()
# null: geometry at arrest vs geometry at matched-depth stations in SUCCESSFUL episodes / random
def stats(vals,lab):
    v=np.array([x for x in vals if np.isfinite(x)])
    print(f"  {lab:26s} n={len(v):3d} med {np.median(v):7.2f} p10 {np.percentile(v,10):7.2f} p90 {np.percentile(v,90):7.2f}")
for f,lab in (("r","r_decl"),("Rc5","Rc5 mm"),("b10","bend10 deg")):
    print(lab)
    stats([at(r["anat"],r["max_s"],f) for r in fai],"at arrest (55 fails)")
    # null: all stations 20..200 pooled over the same anatomies weighted equally
    nul=[]
    for r in fai:
        g=GEO[key(r["anat"])]; m=(g["q"]>=20)&(g["q"]<=200)
        nul.append(np.nanmedian(np.array(g[f])[m]))
    stats(nul,"anat median 20-200")
print("clearance")
stats([clr(r["anat"],r["max_s"]) for r in fai],"at arrest")
