import glob, json, os, math
import numpy as np
ROOT="/d/Arjun/workspace/neve".replace("/d/","D:/",1) if os.name=="nt" else "/d/Arjun/workspace/neve"
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANAT=os.path.join(ROOT,"topbrain_data","anatomies")
HOSTD=os.path.join(ROOT,"eve_bench","data","dualdevicenav","Centrelines_comb")
RF="Centerline curve - RCCA.mrk.json"
SEAM=133.6
def j2b(p):
    x,y,z=p[...,0],p[...,1],p[...,2]
    return np.stack([y,-z,-x],axis=-1)
def read_curve(path):
    d=json.load(open(path,encoding="utf-8")); m=d["markups"][0]
    pos=np.array([cp["position"] for cp in m["controlPoints"]],float)
    rad=None
    for me in m.get("measurements",[]):
        if me.get("name")=="Radius" and "controlPointValues" in me:
            rad=np.array(me["controlPointValues"],float)
    return j2b(pos),rad
def arc(p): return np.concatenate([[0.0],np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))])
def resample(p,r,s0,s1,step):
    s=arc(p); n=max(int(round((s1-s0)/step))+1,3)
    t=np.linspace(s0,s1,n)
    q=np.stack([np.interp(t,s,p[:,i]) for i in range(3)],axis=1)
    rr=np.interp(t,s,r) if r is not None else None
    return q,rr,t
def menger(p):
    a,b,c=p[:-2],p[1:-1],p[2:]
    ab=np.linalg.norm(b-a,axis=1); bc=np.linalg.norm(c-b,axis=1); ca=np.linalg.norm(a-c,axis=1)
    ar=0.5*np.linalg.norm(np.cross(b-a,c-a),axis=1); den=ab*bc*ca
    return np.where(den>1e-12,4*ar/np.maximum(den,1e-12),0.0)
def tangents(p):
    d=np.diff(p,axis=0); n=np.linalg.norm(d,axis=1,keepdims=True)
    return d/np.maximum(n,1e-12)
def bends(p):
    t=tangents(p)
    return np.degrees(np.arccos(np.clip((t[:-1]*t[1:]).sum(1),-1,1)))
def torsion(p):
    # discrete torsion at 8mm nodes: angle between consecutive binormals / arc
    t=tangents(p); b=np.cross(t[:-1],t[1:])
    nb=np.linalg.norm(b,axis=1,keepdims=True); ok=(nb[:,0]>1e-9)
    bu=b/np.maximum(nb,1e-12)
    ang=np.degrees(np.arccos(np.clip((bu[:-1]*bu[1:]).sum(1),-1,1)))
    m=ok[:-1]&ok[1:]
    return ang[m]
def planarity(p):
    q=p-p.mean(0); s=np.linalg.svd(q,compute_uv=False)
    return s[2]/max(s[0],1e-12), s[2]   # ratio, rms-ish out-of-plane extent
def prof(p,r,L,name):
    o={}
    g0,g1=SEAM,L
    # ---- fine grid 0.5mm: calibre + tortuosity
    q5,r5,t5=resample(p,r,g0,g1,0.5)
    o["graft_len"]=g1-g0; o["rcca_len"]=L
    ch=np.linalg.norm(q5[-1]-q5[0]); o["graft_chord"]=ch; o["tort_graft"]=(g1-g0)/max(ch,1e-9)
    # sub-segment tortuosity (halves, thirds, max over 40mm windows)
    def tt(a,b):
        m=(t5>=a)&(t5<=b)
        if m.sum()<3: return float("nan")
        sub=q5[m]; return (t5[m][-1]-t5[m][0])/max(np.linalg.norm(sub[-1]-sub[0]),1e-9)
    mid=(g0+g1)/2
    o["tort_h1"]=tt(g0,mid); o["tort_h2"]=tt(mid,g1)
    o["tort_t1"]=tt(g0,g0+(g1-g0)/3); o["tort_t2"]=tt(g0+(g1-g0)/3,g0+2*(g1-g0)/3); o["tort_t3"]=tt(g0+2*(g1-g0)/3,g1)
    w=40.0; ws=[tt(a,a+w) for a in np.arange(g0,g1-w+1e-6,5.0)]
    o["tort_w40max"]=float(np.nanmax(ws)) if ws else float("nan")
    if r5 is not None:
        o["r_min"]=float(r5.min()); o["r_p05"]=float(np.percentile(r5,5)); o["r_p25"]=float(np.percentile(r5,25))
        o["r_med"]=float(np.median(r5)); o["r_term"]=float(r5[-1])
        o["r_n_lt_066"]=int((r5<0.66).sum())   # 0.35 catheter + 0.31 mesh offset
        o["r_n_lt_061"]=int((r5<0.61).sum())   # 0.30 SOFA + 0.31
    # ---- coarse grid 8mm: curvature / turning / shape
    q8,_,t8=resample(p,r,g0,g1,8.0)
    k=menger(q8); Rc=1.0/np.maximum(k,1e-9)
    o["n8"]=len(q8)
    o["Rc_min"]=float(Rc.min()); o["Rc_p05"]=float(np.percentile(Rc,5))
    o["Rc_p25"]=float(np.percentile(Rc,25)); o["Rc_med"]=float(np.median(Rc))
    o["n_Rc_lt5"]=int((Rc<5).sum()); o["n_Rc_lt8"]=int((Rc<8).sum()); o["n_Rc_lt12"]=int((Rc<12).sum())
    o["f_Rc_lt12"]=float((Rc<12).mean())
    bd=bends(q8)
    o["bend_max"]=float(bd.max()); o["bend_p90"]=float(np.percentile(bd,90))
    o["turn_cum"]=float(bd.sum()); o["turn_per_mm"]=float(bd.sum()/(g1-g0))
    # net vs total direction change
    t_=tangents(q8)
    o["turn_net"]=float(np.degrees(np.arccos(np.clip(np.dot(t_[0],t_[-1]),-1,1))))
    o["turn_eff"]=o["turn_net"]/max(o["turn_cum"],1e-9)
    # inflections: reversal of binormal direction (>90 deg) at 8mm
    t2=tangents(q8); b=np.cross(t2[:-1],t2[1:]); nb=np.linalg.norm(b,axis=1,keepdims=True)
    bu=b/np.maximum(nb,1e-12); dot=(bu[:-1]*bu[1:]).sum(1)
    o["n_infl"]=int((dot<0).sum())
    tor=torsion(q8)
    o["tors_mean"]=float(tor.mean()) if len(tor) else float("nan")
    o["tors_cum"]=float(tor.sum()) if len(tor) else float("nan")
    pr,oop=planarity(q8); o["planarity"]=float(pr); o["oop_extent"]=float(oop)
    # bend structure: fraction of turning in the busiest 24mm window; n of >=30deg runs
    cum=np.concatenate([[0],np.cumsum(bd)]); sm=t8[1:-1] if len(t8)>2 else t8
    tb=t8[:len(bd)]
    best=0.0
    for a in np.arange(g0,g1-24+1e-6,4.0):
        m=(tb>=a)&(tb<a+24); best=max(best,bd[m].sum())
    o["turn_top24"]=float(best); o["frac_top24"]=float(best/max(bd.sum(),1e-9))
    return o

rows={}
host_p,host_r=read_curve(os.path.join(HOSTD,RF))
rows["HOST"]=prof(host_p,host_r,arc(host_p)[-1],"HOST")
names=sorted(os.path.basename(d) for d in glob.glob(os.path.join(ANAT,"topcow_mr_*")))
EX={"topcow_mr_013","topcow_mr_014","topcow_mr_015"}
seam_rms={}
for nm in names:
    if nm in EX: continue
    p,r=read_curve(os.path.join(ANAT,nm,"Centrelines_comb",RF))
    L=arc(p)[-1]
    rows[nm]=prof(p,r,L,nm)
    # seam agreement with host over 0..133.6
    a,_,_=resample(p,r,0,SEAM,0.5); b,_,_=resample(host_p,host_r,0,SEAM,0.5)
    seam_rms[nm]=float(np.sqrt(((a-b)**2).sum(1).mean()))
json.dump({"rows":rows,"seam_rms":seam_rms},open(os.path.join(ROOT,"monitoring","h2diff_geom.json"),"w"),indent=1)
print("seam rms vs host (0-133.6mm): max=%.4f mm over %d anatomies"%(max(seam_rms.values()),len(seam_rms)))
ks=["rcca_len","graft_len","tort_graft","Rc_min","Rc_p05","Rc_med","n_Rc_lt5","n_Rc_lt8","n_Rc_lt12","bend_max","turn_cum","turn_per_mm","n_infl","tors_cum","planarity","r_min","r_p05","r_med","frac_top24"]
print("%-14s"%"anat"+"".join("%9s"%k[:9] for k in ks))
for nm in ["HOST"]+[n for n in rows if n!="HOST"]:
    print("%-14s"%nm.replace("topcow_",""),end="")
    print("".join("%9.3f"%rows[nm][k] for k in ks))
