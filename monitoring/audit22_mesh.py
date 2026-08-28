"""STAGE 3: mesh integrity. STAGE 4: inter-vessel clearance."""
import sys, os, json
sys.path.insert(0,"/opt/eve_training/eve"); sys.path.insert(0,"/opt/eve_training/eve_bench")
import numpy as np, pyvista as pv, vtk
from scipy.spatial import cKDTree
from eve_bench.dualdevicenav import load_branches, DualDeviceNav

ROOT="/opt/eve_training/results_topbrain/anatomies"
RETAIN=["001","002","003","004","005","006","007","008","010","011","012","016","017","018","020","021","022","023","024","025","026","027"]
EXCL=["013","014","015"]
NAMES=["HOST"]+[f"topcow_mr_{n}" for n in RETAIN+EXCL]

def load(name):
    if name=="HOST":
        vt=DualDeviceNav().vessel_tree
        return pv.read(vt.mesh_path).triangulate().clean(), list(vt.branches)
    d0=os.path.join(ROOT,name)
    return (pv.read(os.path.join(d0,"vessel_architecture_collision.obj")).triangulate().clean(),
            load_branches(os.path.join(d0,"Centrelines_comb")))

def edges_of(kind,m):
    e=m.extract_feature_edges(boundary_edges=(kind=="b"),non_manifold_edges=(kind=="n"),
                              feature_edges=False,manifold_edges=False)
    return e.n_cells

def tri_tri_hits(v,f,maxpairs=4_000_000):
    """exact Moller-style tri-tri intersection over candidate pairs; skips pairs sharing a vertex."""
    cen=v[f].mean(1); rad=np.linalg.norm(v[f]-cen[:,None],axis=2).max(1)
    tree=cKDTree(cen); R=float(rad.max())
    pairs=tree.query_pairs(2*R,output_type='ndarray')
    if len(pairs)==0: return 0,0
    ok=(np.linalg.norm(cen[pairs[:,0]]-cen[pairs[:,1]],axis=1)<=rad[pairs[:,0]]+rad[pairs[:,1]])
    pairs=pairs[ok]
    share=np.array([len(set(f[a]).intersection(f[b]))>0 for a,b in pairs]) if len(pairs) else np.zeros(0,bool)
    pairs=pairs[~share]
    hits=0
    for a,b in pairs:
        T1=v[f[a]]; T2=v[f[b]]
        def sep(P,Q):
            n=np.cross(P[1]-P[0],P[2]-P[0]); nn=np.linalg.norm(n)
            if nn<1e-14: return False
            n=n/nn; d=(Q-P[0])@n
            return (d>1e-9).all() or (d<-1e-9).all()
        if sep(T1,T2) or sep(T2,T1): continue
        # edge-axis SAT
        sepd=False
        for E1 in [T1[1]-T1[0],T1[2]-T1[1],T1[0]-T1[2]]:
            for E2 in [T2[1]-T2[0],T2[2]-T2[1],T2[0]-T2[2]]:
                ax=np.cross(E1,E2); na=np.linalg.norm(ax)
                if na<1e-12: continue
                ax=ax/na; p=T1@ax; q=T2@ax
                if p.min()>q.max()+1e-9 or q.min()>p.max()+1e-9: sepd=True; break
            if sepd: break
        if not sepd: hits+=1
    return hits,len(pairs)

print("="*168)
print("C. MESH INTEGRITY  (collision surface, triangulated+cleaned)")
print("="*168)
print(f"{'anatomy':>15} {'grp':>4} {'npts':>7} {'ncells':>7} {'bnd_e':>6} {'nonman_e':>8} {'dupF':>5} {'degF':>5} "
      f"{'selfX':>6} {'ncomp':>5} {'compCells(top3)':>22} {'watertight':>10} {'RCCAcomp':>9} {'volMM3':>10}")
res={}
for name in NAMES:
    grp="HOST" if name=="HOST" else ("keep" if name[-3:] in RETAIN else "EXCL")
    m,brs=load(name)
    v=m.points; f=m.faces.reshape(-1,4)[:,1:]
    be=edges_of("b",m); ne=edges_of("n",m)
    key=np.sort(f,axis=1); _,idx,cnt=np.unique(key,axis=0,return_index=True,return_counts=True)
    dup=int((cnt>1).sum())
    a=0.5*np.linalg.norm(np.cross(v[f[:,1]]-v[f[:,0]],v[f[:,2]]-v[f[:,0]]),axis=1)
    deg=int(((a<1e-10)|(f[:,0]==f[:,1])|(f[:,1]==f[:,2])|(f[:,0]==f[:,2])).sum())
    conn=m.connectivity(); rid=conn.point_data["RegionId"]
    cid=conn.cell_data["RegionId"]
    ncomp=int(cid.max())+1
    sizes=np.bincount(cid); top=np.sort(sizes)[::-1][:3]
    sx,npair=tri_tri_hits(v,f)
    # which component contains the RCCA?
    rc=next(b for b in brs if "RCCA" in str(b.name).upper())
    C=np.asarray(rc.coordinates,float)
    kd=cKDTree(v); _,ii=kd.query(C)
    rcomp=sorted(set(rid[ii].tolist()))
    vol=float(m.volume)
    print(f"{name:>15} {grp:>4} {m.n_points:7d} {m.n_cells:7d} {be:6d} {ne:8d} {dup:5d} {deg:5d} "
          f"{sx:6d} {ncomp:5d} {str(list(top)):>22} {str(bool(be==0 and ne==0)):>10} {str(rcomp):>9} {vol:10.1f}")
    res[name]=dict(npts=int(m.n_points),ncells=int(m.n_cells),bnd=be,nonman=ne,dup=dup,deg=deg,
                   selfX=int(sx),npairs=int(npair),ncomp=ncomp,comp_sizes=[int(x) for x in np.sort(sizes)[::-1][:6]],
                   rcca_comps=[int(x) for x in rcomp],vol=vol)

print("\n"+"="*168)
print("D. INTER-VESSEL CLEARANCE: min over branch-pairs of  |cA-cB| - rA - rB   (negative = fused/interpenetrating)")
print("   restricted to RCCA arclength >= 135 mm (the grafted siphon; below that every branch shares the host trunk)")
print("="*168)
print(f"{'anatomy':>15} {'grp':>4} {'RCCAxRVA_min':>12} {'s@':>6} {'RCCAxLCCA':>10} {'RCCAxLVA':>9} "
      f"{'worst other branch':>26} {'gap':>7} {'#neg pairs(any branch, s>=135)':>30}")
gaps={}
for name in NAMES:
    grp="HOST" if name=="HOST" else ("keep" if name[-3:] in RETAIN else "EXCL")
    m,brs=load(name)
    rc=next(b for b in brs if "RCCA" in str(b.name).upper())
    C=np.asarray(rc.coordinates,float); Rr=np.asarray(rc.radii,float)
    S=np.concatenate(([0.],np.cumsum(np.linalg.norm(np.diff(C,axis=0),axis=1))))
    sel=S>=135.0
    Cs=C[sel]; Rs=Rr[sel]; Ss=S[sel]
    row={}
    for b in brs:
        nm=str(b.name)
        if "RCCA" in nm.upper(): continue
        B=np.asarray(b.coordinates,float); Rb=np.asarray(b.radii,float)
        D=np.linalg.norm(Cs[:,None,:]-B[None,:,:],axis=2)
        G=D-Rs[:,None]-Rb[None,:]
        j=np.unravel_index(np.argmin(G),G.shape)
        row[nm]=(float(G.min()),float(Ss[j[0]]))
    def get(k):
        for nm,(g,s) in row.items():
            if k in nm.upper(): return g,s
        return float('nan'),float('nan')
    gr,sr=get("RVA"); gl,sl=get("LCCA"); gv,sv=get("LVA")
    others={k:v for k,v in row.items() if not any(t in k.upper() for t in("RVA","LCCA","LVA"))}
    wk=min(others,key=lambda k:others[k][0]); wg=others[wk][0]
    nneg=sum(1 for g,_ in row.values() if g<0)
    print(f"{name:>15} {grp:>4} {gr:12.3f} {sr:6.1f} {gl:10.3f} {gv:9.3f} "
          f"{wk.replace('Centerline curve ',''):>26} {wg:7.3f} {nneg:30d}")
    gaps[name]={k:[round(v[0],4),round(v[1],1)] for k,v in row.items()}
json.dump({"integrity":res,"gaps":gaps},open("/opt/eve_training/results_topbrain/_audit_mesh.json","w"),indent=1)
print("\nwrote _audit_mesh.json")
