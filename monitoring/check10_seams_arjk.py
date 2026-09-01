import os,json,collections,itertools,random
import numpy as np
R=r"D:\Arjun\workspace\neve\carotid_data\anatomies"
H=r"D:\Arjun\workspace\neve\eve_bench\data\dualdevicenav\Centrelines_comb"
def pos(p):
    m=json.load(open(p))['markups'][0]
    return np.array([c['position'] for c in m['controlPoints']],float),m['coordinateSystem']
def arclen(P): return np.r_[0.0,np.cumsum(np.linalg.norm(np.diff(P,axis=0),axis=1))]
def rs(P,ds=0.25):
    s=arclen(P);g=np.arange(0,s[-1]+1e-9,ds)
    return g,np.column_stack([np.interp(g,s,P[:,i]) for i in range(3)])
def cdist_min(A,B):
    out=np.empty(len(A))
    for i0 in range(0,len(A),512):
        Ac=A[i0:i0+512]
        out[i0:i0+512]=np.sqrt(((Ac[:,None,:]-B[None,:,:])**2).sum(-1)).min(1)
    return out
def first_exceed(s,d,t):
    i=np.nonzero(d>t)[0]
    return s[i[0]] if len(i) else None
anats=sorted(os.listdir(R))
prov={a:json.load(open(os.path.join(R,a,'provenance.json'))) for a in anats}
RC={};CS=set()
for a in anats:
    P,cs=pos(os.path.join(R,a,'Centrelines_comb','Centerline curve - RCCA.mrk.json'));RC[a]=P;CS.add(cs)
hRC,hcs=pos(os.path.join(H,'Centerline curve - RCCA.mrk.json'))
print("coordinateSystem: anatomies",CS,"host",hcs)
G={a:rs(RC[a]) for a in anats}; hg,hR=rs(hRC)
bylo=collections.defaultdict(list)
for a in anats: bylo[prov[a]['lower']].append(a)
random.seed(2)

# SEAM 1: host arch -> lower donor.  first departure from HOST course
s1=[]
for a in anats:
    sA,A=G[a]; d=cdist_min(A,hR)
    s1.append(first_exceed(sA,d,0.5))
s1=np.array([x if x is not None else np.nan for x in s1])
print("\nSEAM1 (RCCA first >0.5mm departure from HOST course), mm: min %.1f p10 %.1f med %.1f p90 %.1f max %.1f ; n_nan=%d"%(
   np.nanmin(s1),np.nanpercentile(s1,10),np.nanmedian(s1),np.nanpercentile(s1,90),np.nanmax(s1),np.isnan(s1).sum()))
for t in (0.1,0.25,1.0,2.0):
    v=np.array([first_exceed(G[a][0],cdist_min(G[a][1],hR),t) or np.nan for a in random.sample(anats,60)])
    print("   tol %.2f mm: med %.1f  (min %.1f)"%(t,np.nanmedian(v),np.nanmin(v)))

# SEAM 2: lower donor -> siphon.  first departure between SAME-lower siblings
s2=[]
for k,g in bylo.items():
    for x,y in itertools.combinations(g,2):
        sA,A=G[x];d=cdist_min(A,G[y][1])
        v=first_exceed(sA,d,0.5)
        if v is not None: s2.append(v)
s2=np.array(s2)
print("\nSEAM2 (first >0.5mm departure between SAME-lower-donor siblings), mm, n_pairs=%d:"%len(s2))
print("   min %.1f p10 %.1f med %.1f p90 %.1f max %.1f"%(s2.min(),np.percentile(s2,10),np.median(s2),np.percentile(s2,90),s2.max()))

# shared fraction of the target pool (s>=40)
print("\n=== target pool s>=40 mm ===")
sh_host=[];sh_sib=[];Ls=[]
for a in anats:
    sA,A=G[a]; m=sA>=40.0; A2=A[m]; Ls.append(sA[-1])
    dh=cdist_min(A2,hR); sh_host.append((dh<0.5).mean())
    sib=[o for o in bylo[prov[a]['lower']] if o!=a]
    if sib:
        dm=np.full(m.sum(),1e9)
        for o in sib: dm=np.minimum(dm,cdist_min(A2,G[o][1]))
        sh_sib.append((dm<0.5).mean())
sh_host=np.array(sh_host);sh_sib=np.array(sh_sib);Ls=np.array(Ls)
print("frac of s>=40 within 0.5mm of HOST course: med %.4f  p90 %.4f  max %.4f"%(np.median(sh_host),np.percentile(sh_host,90),sh_host.max()))
print("frac of s>=40 within 0.5mm of a SAME-LOWER sibling: med %.3f p10 %.3f p90 %.3f (n=%d)"%(np.median(sh_sib),np.percentile(sh_sib,10),np.percentile(sh_sib,90),len(sh_sib)))
print("mean RCCA arclength beyond 40mm: %.1f mm ; unique-to-anatomy portion (beyond seam2 med %.1f): %.1f mm"%(
   (Ls-40).mean(), np.median(s2), (Ls-np.median(s2)).mean()))
