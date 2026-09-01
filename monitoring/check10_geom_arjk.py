import os, json, collections
import numpy as np
R = r"D:\Arjun\workspace\neve\carotid_data\anatomies"
H = r"D:\Arjun\workspace\neve\eve_bench\data\dualdevicenav\Centrelines_comb"

def pos(p):
    m = json.load(open(p))['markups'][0]
    return np.array([c['position'] for c in m['controlPoints']], float)
def arclen(P):
    d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P,axis=0),axis=1))]
    return d
def resample(P, ds=0.25):
    s = arclen(P); L=s[-1]
    g = np.arange(0, L+1e-9, ds)
    out = np.column_stack([np.interp(g, s, P[:,i]) for i in range(3)])
    return g, out
def first_departure(A,B,tol=1.0,ds=0.25):
    """arclength (mm) of first point where resampled A,B differ by > tol"""
    ga,Ra = resample(A,ds); gb,Rb = resample(B,ds)
    n=min(len(ga),len(gb))
    d=np.linalg.norm(Ra[:n]-Rb[:n],axis=1)
    idx=np.nonzero(d>tol)[0]
    return (ga[idx[0]] if len(idx) else None), d[:n], ga[:n]

anats=sorted(os.listdir(R))
prov={a:json.load(open(os.path.join(R,a,'provenance.json'))) for a in anats}
RC={a:pos(os.path.join(R,a,'Centrelines_comb','Centerline curve - RCCA.mrk.json')) for a in anats}
RE={a:pos(os.path.join(R,a,'Centrelines_comb','Centerline curve - RECA.mrk.json')) for a in anats}
RV={a:pos(os.path.join(R,a,'Centrelines_comb','Centerline curve - RVA.mrk.json')) for a in anats}
hRC=pos(os.path.join(H,'Centerline curve - RCCA.mrk.json'))
hRV=pos(os.path.join(H,'Centerline curve - RVA.mrk.json'))

print("=== RCCA lengths (mm) ===")
Ls=np.array([arclen(RC[a])[-1] for a in anats])
print("total_len min/med/max: %.2f / %.2f / %.2f ; host %.2f" % (Ls.min(),np.median(Ls),Ls.max(),arclen(hRC)[-1]))

# --- byte-identical control point prefix across all 216 ---
ref=anats[0]
def cp_prefix(dic, keys):
    n=min(len(dic[k]) for k in keys); P0=dic[keys[0]]
    for i in range(n):
        for k in keys[1:]:
            if not np.array_equal(dic[k][i],P0[i]): return i
    return n
i=cp_prefix(RC,anats)
print("RCCA identical control-point prefix across all 216: %d pts, s=%.2f mm" % (i, arclen(RC[ref])[i-1] if i else 0))
# vs host
j=0; Pa=RC[ref]
while j<min(len(Pa),len(hRC)) and np.array_equal(Pa[j],hRC[j]): j+=1
print("RCCA identical control-point prefix (anat vs host): %d pts, s=%.2f mm" % (j, arclen(hRC)[j-1] if j else 0))

print("\n=== RCCA vs HOST: first >1mm departure ===")
fd=[]
for a in anats:
    s,_,_=first_departure(RC[a],hRC)
    fd.append(-1 if s is None else s)
fd=np.array(fd)
print("n never departing: %d" % (fd<0).sum())
v=fd[fd>=0]
print("first>1mm departure s (mm): min %.2f  p25 %.2f  med %.2f  p75 %.2f  max %.2f" % (v.min(),np.percentile(v,25),np.median(v),np.percentile(v,75),v.max()))
for t in (0.01,0.1,0.5):
    w=np.array([ (first_departure(RC[a],hRC,tol=t)[0] or -1) for a in anats])
    w=w[w>=0]
    print("  tol=%.2f mm: med first departure %.2f mm (min %.2f)" % (t, np.median(w), w.min()))

print("\n=== RCCA pairwise within/between lower-donor groups ===")
bylo=collections.defaultdict(list); bysi=collections.defaultdict(list)
for a in anats:
    bylo[prov[a]['lower']].append(a); bysi[prov[a]['siphon']].append(a)
import itertools, random
random.seed(0)
def sample_pairs(groups, same=True, n=300):
    ps=[]
    keys=list(groups)
    if same:
        for k in keys:
            g=groups[k]
            for x,y in itertools.combinations(g,2): ps.append((x,y))
    else:
        allp=[(x,y) for x,y in itertools.combinations(anats,2)]
        ps=[(x,y) for x,y in allp]
    random.shuffle(ps); return ps[:n]

for label,groups in (("SAME lower donor",bylo),("SAME siphon donor",bysi)):
    ps=sample_pairs(groups,True,250)
    r=[]
    for x,y in ps:
        s,_,_=first_departure(RC[x],RC[y])
        r.append(1e9 if s is None else s)
    r=np.array(r); nid=(r>=1e8).sum()
    rr=r[r<1e8]
    print("%s: n_pairs=%d identical_course=%d ; first>1mm dep min/med/max = %s" %
          (label,len(r),nid, "n/a" if len(rr)==0 else "%.2f / %.2f / %.2f"%(rr.min(),np.median(rr),rr.max())))
# different lower AND different siphon
ps=[]
for x,y in itertools.combinations(anats,2):
    if prov[x]['lower']!=prov[y]['lower'] and prov[x]['siphon']!=prov[y]['siphon']: ps.append((x,y))
random.shuffle(ps); ps=ps[:250]
r=np.array([ (first_departure(RC[x],RC[y])[0] or -1) for x,y in ps])
print("DIFFERENT lower & siphon: n=%d first>1mm dep min/med/max = %.2f / %.2f / %.2f" % (len(r),r.min(),np.median(r),r.max()))

print("\n=== RECA ===")
Le=np.array([arclen(RE[a])[-1] for a in anats])
print("RECA length mm min/med/max: %.2f / %.2f / %.2f" % (Le.min(),np.median(Le),Le.max()))
# RECA determined solely by lower donor?
ok=True
for k,g in bylo.items():
    hs={RE[a].tobytes() for a in g}
    if len(hs)!=1: ok=False; print("  lower %s has %d distinct RECA"%(k,len(hs)))
print("RECA constant within every lower-donor group:",ok)
# RECA start point vs RCCA (bifurcation location)
d0=[np.linalg.norm(RE[a][0]-RC[a][np.argmin(np.linalg.norm(RC[a]-RE[a][0],axis=1))]) for a in anats]
print("RECA origin distance to nearest RCCA station: min/med/max %.3f / %.3f / %.3f mm"%(min(d0),np.median(d0),max(d0)))
sb=[arclen(RC[a])[np.argmin(np.linalg.norm(RC[a]-RE[a][0],axis=1))] for a in anats]
sb=np.array(sb); print("bifurcation arclength on RCCA (mm): min/p25/med/p75/max %.1f/%.1f/%.1f/%.1f/%.1f"%(sb.min(),np.percentile(sb,25),np.median(sb),np.percentile(sb,75),sb.max()))

print("\n=== RVA ===")
same=[a for a in anats if np.array_equal(RV[a],hRV)]
diff=[a for a in anats if not np.array_equal(RV[a],hRV)]
print("RVA identical to host: %d ; deflected: %d"%(len(same),len(diff)))
dev=[]; span=[]
for a in diff:
    d=np.linalg.norm(RV[a]-hRV,axis=1)
    dev.append(d.max())
    s=arclen(hRV); m=d>0.01
    span.append((s[m][0],s[m][-1],(s[m][-1]-s[m][0])))
dev=np.array(dev); span=np.array(span)
print("max deviation mm: min/med/max %.3f / %.3f / %.3f"%(dev.min(),np.median(dev),dev.max()))
print("affected arclength start mm: min/med/max %.1f/%.1f/%.1f ; end %.1f/%.1f/%.1f ; span %.1f/%.1f/%.1f"%(
  span[:,0].min(),np.median(span[:,0]),span[:,0].max(),span[:,1].min(),np.median(span[:,1]),span[:,1].max(),
  span[:,2].min(),np.median(span[:,2]),span[:,2].max()))
print("RVA total length %.1f mm"%arclen(hRV)[-1])
rep=collections.Counter(prov[a].get('repairs','') for a in anats)
print("provenance 'repairs' values:", dict(rep))
