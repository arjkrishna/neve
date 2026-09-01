import os,json,collections,itertools,random
import numpy as np
R=r"D:\Arjun\workspace\neve\carotid_data\anatomies"
H=r"D:\Arjun\workspace\neve\eve_bench\data\dualdevicenav\Centrelines_comb"
def pos(p):
    m=json.load(open(p))['markups'][0]
    return np.array([c['position'] for c in m['controlPoints']],float)
def arclen(P): return np.r_[0.0,np.cumsum(np.linalg.norm(np.diff(P,axis=0),axis=1))]
def rs(P,ds=0.25):
    s=arclen(P);g=np.arange(0,s[-1]+1e-9,ds)
    return g,np.column_stack([np.interp(g,s,P[:,i]) for i in range(3)])
def curve_dist(A,B):
    """min distance from each point of A to the 0.25mm-sampled polyline B.
    Point-sampled nearest neighbour; overestimates the true curve distance by <=0.125 mm."""
    out=np.empty(len(A))
    for i0 in range(0,len(A),256):
        Ac=A[i0:i0+256]
        D=np.sqrt(((Ac[:,None,:]-B[None,:,:])**2).sum(-1))
        out[i0:i0+256]=D.min(1)
    return out
anats=sorted(os.listdir(R))
prov={a:json.load(open(os.path.join(R,a,'provenance.json'))) for a in anats}
RC={a:pos(os.path.join(R,a,'Centrelines_comb','Centerline curve - RCCA.mrk.json')) for a in anats}
hRC=pos(os.path.join(H,'Centerline curve - RCCA.mrk.json'))
G={a:rs(RC[a]) for a in anats}; hg,hR=rs(hRC)

# --- envelope: spread across all 216 at each arclength ---
nmin=min(len(G[a][0]) for a in anats)
Sall=np.stack([G[a][1][:nmin] for a in anats])          # 216 x nmin x 3
cen=Sall.mean(0)
rad=np.linalg.norm(Sall-cen,axis=2)                      # 216 x nmin
mx=rad.max(0); md=np.median(rad,0)
gs=G[anats[0]][0][:nmin]
print("=== RCCA spread envelope across all 216 (index-matched, 0.25mm arclength grid) ===")
print("  s(mm) where MAX radius-from-mean first exceeds:")
for t in (0.1,0.25,0.5,1.0,2.0,5.0,10.0):
    i=np.nonzero(mx>t)[0]
    print("    >%5.2f mm : s=%s"%(t,"never" if not len(i) else "%.2f"%gs[i[0]]))
print("  s(mm) where MEDIAN radius-from-mean first exceeds:")
for t in (0.1,0.25,0.5,1.0,2.0,5.0):
    i=np.nonzero(md>t)[0]
    print("    >%5.2f mm : s=%s"%(t,"never" if not len(i) else "%.2f"%gs[i[0]]))
for s in (0,5,10,20,30,40,50,60,80,100,120,140,160,180,200):
    i=int(s/0.25)
    if i<nmin: print("    s=%3d mm: median spread %.3f  max spread %.3f"%(s,md[i],mx[i]))

# --- geometric (curve-to-curve) distance, removes reparameterization ---
random.seed(1)
def prof(pairs,bins):
    out={b:[] for b in bins}
    for x,y in pairs:
        A=G[x][1];sA=G[x][0];B=G[y][1]
        d=curve_dist(A,B)
        for b in bins:
            m=(sA>=b[0])&(sA<b[1])
            if m.any(): out[b].append(d[m].max())
    return out
bins=[(0,20),(20,40),(40,70),(70,100),(100,135),(135,170),(170,300)]
bylo=collections.defaultdict(list);bysi=collections.defaultdict(list)
for a in anats: bylo[prov[a]['lower']].append(a);bysi[prov[a]['siphon']].append(a)
samelo=[p for k in bylo for p in itertools.combinations(bylo[k],2)];random.shuffle(samelo)
samesi=[p for k in bysi for p in itertools.combinations(bysi[k],2)];random.shuffle(samesi)
allp=list(itertools.combinations(anats,2));random.shuffle(allp)
hostp=[(a,'HOST') for a in anats]
G['HOST']=(hg,hR)
print("\n=== max curve-to-curve separation (mm) per arclength band, median over pairs ===")
print("band          same-lower(n=200)  same-siphon(n=200)  all-pairs(n=200)  vs-HOST(n=216)")
P1=prof(samelo[:200],bins);P2=prof(samesi[:200],bins);P3=prof(allp[:200],bins);P4=prof(hostp,bins)
for b in bins:
    f=lambda P: ("%8.3f"%np.median(P[b])) if P[b] else "     n/a"
    print("%-13s %s %18s %18s %18s"%("%d-%d"%b,f(P1),f(P2),f(P3),f(P4)))

# --- fraction of target pool on shared geometry ---
print("\n=== target-pool geometry (sampler: RCCA, min_arclength_from_start=40) ===")
Ls=np.array([arclen(RC[a])[-1] for a in anats])
print("RCCA length beyond 40mm: min/med/max %.1f/%.1f/%.1f mm"%((Ls-40).min(),np.median(Ls-40),(Ls-40).max()))
# for each anatomy, fraction of s>40 where nearest other-anatomy course is <1mm away (i.e. non-unique)
fr=[]
for a in random.sample(anats,40):
    sA,A=G[a]; m=sA>=40; A2=A[m]
    others=random.sample([o for o in anats if o!=a],25)
    dmin=np.full(m.sum(),1e9)
    for o in others: dmin=np.minimum(dmin,curve_dist(A2,G[o][1]))
    fr.append((dmin<1.0).mean())
fr=np.array(fr)
print("frac of s>40mm RCCA within 1mm of >=1 of 25 random other anatomies: med %.3f (min %.3f max %.3f)"%(np.median(fr),fr.min(),fr.max()))
