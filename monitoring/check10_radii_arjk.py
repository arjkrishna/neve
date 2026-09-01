import os,json,numpy as np,collections,itertools,random
R=r"D:\Arjun\workspace\neve\carotid_data\anatomies"
H=r"D:\Arjun\workspace\neve\eve_bench\data\dualdevicenav\Centrelines_comb"
def load(p):
    m=json.load(open(p))['markups'][0]
    P=np.array([c['position'] for c in m['controlPoints']],float)
    r=None
    for meas in m['measurements']:
        if meas['name']=='Radius': r=np.array(meas['controlPointValues'],float)
    s=np.r_[0.0,np.cumsum(np.linalg.norm(np.diff(P,axis=0),axis=1))]
    return s,r
an=sorted(os.listdir(R))
pr={a:json.load(open(os.path.join(R,a,'provenance.json'))) for a in an}
hs,hr=load(os.path.join(H,'Centerline curve - RCCA.mrk.json'))
g=np.arange(0,200,0.25); hg=np.interp(g,hs,hr)
fd=[];rmin=[];r40=[]
D={}
for a in an:
    s,r=load(os.path.join(R,a,'Centrelines_comb','Centerline curve - RCCA.mrk.json'))
    D[a]=(s,r)
    gg=g[g<=min(s[-1],hs[-1])]; ri=np.interp(gg,s,r)
    d=np.abs(ri-hg[:len(gg)])
    i=np.nonzero(d>0.1)[0]; fd.append(gg[i[0]] if len(i) else np.nan)
    rmin.append(r.min()); r40.append(np.interp(np.arange(40,s[-1],0.25),s,r).min())
fd=np.array(fd);rmin=np.array(rmin);r40=np.array(r40)
print("RCCA declared-radius first >0.1mm divergence from host, mm: min %.1f p10 %.1f med %.1f p90 %.1f max %.1f nan=%d"%(
 np.nanmin(fd),np.nanpercentile(fd,10),np.nanmedian(fd),np.nanpercentile(fd,90),np.nanmax(fd),np.isnan(fd).sum()))
print("RCCA declared min radius over whole branch (mm): min %.3f med %.3f max %.3f"%(rmin.min(),np.median(rmin),rmin.max()))
print("RCCA declared min radius for s>=40 (mm): min %.3f p10 %.3f med %.3f"%(r40.min(),np.percentile(r40,10),np.median(r40)))
print("host RCCA declared min radius: %.3f"%hr.min())
# radius identical within same-lower group proximally?
bylo=collections.defaultdict(list)
for a in an: bylo[pr[a]['lower']].append(a)
random.seed(3); ps=[p for k in bylo for p in itertools.combinations(bylo[k],2)]; random.shuffle(ps)
v=[]
for x,y in ps[:150]:
    sx,rx=D[x];sy,ry=D[y]
    gg=np.arange(0,min(sx[-1],sy[-1]),0.25)
    d=np.abs(np.interp(gg,sx,rx)-np.interp(gg,sy,ry))
    i=np.nonzero(d>0.1)[0]; v.append(gg[i[0]] if len(i) else np.nan)
v=np.array(v)
print("same-lower siblings: declared-radius first >0.1mm divergence, mm: min %.1f med %.1f max %.1f"%(np.nanmin(v),np.nanmedian(v),np.nanmax(v)))
