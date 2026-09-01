import os,json,collections,numpy as np
R=r"D:\Arjun\workspace\neve\carotid_data\anatomies"
H=r"D:\Arjun\workspace\neve\eve_bench\data\dualdevicenav\Centrelines_comb"
def pos(p):
    m=json.load(open(p))['markups'][0]
    return np.array([c['position'] for c in m['controlPoints']],float)
def arclen(P): return np.r_[0.0,np.cumsum(np.linalg.norm(np.diff(P,axis=0),axis=1))]
def rs(P,ds=0.25):
    s=arclen(P);g=np.arange(0,s[-1]+1e-9,ds)
    return g,np.column_stack([np.interp(g,s,P[:,i]) for i in range(3)])
def cmin(A,B):
    o=np.empty(len(A))
    for i in range(0,len(A),512):
        o[i:i+512]=np.sqrt(((A[i:i+512,None,:]-B[None,:,:])**2).sum(-1)).min(1)
    return o
an=sorted(os.listdir(R))
pr={a:json.load(open(os.path.join(R,a,'provenance.json'))) for a in an}
RC={a:pos(os.path.join(R,a,'Centrelines_comb','Centerline curve - RCCA.mrk.json')) for a in an}
RE={a:pos(os.path.join(R,a,'Centrelines_comb','Centerline curve - RECA.mrk.json')) for a in an}
RV={a:pos(os.path.join(R,a,'Centrelines_comb','Centerline curve - RVA.mrk.json')) for a in an}
hR=rs(pos(os.path.join(H,'Centerline curve - RCCA.mrk.json')))[1]
hV=pos(os.path.join(H,'Centerline curve - RVA.mrk.json'))
rec=[]
for a in an:
    s,A=rs(RC[a]); d=cmin(A,hR)
    i=np.nonzero(d>0.5)[0]; s1=s[i[0]] if len(i) else np.nan
    m=s>=40; fh=(d[m]<0.5).mean()
    rec.append(dict(a=a,L=s[-1],s1=s1,fh=fh,Le=arclen(RE[a])[-1],
                    rv=np.linalg.norm(RV[a]-hV,axis=1).max()))
def top(k,rev,n=4): 
    r=sorted(rec,key=lambda x:x[k],reverse=rev)[:n]
    return "; ".join("%s %.2f"%(x['a'],x[k]) for x in r)
print("earliest seam1 :",top('s1',False))
print("latest   seam1 :",top('s1',True))
print("shortest RCCA  :",top('L',False))
print("longest  RCCA  :",top('L',True))
print("highest host-shared frac(s>=40):",top('fh',True))
print("shortest RECA  :",top('Le',False))
print("largest RVA deflection:",top('rv',True))
