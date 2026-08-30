import json, math, collections
import numpy as np
rng=np.random.default_rng(5)
M="monitoring/"; ROWS=json.load(open(M+"h2diff_eprows.json")); D=json.load(open(M+"h2diff_diff.json"))
anat=[r["anat"] for r in ROWS]; y=np.array([r["succ"] for r in ROWS],float); yh=np.array([r["hsucc"] for r in ROWS],float)
st=np.array([r["s_tgt"] for r in ROWS]); tu=np.array([r["turn_cum"] for r in ROWS])
def logit(X,yv,it=300):
    b=np.zeros(X.shape[1])
    for _ in range(it):
        p=1/(1+np.exp(-X@b)); W=p*(1-p)+1e-9
        d=np.linalg.solve(X.T@(X*W[:,None])+1e-6*np.eye(X.shape[1]),X.T@(yv-p)-1e-6*b); b+=d
        if np.max(np.abs(d))<1e-11: break
    return b
z=lambda v:(v-v.mean())/v.std()
one=np.ones(len(y))
bP=logit(np.stack([one,z(st)],1),y); bH=logit(np.stack([one,z(st)],1),yh)
pP=1/(1+np.exp(-np.stack([one,z(st)],1)@bP)); pH=1/(1+np.exp(-np.stack([one,z(st)],1)@bH))
by=collections.defaultdict(lambda:[0,0,0,0.,0.,[]])
for i,a in enumerate(anat):
    b=by[a]; b[0]+=1; b[1]+=y[i]; b[2]+=yh[i]; b[3]+=pP[i]; b[4]+=pH[i]; b[5].append(st[i])
print("=== observed vs DEPTH-ONLY-predicted grafted rate (policy) ===")
print("%-12s %3s %8s %8s %8s %8s %8s %8s"%("anat","n","s_mean","s_max","obs","pred_dep","resid","DIFFz"))
res=[]
for a in sorted(by,key=lambda a:-by[a][1]/by[a][0]):
    n,k,kh,sp,sh,S=by[a]
    print("%-12s %3d %8.1f %8.1f %8.3f %8.3f %+8.3f %+8.2f"%(a,n,np.mean(S),max(S),k/n,sp/n,k/n-sp/n,D["DIFF"][a]))
    res.append((a,n,k/n,sp/n))
# chi-square heterogeneity around the depth-only prediction
X2=sum((by[a][1]-by[a][3])**2/max(by[a][3]*(1-by[a][3]/by[a][0]),1e-9) for a in by)
print("  (anatomy-factor LRT | depth reported separately: policy G2=29.57 df=21 p=0.101)")
# weighted corr of anatomy mean depth and composite with rate
A=sorted(by); n=np.array([by[a][0] for a in A],float)
rate=np.array([by[a][1]/by[a][0] for a in A]); rateh=np.array([by[a][2]/by[a][0] for a in A])
smean=np.array([np.mean(by[a][5]) for a in A]); smax=np.array([max(by[a][5]) for a in A])
dv=np.array([D["DIFF"][a] for a in A])
def wc(x,yv,w):
    mx=np.average(x,weights=w); my=np.average(yv,weights=w)
    return np.average((x-mx)*(yv-my),weights=w)/math.sqrt(np.average((x-mx)**2,weights=w)*np.average((yv-my)**2,weights=w))
def perm_p(x,kvec,w,B=50000):
    N=int(w.sum()); K=int(kvec.sum()); pool=np.array([1]*K+[0]*(N-K)); cuts=np.cumsum(w).astype(int)[:-1]
    r0=abs(wc(x,kvec/w,w)); c=0
    for _ in range(B):
        rng.shuffle(pool); kb=np.array([s.sum() for s in np.split(pool,cuts)],float)
        if abs(wc(x,kb/w,w))>=r0-1e-12: c+=1
    return (c+1)/(B+1)
kp=np.array([by[a][1] for a in A],float); kh=np.array([by[a][2] for a in A],float)
print("\n=== single pre-specified anatomy-level tests (no scan) ===")
for nm,x in [("mean target depth",smean),("max target depth",smax),("COMPOSITE DIFF",dv)]:
    print("  POLICY  wcorr(%s, rate)=%+.3f perm p=%.4f | HEUR wcorr=%+.3f perm p=%.4f"
          %(nm,wc(x,rate,n),perm_p(x,kp,n),wc(x,rateh,n),perm_p(x,kh,n)))
print("\n  wcorr(mean depth, COMPOSITE DIFF) = %+.3f  <- depth sampling and geometry are confounded"%wc(smean,dv,n))
