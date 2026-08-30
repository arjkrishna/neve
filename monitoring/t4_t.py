import pickle, numpy as np, collections, math, json
D=pickle.load(open("_t4_rows2.pkl","rb")); T,H=D["T"],D["H"]
GEO=pickle.load(open("_t4_geom.pkl","rb"))
mh={r["seed"]:r for r in H}
gt=[r for r in T if r["grafted"]]
by=collections.defaultdict(list)
for r in gt: by[r["anat"]].append(r)
def logit_fit(X,y,it=200):
    X=np.asarray(X,float); y=np.asarray(y,float); b=np.zeros(X.shape[1])
    for _ in range(it):
        p=1/(1+np.exp(-X@b)); W=np.clip(p*(1-p),1e-9,None); z=X@b+(y-p)/W
        b=np.linalg.solve(X.T@(X*W[:,None])+1e-6*np.eye(X.shape[1]), X.T@(W*z))
    return b
u=lambda r:(r["tgt_s"]-183.3)/10.0
b=logit_fit([[1.,u(r),u(r)**2] for r in gt],[r["succ"] for r in gt])
pr=lambda r:1/(1+math.exp(-(b[0]+b[1]*u(r)+b[2]*u(r)**2)))
def wil(k,n,z=1.96):
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return (round(100*max(0,c-h),1),round(100*min(1,c+h),1))
out=[]
for a in sorted(by):
    g=by[a]; n=len(g); k=sum(r["succ"] for r in g)
    fl=[r for r in g if not r["succ"]]
    arr=sorted(round(r["max_s"],1) for r in fl)
    ni=sum(1 for r in fl if r["max_s"]<=130)
    out.append(dict(anat=a.replace("mr_mr","mr_"), n=n, k=k, rate=round(100*k/n,1), wilson=wil(k,n),
        heur_rate=round(100*np.mean([mh[r['seed']]['succ'] for r in g]),1),
        depth_pred_rate=round(100*np.mean([pr(r) for r in g]),1),
        tgt_s_med=round(float(np.median([r['tgt_s'] for r in g])),1),
        n_tgt_gt215=sum(1 for r in g if r['tgt_s']>215),
        succ_tgt_gt215=("%d/%d"%(sum(r['succ'] for r in g if r['tgt_s']>215),sum(1 for r in g if r['tgt_s']>215))) if any(r['tgt_s']>215 for r in g) else None,
        nfail=len(fl), arrest_s=arr, arrest_sd=round(float(np.std(arr)),1) if len(arr)>1 else None,
        fail_in_invariant_zone=ni, fail_distal=len(fl)-ni,
        fold_ge20_rate=round(100*np.mean([r['fold_max']>=20 for r in g]),1),
        heur_fold_ge20_rate=round(100*np.mean([mh[r['seed']]['fold_max']>=20 for r in g]),1),
        n_ostium_freeze=sum(1 for r in fl if r["max_s"]<25),
        disc_Tonly=sum(1 for r in g if r["succ"] and not mh[r["seed"]]["succ"]),
        disc_Honly=sum(1 for r in g if not r["succ"] and mh[r["seed"]]["succ"])))
json.dump(out,open("_t4_peranat.json","w"),indent=0)
for o in out: print(json.dumps(o))
