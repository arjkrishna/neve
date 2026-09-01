import json,os,math,random
from collections import defaultdict
SCR=r"C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-workspace-neve\81b186b6-3a3f-4f63-8491-2172316ef81f\scratchpad"
D=json.load(open(os.path.join(SCR,"ck15_defects.json")))
dn=json.load(open(os.path.join(SCR,"ck15_donors.json")))
LOW,SIP=dn["low"],dn["sip"]
names=sorted(LOW); N=len(names)
random.seed(20260901)

def hyper_sf(k,K,n,N):
    # P(X>=k), X~Hypergeom(N,K,n)
    tot=math.comb(N,n); s=0
    for i in range(k,min(K,n)+1):
        s+=math.comb(K,i)*math.comb(N-K,n-i)
    return s/tot

def groups(f):
    g=defaultdict(list)
    for nm in names: g[f[nm]].append(nm)
    return g

def conc(lab,grp):
    return sum(k*(k-1) for k in (sum(lab[nm] for nm in v) for v in grp.values()))

def perm_global(lab_list,sizes,obs,nperm=50000):
    arr=list(lab_list); hit=0; vals=[]
    for _ in range(nperm):
        random.shuffle(arr)
        i=0; s=0
        for sz in sizes:
            k=sum(arr[i:i+sz]); i+=sz; s+=k*(k-1)
        vals.append(s)
        if s>=obs: hit+=1
    mu=sum(vals)/nperm; sd=(sum((v-mu)**2 for v in vals)/nperm)**0.5
    return (hit+1)/(nperm+1), mu, sd

def strat_perm(lab,strat,grp,obs,nperm=50000):
    # permute labels WITHIN strata (e.g. within lower donor), recompute concordance on grp (siphon)
    blocks=defaultdict(list)
    for nm in names: blocks[strat[nm]].append(nm)
    idx={nm:i for i,nm in enumerate(names)}
    gidx=[[idx[nm] for nm in v] for v in grp.values()]
    base=[lab[nm] for nm in names]
    hit=0; vals=[]
    bl=[[idx[nm] for nm in v] for v in blocks.values()]
    for _ in range(nperm):
        cur=list(base)
        for b in bl:
            vs=[base[i] for i in b]; random.shuffle(vs)
            for i,v in zip(b,vs): cur[i]=v
        s=0
        for g in gidx:
            k=sum(cur[i] for i in g); s+=k*(k-1)
        vals.append(s)
        if s>=obs: hit+=1
    mu=sum(vals)/nperm; sd=(sum((v-mu)**2 for v in vals)/nperm)**0.5
    return (hit+1)/(nperm+1), mu, sd

GL=groups(LOW); GS=groups(SIP)
defects=[d for d in sorted(D) if 3<=sum(D[d].values())<=200]
NTEST=len(defects)*2
print("defects tested",len(defects),"global tests",NTEST,"Bonferroni alpha",0.05/NTEST)
out={}
print("\n%-26s %4s | %-32s | %-32s" % ("DEFECT","n","LOWER  obs/exp  p_perm","SIPHON obs/exp  p_perm"))
for d in defects:
    lab={nm:D[d][nm] for nm in names}
    K=sum(lab.values())
    res={}
    for fac,grp in (("low",GL),("sip",GS)):
        obs=conc(lab,grp)
        sizes=[len(v) for v in grp.values()]
        p,mu,sd=perm_global([lab[nm] for nm in names],sizes,obs)
        z=(obs-mu)/sd if sd>0 else 0.0
        res[fac]=dict(obs=obs,exp=round(mu,2),z=round(z,2),p=p)
    # stratified: siphon effect within lower, lower effect within siphon
    obs_s=conc(lab,GS); ps,mus,sds=strat_perm(lab,LOW,GS,obs_s,20000)
    obs_l=conc(lab,GL); pl,mul,sdl=strat_perm(lab,SIP,GL,obs_l,20000)
    res["sip_given_low"]=dict(p=ps,exp=round(mus,2),obs=obs_s)
    res["low_given_sip"]=dict(p=pl,exp=round(mul,2),obs=obs_l)
    out[d]=dict(K=K,**res)
    print("%-26s %4d | L obs=%5d exp=%7.2f z=%6.2f p=%.5f | S obs=%5d exp=%7.2f z=%6.2f p=%.5f | S|L p=%.4f  L|S p=%.4f"
          %(d,K,res["low"]["obs"],res["low"]["exp"],res["low"]["z"],res["low"]["p"],
            res["sip"]["obs"],res["sip"]["exp"],res["sip"]["z"],res["sip"]["p"],ps,pl))
json.dump(out,open(os.path.join(SCR,"ck15_global.json"),"w"),indent=1)
