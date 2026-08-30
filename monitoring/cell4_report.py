import json,sys,collections,statistics as st

def q(xs,p):
    if not xs: return None
    xs=sorted(xs); i=min(len(xs)-1,max(0,int(round(p*(len(xs)-1)))))
    return xs[i]

def pct(a,b): return None if b==0 else round(100.0*a/b,1)

def analyse(eps, label, succmap=None):
    n=len(eps)
    steps=[e["steps"] for e in eps]
    tot_steps=sum(steps)
    # success
    for e in eps:
        s=e["succ"]
        if succmap is not None and e["seed"] in succmap: s=succmap[e["seed"]]
        e["_s"]=s
    known=[e for e in eps if e["_s"] is not None]
    ev12=[e["ev"]["12"] for e in eps]
    stalls=[x for v in ev12 for x in v]
    mix=collections.Counter(x["k"] for x in stalls)
    tot=len(stalls); res=tot-mix["unrec"]
    nostall=[e for e in known if not e["ev"]["12"]]
    stalled=[e for e in known if e["ev"]["12"]]
    rec=[e for e in stalled if all(x["k"]!="unrec" for x in e["ev"]["12"])]
    unr=[e for e in stalled if any(x["k"]=="unrec" for x in e["ev"]["12"])]
    rd=[x["r"] for x in stalls if x["k"]!="unrec"]
    rr=[e["raw_run_max"] for e in eps]
    sw={k: round(sum(len(e["ev"][str(k)]) for e in eps)/n,3) for k in (4,6,8,12)}
    onset_abs=[]; onset_rel=[]
    for e in eps:
        for o in e["onsets"]:
            onset_abs.append(o)
            if e["ost_s"] is not None: onset_rel.append(round(o-e["ost_s"],2))
    ost=[e["ost_s"] for e in eps if e["ost_s"] is not None]
    plost=[e["pl"]-e["ost_s"] for e in eps if e["ost_s"] is not None and e["pl"]]
    def band(xs,w):
        c=collections.Counter(int(x//w)*w for x in xs)
        return c.most_common(6)
    # coil
    gws=[e["gwslack_max"] for e in eps]; cas=[e["cathslack_max"] for e in eps]
    coil50=[e for e in eps if e["cathslack_max"]>50]
    coil100=[e for e in eps if e["gwslack_max"]>100]
    coilU=[e for e in eps if e["cathslack_max"]>50 or e["gwslack_max"]>100]
    d=dict(run=label,n=n,steps=tot_steps,med_len=st.median(steps),
        stalls=tot, per_ep=round(tot/n,3), per_1k=round(1000.0*tot/max(1,tot_steps),3),
        pct_eps_stalled=pct(sum(1 for v in ev12 if v),n),
        grind=mix["grind"],soft=mix["soft"],hard=mix["hard"],unrec=mix["unrec"],
        mix_pct=[pct(mix[k],tot) for k in ("grind","soft","hard","unrec")],
        frac_resolved=pct(res,tot),
        n_known=len(known),
        P_succ_nostall=(pct(sum(1 for e in nostall if e["_s"]),len(nostall)),len(nostall)),
        P_succ_recov=(pct(sum(1 for e in rec if e["_s"]),len(rec)),len(rec)),
        P_succ_unrec=(pct(sum(1 for e in unr if e["_s"]),len(unr)),len(unr)),
        retract_med=round(st.median(rd),2) if rd else None,
        retract_p90=q(rd,0.9), retract_max=max(rd) if rd else None,
        sweep=sw,
        rawrun_med=st.median(rr), rawrun_p90=q(rr,0.9),
        rawrun_ge4=pct(sum(1 for x in rr if x>=4),n), rawrun_ge12=pct(sum(1 for x in rr if x>=12),n),
        onset_n=len(onset_abs),
        onset_projs_med=round(st.median(onset_abs),1) if onset_abs else None,
        onset_projs_mode20=band(onset_abs,20),
        onset_rel_med=round(st.median(onset_rel),1) if onset_rel else None,
        onset_rel_mode20=band(onset_rel,20),
        ost_med=round(st.median(ost),1) if ost else None,
        ost_p10=q(ost,0.1), ost_p90=q(ost,0.9),
        pl_minus_ost_med=round(st.median(plost),2) if plost else None,
        pl_minus_ost_p10=round(q(plost,0.1),2) if plost else None,
        pl_minus_ost_p90=round(q(plost,0.9),2) if plost else None,
        gwslack_max=round(max(gws),1), gwslack_p99=round(q(gws,0.99),1), gwslack_med=round(st.median(gws),1),
        cathslack_max=round(max(cas),1), cathslack_p99=round(q(cas,0.99),1), cathslack_med=round(st.median(cas),1),
        n_cath50=len(coil50), n_cath50_succ=sum(1 for e in coil50 if e["_s"]),
        n_gw100=len(coil100), n_gw100_succ=sum(1 for e in coil100 if e["_s"]),
        n_coilany=len(coilU), n_coilany_succ=sum(1 for e in coilU if e["_s"]),
        n_coilany_stalled=sum(1 for e in coilU if e["ev"]["12"]),
        dins_med=round(st.median([e["dins_mean"] for e in eps]),3),
        succ_overall=(sum(1 for e in known if e["_s"]),len(known)),
    )
    return d


import os
R=r"D:/Arjun/workspace/neve/saved/eve_paper/neurovascular/full/mesh_ben"
V=R+"/2026-07-25_022443_rcca_p2_teacher_v1bp/checkpoints"
TB=R+"/2026-08-28_075919_rcca_topbrain_v1/checkpoints"
JMAP={
 "v1bp2002292_HOST":V+"/eval_anatomies_checkpoint2002292/episodes_official_20260729_085006.jsonl",
 "v1bp2002292_TB4": V+"/eval_anatomies_checkpoint2002292/episodes_official_20260828_045651.jsonl",
 "v1bp2002292_TB22":V+"/eval_anatomies_checkpoint2002292/episodes_official_20260828_053306.jsonl",
 "v1bp514264_HOST": V+"/eval_anatomies_checkpoint514264/episodes_official_20260729_070938.jsonl",
 "ckpt0_PROC":      V+"/eval_anatomies_checkpoint0/episodes_official_20260728_045004.jsonl",
 "ckpt0_TB22":      V+"/eval_anatomies_checkpoint0/episodes_official_20260828_062606.jsonl",
 "tb256370_HOST":   TB+"/eval_anatomies_checkpoint256370/episodes_official_20260828_185008.jsonl",
 "tb505230_HOST":   TB+"/eval_anatomies_checkpoint505230/episodes_official_20260828_171534.jsonl",
}
SUCC={}
for k,f in JMAP.items():
    if not os.path.exists(f): continue
    m={}
    for l in open(f):
        l=l.strip()
        if not l: continue
        d=json.loads(l)
        m[str(d["seed"])]=bool(d.get("grader_success",d.get("success")))
    SUCC[k]=m

if __name__=="__main__":
    eps=[]
    for p in sys.argv[1:]:
        eps+= [json.loads(l) for l in open(p)]
    by=collections.OrderedDict()
    for e in eps: by.setdefault(e["run"],[]).append(e)
    out=[analyse(v,k, SUCC.get(k)) for k,v in by.items()]
    print(json.dumps(out,indent=1))
