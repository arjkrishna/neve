"""BUCKLE-CLEARING RECOVERY classifier + validation (operates on trace dumps)."""
import json, sys
def med(v): v=sorted(v); return v[len(v)//2]
def q(v,p):
    if not v: return float('nan')
    v=sorted(v); i=(len(v)-1)*p/100.0; lo=int(i); hi=min(lo+1,len(v)-1); f=i-lo
    return v[lo]*(1-f)+v[hi]*f

def features(path):
    """One record per SOFT/HARD event with every raw quantity the labels need."""
    out=[]
    for line in open(path):
        ep=json.loads(line)
        proj,gw,fold=ep["proj"],ep["gw"],ep["fold"]; n=len(proj)
        if n<2: continue
        slack=[gw[j]-proj[j] for j in range(n)]
        spans=[]
        for e in ep["events"]:
            a=max(0,min(e["first"]-1,n-1)); b=(e["close"]-1) if e["close"]>0 else n-1
            spans.append((a,max(a,min(b,n-1)),e))
        for i,(a,b,e) in enumerate(spans):
            if e["k"] not in ("soft","hard"): continue
            pk=a+max(range(b-a+1),key=lambda j:gw[a+j])
            base=med(slack[max(0,a-20):a] or slack[:1])
            post=slack[b:min(b+11,n)]
            nxt=spans[i+1] if i+1<len(spans) else None
            gap=(nxt[2]["first"]-e["close"]) if nxt else None
            out.append(dict(
                seed=ep["seed"], succ=ep["succ"], reason=ep["reason"], pl=ep["pl"],
                k=e["k"], retract=e["r"], p0=e["p0"], stall_len=b-a+1,
                fold_load=max(fold[a:pk+1]), fold_all=max(fold[a:b+1]), fold_close=fold[b],
                slack_load=round(max(slack[a:pk+1])-base,3),
                slack_resid=round(med(post)-base,3),
                slack_rel=round(max(slack[a:b+1])-med(post),3),
                adv10=round(max(proj[b:min(b+11,n)])-e["p0"],3),
                adv25=round(max(proj[b:min(b+26,n)])-e["p0"],3),
                adv50=round(max(proj[b:min(b+51,n)])-e["p0"],3),
                adv_end=round(max(proj[b:])-e["p0"],3),
                steps_left=n-1-b,
                restall_same=int(bool(nxt) and gap is not None and 0<=gap<=20 and abs(nxt[2]["p0"]-e["p0"])<=2.0),
                frac_path=round(max(proj)/ep["pl"],3)))
    return out

def label(e,F=4,S=10.0,A=10.0,R=0.0,use_R=True):
    buckle = (e["fold_load"]>=F) or (e["slack_load"]>=S)
    if not buckle: return "COSMETIC"
    passed = e["adv25"]>=A and not e["restall_same"]
    reduced = (e["slack_resid"]<=R) if use_R else True
    if passed and reduced: return "CLEARING"
    if passed and not reduced: return "PARTIAL"
    return "FUTILE"

def table(evs,**kw):
    cats={}
    for e in evs: cats.setdefault(label(e,**kw),[]).append(e)
    return cats

def report(tag,evs,**kw):
    cats=table(evs,**kw); n=len(evs)
    print("%-28s n=%d"%(tag,n))
    for c in ("CLEARING","PARTIAL","FUTILE","COSMETIC"):
        g=cats.get(c,[])
        if not g: print("   %-9s   0"%c); continue
        s=sum(1 for e in g if e["succ"])
        print("   %-9s %3d (%5.1f%%)  P(ep success)=%.3f (%2d/%2d)  med_adv25=%6.1f  med_adv_end=%6.1f  med_retract=%6.2f  med_fold_load=%4.1f  restall_same=%d  med_frac_path=%.3f"
              %(c,len(g),100.0*len(g)/n,s/len(g),s,len(g),q([e["adv25"] for e in g],50),q([e["adv_end"] for e in g],50),
                q([e["retract"] for e in g],50),q([float(e["fold_load"]) for e in g],50),sum(e["restall_same"] for e in g),
                q([e["frac_path"] for e in g],50)))
    cl=cats.get("CLEARING",[]); fu=cats.get("FUTILE",[])
    if cl and fu:
        print("   SEPARATION  P(succ|CLEARING)=%.3f  P(succ|FUTILE)=%.3f  diff=%+.3f  ratio=%.2f"
              %(sum(1 for e in cl if e["succ"])/len(cl),sum(1 for e in fu if e["succ"])/len(fu),
                sum(1 for e in cl if e["succ"])/len(cl)-sum(1 for e in fu if e["succ"])/len(fu),
                (sum(1 for e in cl if e["succ"])/len(cl))/max(1e-9,sum(1 for e in fu if e["succ"])/len(fu))))
    d=[e for e in cl if not e["succ"]]
    print("   DEFERRED (CLEARING in a failed episode) = %d/%d = %.1f%% of CLEARING"%(len(d),len(cl),100.0*len(d)/max(1,len(cl))))
    return cats

if __name__!="__main__": sys.exit if False else None
if __name__=="__main__":
  A=features(sys.argv[1]); H=features(sys.argv[2])
  print("="*130); print("(a) BASELINE DEFINITION  F=4 fold, S=10mm slack_load, A=10mm adv25, R=0mm slack_resid")
  report("A ckpt2002292 host",A); print(); report("checkpoint0 host (CONTROL)",H)
  print()
  print("="*130); print("(b) SENSITIVITY")
  for nm,kw in [("F=3",dict(F=3)),("F=4 base",dict()),("F=5",dict(F=5)),("F=6",dict(F=6)),
              ("S=8",dict(S=8.0)),("S=12",dict(S=12.0)),("S=15",dict(S=15.0)),("S=inf(fold only)",dict(S=1e9)),
              ("A=6",dict(A=6.0)),("A=8",dict(A=8.0)),("A=12",dict(A=12.0)),("A=15",dict(A=15.0)),("A=25",dict(A=25.0)),
              ("R=-2",dict(R=-2.0)),("R=+2",dict(R=2.0)),("no R gate",dict(use_R=False))]:
    c=table(A,**kw); n=len(A)
    def f(x):
        g=c.get(x,[]); 
        return "%3d/%5.1f%%/succ %.3f"%(len(g),100.0*len(g)/n,(sum(1 for e in g if e["succ"])/len(g)) if g else float('nan'))
    print("  %-16s CLEAR %s | PART %s | FUT %s | COSM %s"%(nm,f("CLEARING"),f("PARTIAL"),f("FUTILE"),f("COSMETIC")))
