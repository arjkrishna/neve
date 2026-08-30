"""FINAL operational definition of a BUCKLE-CLEARING RECOVERY + validation.

Event set: every SOFT or HARD stall event of extract_stuck.py's CANON config
(stall_eps=0.3, push_min=2.0, stuck_steps=12, retract_min=1.0, soft_max=8.0,
pass_eps=1.0), unmodified.

Window: a = first step of the stall run, b = closing step (proj_s > p0+1.0),
pk = argmax of inserted_gw over [a,b] (end of the LOADING phase).

  fold_load   = max fold over [a..pk]                    buckle load, before retraction
  slack_load  = max(slack_gw over [a..pk]) - base        base = median slack_gw over [a-20..a-1]
  fold_close  = fold at step b                           buckle still folding at pass-through?
  adv25       = max(proj_s over [b..b+25]) - p0          real advance past the arrest point
  restall     = next stall run starts <=20 steps after b at |p0|<=2mm of this one

  BUCKLE_PRESENT := fold_load >= 4  OR  slack_load >= 10.0
  REDUCED        := fold_close == 0
  PASSED         := adv25 >= 15.0  AND  not restall

  CLEARING = BUCKLE_PRESENT and REDUCED and PASSED     (DEFERRED = CLEARING in a failed episode)
  FUTILE   = BUCKLE_PRESENT and not (REDUCED and PASSED)
  COSMETIC = not BUCKLE_PRESENT
"""
import json,sys
sys.path.insert(0,"d:/Arjun/workspace/neve/monitoring")
from buckle_clear_classify_v1 import features,q

F_DEF, S_DEF, A_DEF = 4, 10.0, 15.0

def lab(e,F=F_DEF,S=S_DEF,A=A_DEF,use_reduced=True,use_restall=True):
    if not (e["fold_load"]>=F or e["slack_load"]>=S): return "COSMETIC"
    red = (e["fold_close"]==0) if use_reduced else True
    ps  = e["adv25"]>=A and (not e["restall_same"] if use_restall else True)
    return "CLEARING" if (red and ps) else "FUTILE"

def show(tag,ev,**kw):
    n=len(ev); c={}
    for e in ev: c.setdefault(lab(e,**kw),[]).append(e)
    print("%s   soft+hard n=%d"%(tag,n))
    for k in ("CLEARING","FUTILE","COSMETIC"):
        g=c.get(k,[])
        if not g: print("   %-8s   0  ( 0.0%%)"%k); continue
        s=sum(1 for e in g if e["succ"])
        print("   %-8s %3d (%5.1f%%)  P(ep success)=%.3f (%2d/%2d)  med adv25=%6.1f  med retract=%6.2f  med fold_load=%4.1f  hard-share=%.2f  restall=%d"
              %(k,len(g),100.*len(g)/n,s/len(g),s,len(g),q([e["adv25"] for e in g],50),q([e["retract"] for e in g],50),
                q([float(e["fold_load"]) for e in g],50),sum(1 for e in g if e["k"]=="hard")/len(g),sum(e["restall_same"] for e in g)))
    cl,fu=c.get("CLEARING",[]),c.get("FUTILE",[])
    if cl and fu:
        a=sum(1 for e in cl if e["succ"])/len(cl); b=sum(1 for e in fu if e["succ"])/len(fu)
        print("   >> SEPARATION P(succ|CLEARING)=%.3f vs P(succ|FUTILE)=%.3f  diff=%+.3f  ratio=%.2f"%(a,b,a-b,a/max(b,1e-9)))
    if cl: print("   >> DEFERRED = %d/%d CLEARING events sit in a FAILED episode (%.0f%%)"%(sum(1 for e in cl if not e["succ"]),len(cl),100.*sum(1 for e in cl if not e["succ"])/len(cl)))
    return c

if __name__=="__main__":
  A=features(sys.argv[1]); H=features(sys.argv[2])
  print("="*128); print("(a) OUTCOME SEPARATION  --  F=4, S=10mm, A=15mm")
  show("A  ckpt2002292 host (74/98 = 75.5%)",A); print()
  show("H0 checkpoint0 host (25/98 = 25.5%)  [NEGATIVE CONTROL]",H)
  print()
  print("="*128); print("(b) SENSITIVITY  (A run; format: n / %% of soft+hard / P(ep success))")
  def row(nm,**kw):
      c={}
      for e in A: c.setdefault(lab(e,**kw),[]).append(e)
      o=[]
      for k in ("CLEARING","FUTILE","COSMETIC"):
          g=c.get(k,[]); sc=(sum(1 for e in g if e["succ"])/len(g)) if g else float('nan')
          o.append("%-8s %2d/%4.1f%%/%.3f"%(k[:4],len(g),100.*len(g)/len(A),sc))
      cl,fu=c.get("CLEARING",[]),c.get("FUTILE",[])
      d=(sum(1 for e in cl if e["succ"])/len(cl)-sum(1 for e in fu if e["succ"])/len(fu)) if (cl and fu) else float('nan')
      print("  %-24s %s  sep=%+.3f"%(nm," | ".join(o),d))
  print(" -- buckle-load threshold F (fold_load) --")
  for v in (2,3,4,5,6,8): row("F=%d"%v,F=v)
  print(" -- buckle-load threshold S (slack_load, mm) --")
  for v in (6.,8.,10.,12.,15.,1e9): row("S=%s"%("off" if v>1e8 else "%g"%v),S=v)
  print(" -- advance threshold A (adv25, mm) --")
  for v in (5.,10.,14.,15.,20.,25.,30.,40.): row("A=%g"%v,A=v)
  print(" -- gate ablations --")
  row("no REDUCED gate",use_reduced=False); row("no restall gate",use_restall=False)
  row("no REDUCED, no restall",use_reduced=False,use_restall=False)
  print()
  print("="*128); print("(c) NEGATIVE CONTROL DETAIL -- checkpoint0")
  print("  checkpoint0 host: 51,339 steps, 103 canon stall events (31 grind / 3 soft / 0 hard / 69 unrec),")
  print("  max retraction 5.51 mm.  Soft+hard events and their labels:")
  for e in H:
      print("   seed=%s k=%-4s retract=%5.2f fold_load=%2d slack_load=%6.2f fold_close=%d adv25=%5.2f adv50=%5.2f restall=%d -> %s"
            %(e["seed"],e["k"],e["retract"],e["fold_load"],e["slack_load"],e["fold_close"],e["adv25"],e["adv50"],e["restall_same"],lab(e)))
  print("  CLEARING events on the control: %d"%sum(1 for e in H if lab(e)=="CLEARING"))
  print()
  print("="*128); print("(d) EPISODE-LEVEL VIEW (A)")
  seeds={}
  for e in A: seeds.setdefault(e["seed"],[]).append(e)
  import collections
  cats=collections.Counter()
  for s,g in seeds.items():
      ls=[lab(e) for e in g]
      top="CLEARING" if "CLEARING" in ls else ("FUTILE" if "FUTILE" in ls else "COSMETIC")
      cats[(top,g[0]["succ"])]+=1
  for t in ("CLEARING","FUTILE","COSMETIC"):
      a=cats[(t,True)]; b=cats[(t,False)]
      print("  episodes whose best soft/hard event is %-8s : n=%2d  success=%.3f (%d/%d)"%(t,a+b,a/max(1,a+b),a,a+b))
