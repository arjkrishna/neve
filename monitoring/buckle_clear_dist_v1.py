import json, sys
def load(p):
    return [json.loads(l) for l in open(p)]
def q(v, p):
    if not v: return float('nan')
    v = sorted(v); i = (len(v)-1)*p/100.0
    lo = int(i); hi = min(lo+1, len(v)-1); f = i-lo
    return v[lo]*(1-f)+v[hi]*f
def desc(name, v):
    if not v:
        print("%-18s n=0" % name); return
    print("%-18s n=%-4d min=%8.2f p10=%8.2f p25=%8.2f med=%8.2f p75=%8.2f p90=%8.2f max=%9.2f mean=%8.2f"
          % (name, len(v), min(v), q(v,10), q(v,25), q(v,50), q(v,75), q(v,90), max(v), sum(v)/len(v)))
for tag, path in [("A", sys.argv[1]), ("H0", sys.argv[2])]:
    ev = load(path)
    sh = [e for e in ev if e["k"] in ("soft","hard")]
    print("="*110); print("%s  all=%d  soft=%d hard=%d grind=%d unrec=%d   SOFT+HARD n=%d"
          % (tag, len(ev), sum(1 for e in ev if e["k"]=="soft"), sum(1 for e in ev if e["k"]=="hard"),
             sum(1 for e in ev if e["k"]=="grind"), sum(1 for e in ev if e["k"]=="unrec"), len(sh)))
    for f in ["retract","slack_first","slack_onset","slack_max","slack_min","slack_drop","slack_at_close",
              "fold_first","fold_max","fold_at_close","stall_len","adv_10","adv_25","adv_50","adv_end","steps_left"]:
        desc(f, [e[f] for e in sh if e.get(f) is not None])
    print("fold_reset frac: %.3f" % (sum(e["fold_reset"] for e in sh)/max(1,len(sh))))
    rs = [e for e in sh if e.get("restall_dp") is not None]
    print("next-event-within-20-steps: %d/%d   of those dp<=2mm: %d" %
          (len(rs), len(sh), sum(1 for e in rs if e["restall_dp"]<=2.0)))
    # grind baseline for contrast
    gr = [e for e in ev if e["k"]=="grind"]
    print("-- grind contrast --")
    for f in ["slack_max","fold_max","adv_25","adv_end"]:
        desc("grind."+f, [e[f] for e in gr if e.get(f) is not None])
    un = [e for e in ev if e["k"]=="unrec"]
    print("-- unrec contrast --")
    for f in ["slack_max","fold_max","retract"]:
        desc("unrec."+f, [e[f] for e in un if e.get(f) is not None])
