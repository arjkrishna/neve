import json, sys, collections, statistics as st
for name in ("tb","host"):
    rows = json.load(open(sys.argv[1] + "/" + name + ".json"))
    print("=====", name, "n=", len(rows))
    agg = collections.Counter(); tot = collections.Counter()
    for r in rows:
        tot[r["section"]] += 1
        agg[r["section"]] += 1 if r["success"] else 0
    for s in ("CCA","ICA-mid","siphon"):
        if tot[s]: print(f"  {s:8s} {agg[s]}/{tot[s]} = {100*agg[s]/tot[s]:.1f}%")
    ns = sum(agg.values()); nt = sum(tot.values())
    print(f"  TOTAL    {ns}/{nt} = {100*ns/nt:.1f}%")
    pl = [r["path_len"] for r in rows]
    print("  path_len min/med/max %.1f %.1f %.1f" % (min(pl), st.median(pl), max(pl)))
    for s in ("CCA","ICA-mid","siphon"):
        v=[r["path_len"] for r in rows if r["section"]==s]
        if v: print(f"    {s:8s} n={len(v)} pl {min(v):.1f}-{max(v):.1f}")
    stp_s=[r["steps"] for r in rows if r["success"]]
    stp_f=[r["steps"] for r in rows if not r["success"]]
    print("  steps success med %.0f  (n=%d); fail med %.0f (n=%d)" % (st.median(stp_s), len(stp_s), st.median(stp_f) if stp_f else -1, len(stp_f)))
    # siphon detail
    sr=[r for r in rows if r["section"]=="siphon"]
    print("  siphon steps med succ %.0f" % st.median([r["steps"] for r in sr if r["success"]] or [0]))
