import json, os, random, itertools
from collections import Counter, defaultdict

ROOT = "/d/Arjun/workspace/neve/carotid_data/anatomies"
if not os.path.isdir(ROOT):
    ROOT = r"D:\Arjun\workspace\neve\carotid_data\anatomies"
rows = []
for d in sorted(os.listdir(ROOT)):
    p = os.path.join(ROOT, d, "provenance.json")
    j = json.load(open(p, encoding="utf-8"))
    low, sip = j["lower"], j["siphon"]
    pat = sip[:-2] if sip.endswith("_L") else sip
    side = "left" if low.endswith("_left") else "right"
    rows.append(dict(name=d, low=low, sip=sip, pat=pat, side=side))
N = len(rows)
print("composites", N, "lowers", len(set(r['low'] for r in rows)),
      "siphons", len(set(r['sip'] for r in rows)), "patients", len(set(r['pat'] for r in rows)))

# ---- connected components on bipartite graph lower <-> patient ----
adj = defaultdict(set)
for r in rows:
    a, b = ("L:" + r["low"]), ("P:" + r["pat"])
    adj[a].add(b); adj[b].add(a)
seen, comps = set(), []
for n in adj:
    if n in seen: continue
    stack, comp = [n], []
    seen.add(n)
    while stack:
        x = stack.pop(); comp.append(x)
        for y in adj[x]:
            if y not in seen:
                seen.add(y); stack.append(y)
    comps.append(comp)
comps.sort(key=len, reverse=True)
print("\nCOMPONENTS (lower<->patient):", len(comps))
for c in comps:
    nl = sum(1 for x in c if x.startswith("L:")); npat = len(c) - nl
    ncomp = sum(1 for r in rows if ("L:" + r["low"]) in set(c))
    print("  comp: lowers=%d patients=%d composites=%d" % (nl, npat, ncomp))

# ---- naive random 80/20 leakage (Monte Carlo) ----
random.seed(0)
T = 5000
ntest = round(0.2 * N)
acc = Counter(); anyleak = 0.0; clean_counts = []
for _ in range(T):
    idx = list(range(N)); random.shuffle(idx)
    te = set(idx[:ntest]); tr = set(idx[ntest:])
    trl = set(rows[i]["low"] for i in tr); trs = set(rows[i]["sip"] for i in tr)
    trp = set(rows[i]["pat"] for i in tr)
    cl = 0
    for i in te:
        r = rows[i]
        a = r["low"] in trl; b = r["sip"] in trs; c = r["pat"] in trp
        acc["low"] += a; acc["sip"] += b; acc["pat"] += c
        if a or b or c: anyleak += 1
        else: cl += 1
    clean_counts.append(cl)
tot = T * ntest
print("\nNAIVE RANDOM 80/20 (n_test=%d, %d trials)" % (ntest, T))
for k in ("low", "sip", "pat"):
    print("  share %-4s donor with train: %.4f" % (k, acc[k] / tot))
print("  share ANY (lower or siphon or patient): %.4f" % (anyleak / tot))
print("  mean fully-clean test composites per trial: %.3f of %d" % (sum(clean_counts)/T, ntest))
print("  trials with >=1 clean test composite: %d/%d" % (sum(1 for c in clean_counts if c), T))
