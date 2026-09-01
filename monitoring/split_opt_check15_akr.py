import json, os, random, math
from collections import Counter, defaultdict

ROOT = r"D:\Arjun\workspace\neve\carotid_data\anatomies"
rows = []
for d in sorted(os.listdir(ROOT)):
    j = json.load(open(os.path.join(ROOT, d, "provenance.json"), encoding="utf-8"))
    sip = j["siphon"]; pat = sip[:-2] if sip.endswith("_L") else sip
    rows.append(dict(name=d, low=j["lower"], sip=sip, pat=pat,
                     side="left" if j["lower"].endswith("_left") else "right", j=j))
lows = sorted(set(r["low"] for r in rows)); pats = sorted(set(r["pat"] for r in rows))
LI = {n:i for i,n in enumerate(lows)}; PI = {n:i for i,n in enumerate(pats)}
NL, NP = len(lows), len(pats)
Ladj = defaultdict(list); Padj = defaultdict(list)
for r in rows:
    i, j = LI[r["low"]], PI[r["pat"]]
    Ladj[i].append(j); Padj[j].append(i)

def run(K, targets, iters, seed, w=3.0):
    rnd = random.Random(seed)
    al = [rnd.randrange(K) for _ in range(NL)]
    ap = [rnd.randrange(K) for _ in range(NP)]
    cnt = [0]*K
    for i in range(NL):
        for j in Ladj[i]:
            if al[i]==ap[j]: cnt[al[i]] += 1
    def obj(c): return sum(c) - w*sum(abs(c[k]-targets[k]) for k in range(K))
    s = obj(cnt)
    best = (s, list(al), list(ap), list(cnt))
    T0,T1 = 6.0, 0.03
    for it in range(iters):
        T = T0*(T1/T0)**(it/iters)
        if rnd.random() < NL/(NL+NP):
            idx = rnd.randrange(NL); arr, adj, oth = al, Ladj[idx], ap
        else:
            idx = rnd.randrange(NP); arr, adj, oth = ap, Padj[idx], al
        old = arr[idx]; new = rnd.randrange(K)
        if new == old: continue
        d = [0]*K
        for m in adj:
            g = oth[m]
            if g == old: d[old] -= 1
            if g == new: d[new] += 1
        nc = [cnt[k]+d[k] for k in range(K)]
        s2 = obj(nc)
        if s2 >= s or rnd.random() < math.exp((s2-s)/T):
            arr[idx]=new; cnt=nc; s=s2
            if s > best[0]: best = (s, list(al), list(ap), list(cnt))
    return best

def bestof(K, tgt, iters=300000, seeds=8):
    bb=None
    for sd in range(seeds):
        b = run(K, list(tgt), iters, sd)
        key = (sum(b[3]), -sum(abs(b[3][k]-tgt[k]) for k in range(K)))
        if bb is None or key > bb[0]: bb=(key,b)
    return bb[1]

print("=== 3-WAY donor-level (train/val/test) ===")
res3 = {}
for tgt in [(151,32,33),(140,30,30),(130,25,25),(120,20,20)]:
    b = bestof(3, tgt); c = b[3]
    res3[tgt] = b
    print("  target %-16s -> kept %-16s total=%3d dropped=%3d" % (tgt, c, sum(c), 216-sum(c)))
print("\n=== 2-WAY donor-level (train/test) ===")
for tgt in [(173,43),(160,40),(150,38),(140,35)]:
    b = bestof(2, tgt); c = b[3]
    print("  target %-12s -> kept %-14s total=%3d dropped=%3d" % (tgt, c, sum(c), 216-sum(c)))
    if tgt==(173,43):
        json.dump({"al":b[1],"ap":b[2],"lows":lows,"pats":pats}, open(r"D:\Arjun\workspace\neve\monitoring\_split2.json","w"))
b = res3[(151,32,33)]
json.dump({"al":b[1],"ap":b[2],"lows":lows,"pats":pats}, open(r"D:\Arjun\workspace\neve\monitoring\_split3.json","w"))
