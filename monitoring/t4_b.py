import pickle, numpy as np, collections
D = pickle.load(open("_t4_rows.pkl","rb")); T,H=D["T"],D["H"]
mh={r["seed"]:r for r in H}
gt=[r for r in T if r["grafted"]]
f=[r for r in gt if not r["succ"]]
print("grafted fails:", len(f))
print("steps distribution (all grafted T):", sorted(set(r["n"] for r in gt))[-6:], "max", max(r["n"] for r in gt))
print("fails steps: min",min(r["n"] for r in f),"med",int(np.median([r["n"] for r in f])),"max",max(r["n"] for r in f))
print("term=True among fails:", sum(1 for r in f if r["term"]=="True"), " trunc=True:", sum(1 for r in f if r["trunc"]=="True"))
print("n==600:", sum(1 for r in f if r["n"]==600), " n>=595:", sum(1 for r in f if r["n"]>=595))
c=collections.Counter((r["term"],r["trunc"]) for r in f); print("term/trunc:",c)
c=collections.Counter(r["n"] for r in f); print("most common step counts:", c.most_common(8))
# same for successes
s=[r for r in gt if r["succ"]]
print("succ steps med",int(np.median([r["n"] for r in s])),"max",max(r["n"] for r in s))
