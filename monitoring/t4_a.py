import pickle, numpy as np, collections
D = pickle.load(open("_t4_rows.pkl","rb")); T,H = D["T"], D["H"]
mh = {r["seed"]:r for r in H}
gt = [r for r in T if r["grafted"]]
by = collections.defaultdict(list)
for r in gt: by[r["anat"]].append(r)
print("anat  n  Tsucc  rate   Hsucc  Hrate")
for a in sorted(by):
    rs = by[a]; n=len(rs); ts=sum(r["succ"] for r in rs); hs=sum(mh[r["seed"]]["succ"] for r in rs)
    print(f"{a} {n:3d} {ts:3d} {100*ts/n:6.1f}  {hs:3d} {100*hs/n:6.1f}")
print()
print("TOTAL grafted", len(gt), sum(r['succ'] for r in gt), sum(mh[r['seed']]['succ'] for r in gt))
