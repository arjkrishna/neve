"""traj_paired.py -- seed-paired analysis of validation blocks within a run.

The validation evals reuse the same seeds every block, so for two blocks of the
same run each seed can be classified both-succeed / flip-up / flip-down /
both-fail, and the trajectory TYPE of each side inspected. Answers: when a run
climbs from X% to Y%, are the newly-won seeds recoveries or clean runs, and do
the seeds that always succeed stop needing recovery?

usage: python traj_paired.py car_v3 car_nopriv [--pairs 1:3,4:6]
"""
import sys, json, argparse, os
import numpy as np, pandas as pd
from traj_load import load, make_key, eval_blocks, OUT

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
SHORT = {0: "clean", 1: "clean", 2: "cathled", 3: "light-rec", 4: "deep-rec", 5: "shove-fight",
         6: "near-fight", 7: "mid-thrash", 8: "coil", 9: "shove-cap", 10: "prox-thrash"}
REC = ["light-rec", "deep-rec"]


def pair(v, tag, lo, hi):
    a = v[(v.tag == tag) & (v.block == lo)].dropna(subset=["seed"]).drop_duplicates("seed").set_index("seed")
    b = v[(v.tag == tag) & (v.block == hi)].dropna(subset=["seed"]).drop_duplicates("seed").set_index("seed")
    j = a[["success", "ct", "steps"]].join(b[["success", "ct", "steps"]], lsuffix="_lo", rsuffix="_hi", how="inner")
    up = j[~j.success_lo & j.success_hi]; both = j[j.success_lo & j.success_hi]
    down = j[j.success_lo & ~j.success_hi]; nn = j[~j.success_lo & ~j.success_hi]
    print("%s: block %d (%.1f%%) -> block %d (%.1f%%): paired %d | both-succeed %d | flip UP %d | flip DOWN %d | both-fail %d" % (
        tag, lo, 100 * a.success.mean(), hi, 100 * b.success.mean(), len(j), len(both), len(up), len(down), len(nn)))
    print("   flip-UP successes (higher block) by type:", up.ct_hi.value_counts().to_dict(), "| median steps %.0f" % up.steps_hi.median() if len(up) else "")
    print("   flip-UP seeds' failure type (lower block):", up.ct_lo.value_counts().to_dict())
    print("   both-succeed seeds, type in lower block:", both.ct_lo.value_counts().to_dict())
    print("   both-succeed seeds, type in higher block:", both.ct_hi.value_counts().to_dict())
    if len(down): print("   flip-DOWN failures (higher block) by type:", down.ct_hi.value_counts().to_dict())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("tags", nargs="+"); ap.add_argument("--pairs", default="")
    a = ap.parse_args()
    df = load(only=a.tags); df["key"] = make_key(df)
    cl = pd.read_csv(os.path.join(OUT, "clusters.csv"))[["key", "cluster"]]; df = df.merge(cl, on="key")
    df["ct"] = df.cluster.map(SHORT); df, blocks = eval_blocks(df)
    v = df[(df.kind == "val") & (df.block >= 0)].copy()
    bs = v.groupby(["tag", "block"]).success.mean()
    print("=== validation blocks: successes / failures by type (counts) ===")
    print(pd.crosstab([v.tag, v.block], [v.success, v.ct]).to_string())
    print("\n=== counterfactual: block success if recovery-type successes had failed ===")
    for (tag, b), g in v.groupby(["tag", "block"]):
        s = g.success.sum(); rec = (g.success & g.ct.isin(REC)).sum(); non = (g.success & ~g.ct.isin(["clean", "cathled"])).sum()
        print("%-11s blk %d  succ %5.1f%%  recovery-type succ %2d (%3.0f%% of succ) -> %5.1f%% without | any non-clean succ %2d -> %5.1f%%" % (
            tag, b, 100 * s / len(g), rec, 100 * rec / max(s, 1), 100 * (s - rec) / len(g), non, 100 * (s - non) / len(g)))
    print("\n=== seed-paired flips ===")
    for tag in a.tags:
        bl = sorted(v[v.tag == tag].block.unique())
        pairs = [tuple(int(x) for x in p.split(":")) for p in a.pairs.split(",")] if a.pairs else [(bl[1], bl[-1])] if len(bl) > 2 else []
        for lo, hi in pairs:
            if lo in bl and hi in bl: pair(v, tag, lo, hi)
    print("\n=== always-succeed vs marginal seeds across all >=90% blocks of each run (block>0) ===")
    for tag in a.tags:
        hb = [b for (t, b), s in bs.items() if t == tag and b > 0 and s >= 0.9]
        if len(hb) < 2: continue
        g = v[(v.tag == tag) & v.block.isin(hb)].dropna(subset=["seed"])
        per = g.groupby("seed").agg(n=("success", "size"), wins=("success", "sum")); per = per[per.n == len(hb)]
        g2 = g.merge(per, left_on="seed", right_index=True)
        print("%s blocks %s: seeds in all: %d | always-succeed %d | marginal %d | never %d" % (tag, hb, len(per), (per.wins == len(hb)).sum(), ((per.wins > 0) & (per.wins < len(hb))).sum(), (per.wins == 0).sum()))
        print(pd.crosstab(np.where(g2.wins == len(hb), "always-succeed", "marginal/never"), [g2.success, g2.ct]).to_string())
    print("\n=== where the recovered stall is: recovery-type successes in >=90% blocks (block>0) ===")
    hi = v[v.success & v.ct.isin(REC) & (v.block > 0)].merge(bs[bs >= 0.9].reset_index()[["tag", "block"]], on=["tag", "block"])
    hi["p0f"] = hi.events.apply(lambda ev: max([e["p0"] for e in ev if e["close"] > 0], default=np.nan)) / hi.pl
    print("n=%d  median steps %.0f  >=200 steps %.0f%%" % (len(hi), hi.steps.median(), 100 * (hi.steps >= 200).mean()))
    print(pd.cut(hi.p0f, [-1, 0.33, 0.66, 0.9, 2], labels=["proximal <33%", "mid 33-66%", "distal 66-90%", "at-target >90%"]).value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
