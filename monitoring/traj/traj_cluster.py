"""traj_cluster.py -- unsupervised clustering of whole-episode trajectories.

Core feature set = what every run (incl. the stuck-format extracts) has:
progress profile, slack profile, timing, stall/retract/fold statistics.
PCA -> k-means (scipy, k-means++ init, best of N restarts), k chosen by
silhouette on a subsample. Writes out/clusters.csv (key, cluster) and
out/cluster_profile.csv (per-cluster feature means).
"""
import os, sys, json, argparse
import numpy as np, pandas as pd
from scipy.cluster.vq import kmeans2
from scipy.spatial.distance import cdist
from traj_load import load, make_key, OUT

SCALARS = ["steps", "prog_final", "prog_max", "t_max_frac", "slack_max", "slack_final", "slack_mean",
           "slack_t_max_frac", "giveback_mm", "flat_frac", "flat_run_max", "t_25", "t_50", "t_75", "t_90",
           "fold_max", "fold_ge4", "fold_ge10", "fold_onsets", "cmd_mean", "cmd_push_frac", "ret_steps",
           "withdrawn_mm", "fed_mm", "ret_bouts", "ret_bout_max_len", "ret_bout_max_mm", "reversals",
           "ev_n", "ev_grind", "ev_soft", "ev_hard", "ev_unrec", "ev_ret_max", "ev_first_onset_frac",
           "ev_last_close_frac", "ev_stalled_steps", "cath_slack_max"]
LOG1P = {"steps", "flat_run_max", "fold_max", "fold_ge4", "fold_ge10", "fold_onsets", "ret_steps",
         "withdrawn_mm", "fed_mm", "ret_bouts", "ret_bout_max_len", "ret_bout_max_mm", "reversals",
         "ev_n", "ev_stalled_steps", "ev_ret_max", "cath_slack_max"}
SLOG = {"slack_max", "slack_final", "slack_mean", "giveback_mm"}   # signed log


def feature_matrix(df):
    cols = []; names = []
    for c in SCALARS:
        x = df[c].astype(float).values.copy()
        if c == "cath_slack_max": x = np.nan_to_num(x, nan=0.0)
        x = np.nan_to_num(x, nan=0.0)
        if c in LOG1P: x = np.log1p(np.clip(x, 0, None))
        elif c in SLOG: x = np.sign(x) * np.log1p(np.abs(x))
        cols.append(x); names.append(c)
    prof = np.vstack(df["prof"].values); sp = np.vstack(df["slack_prof"].values)
    sp = np.sign(sp) * np.log1p(np.abs(sp))
    X = np.column_stack(cols + [prof, sp])
    names += ["prof_%02d" % i for i in range(prof.shape[1])] + ["sprof_%02d" % i for i in range(sp.shape[1])]
    return X, names


def zscore(X):
    mu = X.mean(0); sd = X.std(0); sd[sd == 0] = 1.0
    return (X - mu) / sd, mu, sd


def pca(Z, k):
    U, S, Vt = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
    expl = (S ** 2) / (S ** 2).sum()
    return (Z - Z.mean(0)) @ Vt[:k].T, expl[:k], Vt[:k]


def silhouette(P, lab, n=6000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(P), min(n, len(P)), replace=False)
    Ps = P[idx]; L = lab[idx]; D = cdist(Ps, Ps)
    ks = np.unique(L); s = np.zeros(len(Ps))
    for i in range(len(Ps)):
        own = L[i]; m = L == own
        if m.sum() <= 1: s[i] = 0; continue
        a = D[i, m].sum() / (m.sum() - 1)
        b = min(D[i, L == k].mean() for k in ks if k != own)
        s[i] = (b - a) / max(a, b)
    return float(s.mean())


def best_kmeans(P, k, restarts=6, seed=0):
    best = None
    for r in range(restarts):
        c, lab = kmeans2(P, k, minit="++", iter=60, seed=seed + r)
        d = ((P - c[lab]) ** 2).sum()
        if best is None or d < best[0]: best = (d, c, lab)
    return best[1], best[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", default="5,6,7,8,9,10,12")
    ap.add_argument("--npc", type=int, default=15)
    ap.add_argument("--k", type=int, default=0, help="force k (skip selection)")
    ap.add_argument("--hier", default="", help="level-2 spec by level-1 success rank, e.g. 0:4,1:3,2:2,3:2")
    a = ap.parse_args()
    df = load()
    df["key"] = make_key(df)
    assert df["key"].is_unique, "episode keys must be unique (%d dups)" % df["key"].duplicated().sum()
    X, names = feature_matrix(df)
    Z, mu, sd = zscore(X)
    P, expl, comps = pca(Z, a.npc)
    print("episodes", len(df), "features", X.shape[1], "PCA var explained", expl.sum().round(3))
    if a.k:
        ks = [a.k]
    else:
        ks = [int(k) for k in a.ks.split(",")]
    res = {}
    for k in ks:
        c, lab = best_kmeans(P, k)
        s = silhouette(P, lab)
        res[k] = (s, c, lab); print("k=%d silhouette=%.3f sizes=%s" % (k, s, np.bincount(lab).tolist()))
    k = max(res, key=lambda k: res[k][0]) if not a.k else a.k
    s, c, lab = res[k]
    print("chosen k", k)
    if a.hier:
        # level-2: sub-cluster each level-1 group with its own k
        k2map = {int(p.split(":")[0]): int(p.split(":")[1]) for p in a.hier.split(",")}
        # level-1 ids are arbitrary; address them by success-rank so the spec is stable
        rank = df.assign(l1=lab).groupby("l1")["success"].mean().sort_values(ascending=False).index.tolist()
        lab2 = np.full(len(P), -1); nxt = 0
        for r_i, c1 in enumerate(rank):
            idx = np.where(lab == c1)[0]; k2 = k2map.get(r_i, 1)
            if k2 <= 1 or len(idx) < 50 * k2: sub = np.zeros(len(idx), int)
            else: _, sub = best_kmeans(P[idx], k2)
            lab2[idx] = nxt + sub; nxt += int(sub.max()) + 1
            print("L1 rank %d (succ %.2f, n=%d) -> %d sub-clusters" % (r_i, df["success"].values[idx].mean(), len(idx), int(sub.max()) + 1))
        lab = lab2; k = nxt; c = np.vstack([P[lab == j].mean(0) for j in range(k)])
    df["cluster"] = lab
    # order clusters by success rate then steps for readable ids
    order = df.groupby("cluster").agg(succ=("success", "mean"), steps=("steps", "median")).sort_values(["succ", "steps"], ascending=[False, True])
    remap = {old: new for new, old in enumerate(order.index)}
    df["cluster"] = df["cluster"].map(remap)
    df[["key", "tag", "kind", "cluster"]].to_csv(os.path.join(OUT, "clusters.csv"), index=False)
    # profile table
    desc_cols = ["steps", "prog_max", "prog_final", "slack_max", "cath_slack_max", "cath_lead_max", "giveback_mm",
                 "flat_frac", "fold_max", "withdrawn_mm", "ret_bouts", "ret_bout_max_mm", "reversals", "ev_n",
                 "ev_unrec", "ev_ret_max", "ev_last_close_frac", "off_steps", "phys_RVA", "tip_tortuosity", "xt_max"]
    g = df.groupby("cluster")
    prof = g[desc_cols].median(numeric_only=True)
    prof.insert(0, "n", g.size()); prof.insert(1, "succ", g["success"].mean().round(3))
    prof.insert(2, "coil", g["coil"].mean().round(3)); prof.insert(3, "shove", g["cath_shove"].mean().round(3))
    prof.insert(4, "recovered", g["recovered"].mean().round(3)); prof.insert(5, "long_succ", g["long_succ"].mean().round(3))
    prof.to_csv(os.path.join(OUT, "cluster_profile.csv"))
    pd.set_option("display.width", 260); pd.set_option("display.max_columns", 40)
    print(prof.round(2).to_string())
    # top loading features per cluster (z-scored centroid)
    Zc = np.vstack([Z[lab == old].mean(0) for old in order.index])
    for ci in range(k):
        top = np.argsort(-np.abs(Zc[ci]))[:8]
        print("cluster %d: " % ci + ", ".join("%s=%+.1f" % (names[j], Zc[ci, j]) for j in top))
    np.savez(os.path.join(OUT, "cluster_model.npz"), mu=mu, sd=sd, comps=comps, centers=c, names=np.asarray(names), remap=np.asarray([remap[i] for i in range(k)]))


if __name__ == "__main__":
    main()
