"""traj_report.py -- cross-tabulate trajectory clusters against runs, eval blocks,
explore windows and host-test checkpoints. Reads out/clusters.csv."""
import os, re, sys, json
import numpy as np, pandas as pd
from scipy.stats import spearmanr, mannwhitneyu
from traj_load import load, make_key, eval_blocks, explore_windows, OUT

pd.set_option("display.width", 260); pd.set_option("display.max_columns", 60); pd.set_option("display.max_rows", 400)


def label_clusters(prof):
    p = os.path.join(OUT, "cluster_names.json")
    if os.path.exists(p):
        return {int(k): v for k, v in json.load(open(p)).items()}
    names = {}
    for c, r in prof.iterrows():
        if r.succ >= 0.6 and r.recovered >= 0.6: n = "recovery-success"
        elif r.succ >= 0.9 and r.shove >= 0.3: n = "cathled-success"
        elif r.succ >= 0.9: n = "clean-success"
        elif r.succ >= 0.6: n = "mixed"
        elif r.coil >= 0.5: n = "coil-fail"
        elif r.shove >= 0.6: n = "shove-fail"
        elif r.giveback_mm >= 100: n = "thrash-fail"
        elif r.prog_max >= 0.75: n = "near-timeout-fail"
        else: n = "stall-far-fail"
        names[c] = "%d:%s" % (c, n)
    return names


def share_table(g, col="cname"):
    t = pd.crosstab(g.index if False else g["_grp"], g[col], normalize="index")
    return t


def main():
    df = load()
    df["key"] = make_key(df)
    cl = pd.read_csv(os.path.join(OUT, "clusters.csv"))[["key", "cluster"]]
    df = df.merge(cl, on="key", how="inner")
    assert df["key"].is_unique
    prof = pd.read_csv(os.path.join(OUT, "cluster_profile.csv"), index_col=0)
    names = label_clusters(prof); df["cname"] = df["cluster"].map(names)
    print("=== R1 cluster glossary ===")
    print(prof[["n", "succ", "coil", "shove", "recovered", "long_succ", "steps", "prog_max", "slack_max", "cath_lead_max",
                "giveback_mm", "withdrawn_mm", "ret_bouts", "ev_n", "ev_unrec", "tip_tortuosity"]].assign(name=pd.Series(names)).round(2).to_string())

    df, blocks = eval_blocks(df); df = explore_windows(df, blocks)
    order = [names[c] for c in sorted(names)]

    print("\n=== R2 cluster composition by run x kind (% of episodes) ===")
    t = pd.crosstab([df["tag"], df["kind"]], df["cname"], normalize="index").reindex(columns=order).fillna(0) * 100
    t.insert(0, "n", df.groupby(["tag", "kind"]).size()); t.insert(1, "succ%", df.groupby(["tag", "kind"])["success"].mean() * 100)
    print(t.round(1).to_string())

    print("\n=== R3 failure composition by run (explore failures only, % of failures) ===")
    f = df[(df["kind"] == "explore") & ~df["success"]]
    t = pd.crosstab(f["tag"], f["cname"], normalize="index").reindex(columns=order).fillna(0) * 100
    t.insert(0, "n_fail", f.groupby("tag").size())
    fin = blocks.sort_values("block").groupby("tag")["succ"].last() * 100
    t.insert(1, "final_eval%", fin.reindex(t.index))
    print(t.round(1).sort_values("final_eval%").to_string())

    print("\n=== R4 success composition by run (explore successes, % of successes) ===")
    s = df[(df["kind"] == "explore") & df["success"]]
    t = pd.crosstab(s["tag"], s["cname"], normalize="index").reindex(columns=order).fillna(0) * 100
    t.insert(0, "n_succ", s.groupby("tag").size()); t.insert(1, "final_eval%", fin.reindex(t.index))
    t["succ>=200"] = s.groupby("tag")["long_succ"].mean() * 100; t["succ<120"] = s.groupby("tag")["fast_succ"].mean() * 100
    print(t.round(1).sort_values("final_eval%").to_string())

    print("\n=== R5 validation eval blocks: success vs composition ===")
    v = df[df["kind"] == "val"]
    vb = v.groupby(["tag", "block"]).apply(lambda g: pd.Series(dict(
        n=len(g), succ=g["success"].mean() * 100, med_steps=g["steps"].median(),
        succ_ge200=int(g["long_succ"].sum()), rec_share_of_succ=(g.loc[g["success"], "cname"].str.contains("recovery").mean() * 100 if g["success"].any() else np.nan),
        fail_coil=(g.loc[~g["success"], "cname"].str.contains("coil").mean() * 100 if (~g["success"]).any() else np.nan),
        fail_shove=(g.loc[~g["success"], "cname"].str.contains("shove").mean() * 100 if (~g["success"]).any() else np.nan),
        fail_thrash=(g.loc[~g["success"], "cname"].str.contains("thrash").mean() * 100 if (~g["success"]).any() else np.nan),
        fail_near=(g.loc[~g["success"], "cname"].str.contains("near").mean() * 100 if (~g["success"]).any() else np.nan),
        fail_far=(g.loc[~g["success"], "cname"].str.contains("stall-far").mean() * 100 if (~g["success"]).any() else np.nan)))).reset_index()
    print(vb.round(1).to_string())
    vb.to_csv(os.path.join(OUT, "report_val_blocks.csv"), index=False)
    print("\nSpearman across val blocks (n=%d): success vs ..." % len(vb))
    for c in ["med_steps", "succ_ge200", "rec_share_of_succ", "fail_coil", "fail_shove", "fail_thrash", "fail_near", "fail_far"]:
        m = vb[c].notna(); r, p = spearmanr(vb.loc[m, "succ"], vb.loc[m, c]); print("  %-18s rho=%+.2f p=%.3f n=%d" % (c, r, p, m.sum()))

    print("\n=== R6 same success rate, different character: val blocks within 4 pts of success but >=2x median steps ===")
    pairs = []
    for i in range(len(vb)):
        for j in range(i + 1, len(vb)):
            a, b = vb.iloc[i], vb.iloc[j]
            if abs(a.succ - b.succ) <= 4 and a.succ >= 60 and max(a.med_steps, b.med_steps) >= 2 * min(a.med_steps, b.med_steps):
                pairs.append((a.tag, a.block, a.succ, a.med_steps, a.succ_ge200, b.tag, b.block, b.succ, b.med_steps, b.succ_ge200))
    print(pd.DataFrame(pairs, columns=["tagA", "blkA", "succA", "stepsA", "ge200A", "tagB", "blkB", "succB", "stepsB", "ge200B"]).head(25).to_string())

    print("\n=== R7 explore windows -> next validation success ===")
    e = df[(df["kind"] == "explore") & (df["win"] >= 0) & df["next_succ"].notna()]
    ew = e.groupby(["tag", "win"]).apply(lambda g: pd.Series(dict(
        n=len(g), ex_succ=g["success"].mean() * 100, next_succ=g["next_succ"].iloc[0] * 100, prev_succ=g["prev_succ"].iloc[0] * 100 if pd.notna(g["prev_succ"].iloc[0]) else np.nan,
        med_steps=g["steps"].median(), rec_share=g["cname"].str.contains("recovery").mean() * 100,
        rec_share_of_succ=(g.loc[g["success"], "cname"].str.contains("recovery").mean() * 100 if g["success"].any() else np.nan),
        coil=g["cname"].str.contains("coil").mean() * 100, shove=g["cname"].str.contains("shove").mean() * 100,
        thrash=g["cname"].str.contains("thrash").mean() * 100, near=g["cname"].str.contains("near").mean() * 100,
        far=g["cname"].str.contains("stall-far").mean() * 100,
        fail_coil=(g.loc[~g["success"], "cname"].str.contains("coil").mean() * 100 if (~g["success"]).any() else np.nan),
        fail_thrash=(g.loc[~g["success"], "cname"].str.contains("thrash").mean() * 100 if (~g["success"]).any() else np.nan),
        fail_shove=(g.loc[~g["success"], "cname"].str.contains("shove").mean() * 100 if (~g["success"]).any() else np.nan),
        fail_near=(g.loc[~g["success"], "cname"].str.contains("near").mean() * 100 if (~g["success"]).any() else np.nan)))).reset_index()
    ew = ew[ew["n"] >= 100]
    print(ew.round(1).to_string()); ew.to_csv(os.path.join(OUT, "report_explore_windows.csv"), index=False)
    print("\nSpearman across explore windows (n=%d): next_succ vs ..." % len(ew))
    for c in ["ex_succ", "med_steps", "rec_share", "rec_share_of_succ", "coil", "shove", "thrash", "near", "far", "fail_coil", "fail_thrash", "fail_shove", "fail_near"]:
        m = ew[c].notna(); r, p = spearmanr(ew.loc[m, "next_succ"], ew.loc[m, c]); print("  %-18s rho=%+.2f p=%.3f n=%d" % (c, r, p, m.sum()))
    print("within-run (demeaned by tag) Spearman:")
    ewd = ew.copy()
    for c in ["next_succ", "ex_succ", "rec_share", "rec_share_of_succ", "coil", "shove", "thrash", "near", "far", "fail_coil", "fail_thrash", "fail_shove", "fail_near"]:
        ewd[c] = ewd[c] - ewd.groupby("tag")[c].transform("mean")
    for c in ["ex_succ", "rec_share", "rec_share_of_succ", "coil", "shove", "thrash", "near", "far", "fail_coil", "fail_thrash", "fail_shove", "fail_near"]:
        m = ewd[c].notna(); r, p = spearmanr(ewd.loc[m, "next_succ"], ewd.loc[m, c]); print("  %-18s rho=%+.2f p=%.3f" % (c, r, p))

    print("\n=== R8 host-test checkpoints (real patient): success vs composition ===")
    h = df[df["kind"] == "host"].copy()
    m = h["tag"].str.extract(r"host_(topbrain_v\d|p2_teacher_v1bp|tbv1r1|v1bp)_(?:ck|checkpoint)?(\d+|best_checkpoint|H0)(?:_(\d+_\d+))?")
    h["family"] = m[0]; h["ck"] = m[1]; h["sess"] = m[2]
    hb = h.groupby(["tag"]).apply(lambda g: pd.Series(dict(
        family=g["family"].iloc[0], ck=g["ck"].iloc[0], n=len(g), succ=g["success"].mean() * 100, med_steps=g["steps"].median(),
        succ_ge200=int(g["long_succ"].sum()), rec_share_of_succ=(g.loc[g["success"], "cname"].str.contains("recovery").mean() * 100 if g["success"].any() else np.nan),
        fail_coil=(g.loc[~g["success"], "cname"].str.contains("coil").mean() * 100 if (~g["success"]).any() else np.nan),
        fail_shove=(g.loc[~g["success"], "cname"].str.contains("shove").mean() * 100 if (~g["success"]).any() else np.nan),
        fail_thrash=(g.loc[~g["success"], "cname"].str.contains("thrash").mean() * 100 if (~g["success"]).any() else np.nan),
        fail_near=(g.loc[~g["success"], "cname"].str.contains("near").mean() * 100 if (~g["success"]).any() else np.nan),
        fail_far=(g.loc[~g["success"], "cname"].str.contains("stall-far").mean() * 100 if (~g["success"]).any() else np.nan)))).reset_index()
    hb = hb[hb["n"] >= 60].sort_values(["family", "succ"])
    print(hb.round(1).to_string()); hb.to_csv(os.path.join(OUT, "report_host.csv"), index=False)
    print("\nSpearman across host checkpoints (n=%d): success vs ..." % len(hb))
    for c in ["med_steps", "succ_ge200", "rec_share_of_succ", "fail_coil", "fail_shove", "fail_thrash", "fail_near", "fail_far"]:
        mm = hb[c].notna(); r, p = spearmanr(hb.loc[mm, "succ"], hb.loc[mm, c]); print("  %-18s rho=%+.2f p=%.3f n=%d" % (c, r, p, mm.sum()))

    print("\n=== R9 long successes (>=200 steps): where the stall was, how it was escaped ===")
    ls = df[df["long_succ"]].copy()
    ls["p0_frac"] = ls["events"].apply(lambda ev: (max([e["p0"] for e in ev if e["close"] > 0], default=np.nan)))
    ls["p0_frac"] = ls["p0_frac"] / ls["pl"]
    ls["band"] = pd.cut(ls["p0_frac"], [-1, 0.33, 0.66, 0.9, 2], labels=["proximal", "mid", "distal", "at-target"])
    print(pd.crosstab([ls["era"], ls["kind"]], ls["band"]))
    print(ls.groupby(["era", "kind"])[["steps", "ev_n", "ev_ret_max", "withdrawn_mm", "ret_bout_max_mm", "giveback_mm", "ev_last_close_frac"]].median().round(1))
    print("\nfast (<120) vs long (>=200) successes, explore, per era: median retract stats")
    ss = df[(df["kind"] == "explore") & df["success"]].copy(); ss["lc"] = np.where(ss["long_succ"], "long", np.where(ss["fast_succ"], "fast", "mid"))
    print(ss.groupby(["era", "lc"])[["steps", "withdrawn_mm", "ret_bouts", "ret_bout_max_mm", "ev_n", "giveback_mm", "slack_max"]].median().round(1))

    print("\n=== R10 time course of failure composition (explore windows) for the key runs ===")
    for tag in ["v3c2", "tb_v1r2", "car_v3", "car_nopriv", "tb_v2"]:
        t = ew[ew["tag"] == tag][["win", "n", "ex_succ", "next_succ", "med_steps", "rec_share_of_succ", "fail_coil", "fail_shove", "fail_thrash", "fail_near"]]
        if len(t): print("--", tag); print(t.round(1).to_string(index=False))
    df[["key", "tag", "kind", "cluster", "cname", "success", "steps", "block", "win", "next_succ"]].to_csv(os.path.join(OUT, "episodes_labeled.csv"), index=False)


if __name__ == "__main__":
    main()
