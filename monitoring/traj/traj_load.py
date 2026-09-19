"""traj_load.py -- load every *.features.jsonl into one DataFrame and add the
derived episode-class columns used by the descriptive tables and clustering."""
import glob, json, os
import numpy as np, pandas as pd

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")

# run tags whose logs come from training runs (explore + in-run validation evals)
TRAIN_TAGS = ["v3a", "harvest_v3c", "v3c", "v3c2", "v3c3", "tb_v1r1", "tb_v1r2", "tb_repl",
              "tb_v2", "car_v3a", "car_v3", "car_nopriv"]
ERA = {"v3a": "procedural", "harvest_v3c": "procedural", "v3c": "procedural", "v3c2": "procedural",
       "v3c3": "procedural", "tb_v1r1": "topbrain_v1", "tb_v1r2": "topbrain_v1", "tb_repl": "topbrain_v1",
       "tb_v2": "topbrain_v2", "car_v3a": "carotid_v3", "car_v3": "carotid_v3", "car_nopriv": "carotid_v3"}


def make_key(df):
    """Unique episode key. pid+ep is NOT unique: eval workers are respawned per
    validation block and the container reuses pids, and a resumed run restarts
    pid numbering. worker_file + ep + wall-time start is unique."""
    wt = df["wt_start"].fillna(-1).astype(float).round(0).astype("int64").astype(str)
    return df["tag"] + "|" + df["worker_file"].astype(str) + "|" + df["ep"].astype(str) + "|" + wt


def load(only=None):
    rows = []
    for f in sorted(glob.glob(os.path.join(OUT, "*.features.jsonl"))):
        base = os.path.basename(f).replace(".features.jsonl", "")
        if only and not any(base.startswith(o) for o in only): continue
        with open(f) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except ValueError:      # file still being written
                    continue
                r["src"] = base; rows.append(r)
    df = pd.DataFrame(rows)
    df = df[df["complete"] & (df["steps"] >= 2)].copy()
    df["success"] = df["success"].astype(bool)
    df["is_eval"] = df["is_eval"].astype(bool)
    df["kind"] = np.where(df["tag"].str.startswith("host_") | df["tag"].str.startswith("tb22_"), "host",
                          np.where(df["is_eval"], "val", "explore"))
    df["era"] = df["tag"].map(ERA).fillna("host")
    # ---- derived episode classes -------------------------------------------
    cs = df["cath_slack_max"]; sl = df["slack_max"]; lead = df["cath_lead_max"]
    # project's own coil definition (buckle_clear_b_coil_v1): cath_slack>50 OR gw_slack>100
    df["coil"] = (cs.fillna(-1) > 50) | (sl > 100)
    df["cath_shove"] = lead.fillna(-1) > 50            # catheter shoved >50 mm ahead of wire
    df["long_succ"] = df["success"] & (df["steps"] >= 200)
    df["fast_succ"] = df["success"] & (df["steps"] < 120)
    df["len_class"] = pd.cut(df["steps"], [0, 119, 199, 399, 10000], labels=["<120", "120-199", "200-399", ">=400"])
    df["recovered"] = (df["ev_n"] - df["ev_unrec"]) > 0     # >=1 canon stall closed
    df["retract_any"] = df["withdrawn_mm"] > 5.0
    df["deep_retract"] = df["ret_bout_max_mm"] > 20.0
    # failure character, applied to failures only
    fc = np.full(len(df), "", dtype=object)
    fail = ~df["success"].values
    prog = df["prog_max"].values
    fc[fail] = "stable_far"                                # clean partial advance, stopped far from target
    fc[fail & (prog >= 0.75)] = "stable_near"              # got >=75% of the way, timed out near target
    fc[fail & (df["off_final"].values == 1)] = "wrong_branch"
    fc[fail & (df["phys_final"].values == 2)] = "wrong_branch"   # ends physically in RVA
    fc[fail & df["cath_shove"].values] = "cath_shove"
    fc[fail & df["coil"].values] = "coil"
    df["fail_class"] = fc
    return df


def eval_blocks(df, gap_s=1800.0):
    """Assign validation-eval episodes to blocks by wall-time gaps (README rule),
    per run tag. Returns df with 'block' (int, -1 for non-val) and a block table."""
    df = df.copy(); df["block"] = -1
    tabs = []
    for tag, g in df[df["kind"] == "val"].groupby("tag"):
        if g["wt_start"].notna().all():
            g = g.sort_values("wt_start")
            b = (g["wt_start"].diff().fillna(0) > gap_s).cumsum().astype(int)
        else:   # tb_v1r1: three files eval1/2/3, no wall time
            b = g["src"].str.extract(r"eval(\d+)")[0].astype(int) - 1
        df.loc[g.index, "block"] = b.values
        for bi, gb in g.assign(block=b.values).groupby("block"):
            tabs.append(dict(tag=tag, block=bi, n=len(gb), succ=gb["success"].mean(),
                             wt0=gb["wt_start"].min(), wt1=gb["wt_start"].max(),
                             steps_med=gb["steps"].median(),
                             n_long_succ=int(gb["long_succ"].sum()), n_fast_succ=int(gb["fast_succ"].sum()),
                             fail_coil=(gb.loc[~gb["success"], "coil"].mean() if (~gb["success"]).any() else np.nan)))
    return df, pd.DataFrame(tabs)


def explore_windows(df, blocks):
    """Window = explore episodes between consecutive validation blocks of the same run;
    'next_succ' = success rate of the eval that closes the window."""
    df = df.copy(); df["win"] = -1; df["next_succ"] = np.nan; df["prev_succ"] = np.nan
    for tag, bt in blocks.groupby("tag"):
        bt = bt.sort_values("block")
        if bt["wt0"].isna().any(): continue
        ex = df[(df["tag"] == tag) & (df["kind"] == "explore")]
        edges = bt["wt0"].values; succ = bt["succ"].values
        w = np.searchsorted(edges, ex["wt_start"].values)   # number of eval blocks that started before
        df.loc[ex.index, "win"] = w
        nxt = np.where(w < len(succ), succ[np.minimum(w, len(succ) - 1)], np.nan)
        prv = np.where(w > 0, succ[np.maximum(w - 1, 0)], np.nan)
        df.loc[ex.index, "next_succ"] = nxt; df.loc[ex.index, "prev_succ"] = prv
    return df
