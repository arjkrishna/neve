"""traj_frontier.py -- analyse a replay campaign (run_campaign.sh results) seed by seed.

For every recorded episode: atlas type, outcome, stall/coil signals. Then, per
checkpoint ("name"), per seed, success probability over the repetitions; seed
classes (easy / frontier / floor) from the pooled reps; paired comparisons between
checkpoints on the common seeds; recovery share of frontier successes; failure
type shares (coil / buckle-fight / thrash / shove). Writes CSVs + a markdown summary
+ a render list of frontier-seed records for traj_render.py.

usage: python monitoring/traj/traj_frontier.py <results.tsv> [<results.tsv> ...] --out <dir>
       [--pairs car_best:abl_best,car_best:car_peak] [--reps-min 1]
"""
import os, sys, csv, glob, json, argparse
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import traj_render as tr  # noqa: E402

MAP = {"/opt/eve_training/results16": "D:/neve/.claude/worktrees/rl_improv_16_resume/saved",
       "/opt/eve_training/results": "D:/neve/.claude/worktrees/rl_improv_18_p2/saved"}

# Second-pass modules (written by the taxonomy / indicator agents); optional.
sys.path.insert(0, os.path.join(HERE, "secondpass"))
try:
    from phase_taxonomy import phase_type as _phase_type
except Exception:
    _phase_type = None
try:
    import indicators as _ind
except Exception:
    _ind = None


def second_pass_cols(r, feat, row):
    """Add phase-taxonomy and coil / catheter-led indicator columns when the
    second-pass modules exist; every call is guarded so a missing or failing
    module never breaks the campaign analysis."""
    base = dict(feat); base.update(dict(success=row["success"], steps=r["T"], pl=float(r["path_len"]),
                                       events=feat.get("events", []), cath_slack_max=row.get("cath_slack_max"),
                                       cath_lead_max=row.get("cath_lead_max"), slack_max=row.get("slack_max")))
    out = {}
    if _phase_type is not None:
        try: out["phase"] = str(_phase_type(base))
        except Exception as e: out["phase"] = "err:%s" % type(e).__name__
        # A success whose last stall is still "open" ended in the escape itself (the
        # success terminated the episode before the detector could close the event):
        # it is a recovery that finished late, not a doorstep stall / shove cap.
        if row["success"] and out["phase"] in OPEN_STALL:
            out["phase_raw"] = out["phase"]; out["late_escape"] = True
            hard = getattr(sys.modules.get("phase_taxonomy"), "HARD_MM", 8.0)
            ret = feat.get("ev_ret_max", feat.get("withdrawn_mm", 0.0)) or 0.0
            out["phase"] = "deep-recovery" if float(ret) > hard else "light-recovery"
    if _ind is not None:
        for fn, prefix, arg in (("catheter_led", "cl_", base), ("coil_from_features", "coilf_", base), ("coil_from_record", "coilr_", r)):
            f = getattr(_ind, fn, None)
            if f is None: continue
            try:
                d = f(arg)
                if isinstance(d, dict): out.update({prefix + k: v for k, v in d.items()})
            except Exception as e:
                out[prefix + "err"] = type(e).__name__
    return out
REC = {3, 4}; CLEAN = {0, 1, 2}; COIL = {8}; BUCKLE = {6}; THRASH = {7, 10}; SHOVE = {5, 9}
OPEN_STALL = {"doorstep-stall", "shove-capped", "coiled-stuck", "stuck-proximal", "stuck-midpath"}
REC_PH = {"light-recovery", "deep-recovery", "fight-escaped-near", "fight-escaped-far"}
PH_GROUP = {"free-run": "free", "idle-creep": "free", "cath-led-run": "cathled", "light-recovery": "light-rec",
            "deep-recovery": "deep-rec", "fight-escaped-near": "fight", "fight-escaped-far": "fight"}


def md_table(df):
    """DataFrame -> markdown table without tabulate."""
    d = df.reset_index() if df.index.name or not isinstance(df.index, pd.RangeIndex) else df
    cols = [str(c) for c in d.columns]
    rows = [[("%.3f" % v if isinstance(v, float) else str(v)) for v in r] for r in d.itertuples(index=False)]
    return "\n".join(["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] + ["| " + " | ".join(r) + " |" for r in rows])


def write_decisive_html(anim, out, ep, st, names):
    """<out>/decisive.html: one section per non-easy seed, one row per replayed episode,
    linking the GIF, the trace and the key frames rendered by traj_render.py into <anim>."""
    idx = {}
    for f in glob.glob(os.path.join(anim, "*", "index.csv")):
        tag = os.path.basename(os.path.dirname(f))
        for r in csv.DictReader(open(f, encoding="utf-8")):
            try: idx[(tag, int(r["seed"]))] = r["dir"].replace("\\", "/")
            except (KeyError, ValueError): pass
    try: rel = os.path.relpath(anim, out).replace("\\", "/")
    except ValueError: rel = "file:///" + os.path.abspath(anim).replace("\\", "/")   # different drive
    css = ("body{font:14px system-ui;margin:24px;max-width:1200px}table{border-collapse:collapse;margin:6px 0 18px}"
           "td,th{padding:3px 10px;border-bottom:1px solid #ddd;text-align:left}tr.fail td{background:#fdecea}"
           "tr.ok td{background:#eef8ee}h2{margin:26px 0 4px}p.p{margin:0 0 6px;color:#444}.cls{font-weight:600}")
    H = ["<!doctype html><meta charset='utf-8'><title>decisive seeds</title><style>%s</style>" % css,
         "<h1>Decisive seeds: every replay, with its film</h1>",
         "<p>easy = every checkpoint always solves it (not listed); strength = all strong checkpoints always solve it but a weak one fails it; "
         "frontier = a strong checkpoint fails it in some repetition; floor = no strong checkpoint ever solves it. "
         "Numbers after the seed are the per-checkpoint success probabilities over the repetitions.</p>"]
    for seed, row in st[st.cls != "easy"].iterrows():
        probs = " &middot; ".join("%s %.2f" % (n, row[n]) for n in names if n in row.index and pd.notna(row[n]))
        H.append("<h2>seed %d <span class='cls'>[%s]</span> &mdash; %s</h2><p class='p'>%s</p>" % (int(seed), row.cls, row.get("mesh", ""), probs))
        H.append("<table><tr><th>checkpoint</th><th>rep</th><th>outcome</th><th>phase</th><th>steps</th><th>atlas type</th><th>film</th></tr>")
        for _, e in ep[ep.seed == seed].sort_values(["name", "rep"]).iterrows():
            d = idx.get((e["tag"], int(seed)))
            links = ("<a href='%s/%s/episode.gif'>gif</a> &middot; <a href='%s/%s/trace.png'>trace</a> &middot; <a href='%s/%s/key/'>key frames</a>"
                     % (rel, d, rel, d, rel, d)) if d else "(not rendered)"
            H.append("<tr class='%s'><td>%s</td><td>%d</td><td>%s</td><td>%s</td><td>%d</td><td>%s</td><td>%s</td></tr>" % (
                "ok" if e["success"] else "fail", e["name"], int(e["rep"]), "success" if e["success"] else "FAIL",
                e.get("phase", ""), int(e["T"]), e["cname"], links))
        H.append("</table>")
    open(os.path.join(out, "decisive.html"), "w", encoding="utf-8").write("\n".join(H))


def host_path(p):
    for k, v in sorted(MAP.items(), key=lambda kv: -len(kv[0])):
        if p.startswith(k): return v + p[len(k):]
    return p


def load_results(files):
    rows = []
    for f in files:
        for line in open(f):
            t = line.rstrip("\n").split("\t")
            if len(t) < 9: continue
            rows.append(dict(tag=t[0], name=t[1], rep=int(t[2]), ckpt=t[3], rec=host_path(t[4]), official=t[5], t0=t[6], t1=t[7], rc=t[8]))
    return rows


def episodes(run):
    out = []
    for f in sorted(glob.glob(os.path.join(run["rec"], "*.npz"))):
        try:
            r = tr.load_record(f)
        except Exception as e:
            sys.stderr.write("skip %s: %s\n" % (f, e)); continue
        if r["T"] < 3 or r["seed"] < 0: continue
        cid, cname, feat, ev = tr.type_record(r)
        slack = (r["ins_gw"] - r["proj_s"]); cs = r["cath_slack"]
        row = dict(name=run["name"], rep=run["rep"], tag=run["tag"], seed=r["seed"], mesh=r["mesh_fp"], T=r["T"],
                   success=bool(r["success"]), reason=r["reason"], type=cid, cname=cname,
                   n_events=len(ev), n_unrec=sum(e["close"] <= 0 for e in ev),
                   slack_max=float(np.nanmax(slack)), cath_slack_max=float(np.nanmax(cs)) if np.isfinite(cs).any() else np.nan,
                   cath_lead_max=float(np.nanmax(r["ins_cath"] - r["ins_gw"])), withdrawn=float(feat["withdrawn_mm"]),
                   coil=bool((np.isfinite(cs).any() and np.nanmax(cs) > 50) or np.nanmax(slack) > 100),
                   # explicit catheter-lead measurements (the k-means "cathled sprint" type also
                   # captures merely-fast clean episodes with 8-30 mm lead; do not score
                   # catheter-led behaviour on the type id)
                   lead_end=float((r["ins_cath"] - r["ins_gw"])[-1]),
                   lead_frac20=float(np.mean((r["ins_cath"] - r["ins_gw"]) > 20)),
                   lead_frac50=float(np.mean((r["ins_cath"] - r["ins_gw"]) > 50)),
                   file=f)
        tr_max, tr_steps = np.nan, np.nan
        try:
            path = np.asarray(r["path"], float); gwp = np.asarray(r["gw"], float)
            if path.ndim == 2 and len(path) > 1 and gwp.ndim == 3:
                cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])
                tipg = gwp[:, 0, :]                                   # node 0 is the tip
                dd = np.linalg.norm(tipg[:, None, :] - path[None, :, :], axis=2)
                pg = cum[np.nanargmin(np.where(np.isnan(dd), np.inf, dd), axis=1)]
                back = np.maximum.accumulate(pg) - pg
                tr_max = float(np.nanmax(back)); tr_steps = int(np.sum(back > 3.0))
        except Exception:
            pass
        row.update(tip_retreat_max=tr_max, tip_retreat_steps=tr_steps)
        ps = np.asarray(r["proj_s"], float); pl = float(r["path_len"])
        pf = ps / pl if pl > 0 else ps * np.nan
        imax = int(np.nanargmax(pf)); adv = np.where(np.diff(ps) > 1.0)[0]
        cmd = np.asarray(r["cmd"], float)
        row.update(p_end=float(pf[-1]), p_max=float(pf[imax]), t_pmax=imax,
                   idle_since=int(r["T"] - 1 - adv.max()) if adv.size else int(r["T"]),
                   push_frac=float(np.mean(np.abs(cmd[:, 0]) > 2)) if cmd.ndim == 2 and cmd.shape[1] >= 1 else np.nan,
                   loop=bool(False))
        feat["events"] = ev
        row.update(second_pass_cols(r, feat, row))
        if "coilr_loop_any_gw" in row or "coilr_loop_any_cath" in row:
            row["loop"] = bool(row.get("coilr_loop_any_gw", False)) or bool(row.get("coilr_loop_any_cath", False))
        out.append(row)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+"); ap.add_argument("--out", required=True)
    ap.add_argument("--pairs", default=""); ap.add_argument("--reps-min", type=int, default=1)
    ap.add_argument("--strong", default="", help="comma list of checkpoint names that define the frontier (default: success >= 0.9)")
    ap.add_argument("--weak", default="", help="comma list of weaker checkpoints (default: every other name)")
    ap.add_argument("--anim", default="", help="rendered campaign dir (traj_render --block outputs); writes decisive.html with GIF links")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    runs = load_results(a.results)
    ep = pd.DataFrame([e for run in runs for e in episodes(run)])
    if ep.empty: sys.exit("no episodes found")
    ep.to_csv(os.path.join(a.out, "episodes.csv"), index=False)
    ep["rec_succ"] = ep.success & ep.type.isin(REC); ep["clean_succ"] = ep.success & ep.type.isin(CLEAN)
    ep["fight_succ"] = ep.success & ~ep.type.isin(REC | CLEAN)
    for k, S in (("coil", COIL), ("buckle", BUCKLE), ("thrash", THRASH), ("shove", SHOVE)):
        ep["fail_" + k] = ~ep.success & ep.type.isin(S)

    # T1 per checkpoint
    t1 = ep.groupby("name").agg(reps=("rep", "nunique"), episodes=("seed", "size"), seeds=("seed", "nunique"), succ=("success", "mean"),
                                clean=("clean_succ", "mean"), recovery=("rec_succ", "mean"), fight_won=("fight_succ", "mean"),
                                f_coil=("fail_coil", "mean"), f_buckle=("fail_buckle", "mean"), f_thrash=("fail_thrash", "mean"), f_shove=("fail_shove", "mean"),
                                med_steps=("T", "median"))
    t1["lead_end_med"] = ep.groupby("name").lead_end.median(); t1["lead_frac50"] = ep.groupby("name").lead_frac50.mean()
    t1["cathled_succ"] = ep[ep.success].groupby("name").apply(lambda g: float(((g.lead_end > 20) & (g.lead_frac20 > 0.5)).mean()))
    t1["rec_share_of_succ"] = ep[ep.success].groupby("name").rec_succ.mean()
    if "phase" in ep.columns and "tip_retreat_max" in ep.columns:
        RS = ep[ep.success & ep.phase.isin(REC_PH)]
        t1["rec_tipretreat_share"] = RS.groupby("name").tip_retreat_max.apply(lambda x: float((x > 8).mean()))
        t1["tipretreat_med_rec"] = RS.groupby("name").tip_retreat_max.median()
    t1["coil_share_of_fail"] = ep[~ep.success].groupby("name").fail_coil.mean()
    t1["buckle_share_of_fail"] = ep[~ep.success].groupby("name").fail_buckle.mean()
    t1["thrash_share_of_fail"] = ep[~ep.success].groupby("name").fail_thrash.mean()
    F0 = ep[~ep.success]
    if len(F0):
        t1["fail_pmax_med"] = F0.groupby("name").p_max.median()
        t1["fail_midpath_share"] = F0.groupby("name").p_max.apply(lambda x: float((x < 0.66).mean()))
        t1["fail_loop_share"] = F0.groupby("name").loop.mean()
        t1["loop_any"] = ep.groupby("name").loop.mean()
        t1["loop_escaped"] = ep[ep.loop].groupby("name").success.mean() if ep.loop.any() else np.nan
    t1 = t1.round(3); t1.to_csv(os.path.join(a.out, "per_checkpoint.csv"))
    fcols = [c for c in ["name", "rep", "seed", "mesh", "T", "phase", "cname", "p_end", "p_max", "t_pmax", "idle_since", "push_frac",
                         "withdrawn", "tip_retreat_max", "n_events", "n_unrec", "lead_end", "cath_slack_max", "slack_max", "loop", "file"] if c in ep.columns]
    F0.sort_values(["seed", "name", "rep"])[fcols].round(3).to_csv(os.path.join(a.out, "failures.csv"), index=False)
    # second-pass views, when the modules were available
    if "phase" in ep.columns:
        pd.crosstab([ep.name, ep.success], ep.phase).to_csv(os.path.join(a.out, "phase_by_checkpoint.csv"))
    ind_cols = [c for c in ep.columns if c.startswith(("cl_", "coilf_", "coilr_")) and pd.api.types.is_numeric_dtype(ep[c])]
    if ind_cols:
        ep.groupby(["name", "success"])[ind_cols].mean().round(3).to_csv(os.path.join(a.out, "indicators_by_checkpoint.csv"))

    # T2 per seed x checkpoint: success probability over reps + dominant type
    ps = ep.groupby(["seed", "name"]).agg(n=("rep", "size"), p=("success", "mean"), p_rec=("rec_succ", "mean"),
                                          dom=("cname", lambda s: s.value_counts().index[0]), mesh=("mesh", "first")).reset_index()
    piv = ps.pivot(index="seed", columns="name", values="p")
    piv.to_csv(os.path.join(a.out, "seed_by_checkpoint.csv"))
    ps.to_csv(os.path.join(a.out, "seed_checkpoint_long.csv"), index=False)
    # seed classes.  strong = the checkpoints whose frontier we want to measure; weak = the rest.
    #   easy      : every checkpoint (strong and weak) solves it in every repetition
    #   strength  : all strong checkpoints always solve it, a weak one fails it (measures 76->95 %, not the frontier)
    #   frontier  : a strong checkpoint fails it in some repetition while some strong checkpoint solves it
    #   floor     : no strong checkpoint ever solves it
    names = list(t1.index)
    strong = [n for n in a.strong.split(",") if n] or [n for n in names if t1.loc[n, "succ"] >= 0.9]
    weak = [n for n in a.weak.split(",") if n] or [n for n in names if n not in strong]
    strong = [n for n in strong if n in piv.columns]; weak = [n for n in weak if n in piv.columns]
    pooled = ep.groupby("seed").agg(n=("success", "size"), p=("success", "mean"), clean=("clean_succ", "mean"), rec=("rec_succ", "mean"))
    pooled["min_strong"] = piv[strong].min(axis=1); pooled["max_strong"] = piv[strong].max(axis=1)
    pooled["spread_strong"] = pooled.max_strong - pooled.min_strong
    pooled["min_weak"] = piv[weak].min(axis=1) if weak else 1.0
    def _cls(r):
        if r.max_strong == 0: return "floor"
        if r.min_strong == 1: return "easy" if r.min_weak == 1 else "strength"
        return "frontier"
    pooled["cls"] = pooled.apply(_cls, axis=1)
    pooled.to_csv(os.path.join(a.out, "seed_classes.csv"))
    # per-seed table: p per checkpoint, how it is solved (phase), how it fails
    st = piv.copy()
    if "phase" in ep.columns:
        ep["pgrp"] = ep.phase.map(PH_GROUP).fillna("stall")
        top = lambda x: x.value_counts().index[0]
        st = st.join(ep[ep.success].groupby("seed").phase.agg(top).rename("succ_phase"))
        st = st.join(ep[~ep.success].groupby("seed").phase.agg(top).rename("fail_phase"))
        st = st.join(ep[ep.success].groupby("seed").phase.agg(lambda x: float(x.isin(REC_PH).mean())).rename("rec_share"))
        st = st.join(ep[ep.success].groupby("seed").phase.agg(lambda x: float((x == "cath-led-run").mean())).rename("cathled_share"))
    st["medT"] = ep.groupby("seed").T.median(); st["mesh"] = ep.groupby("seed").mesh.first()
    st["cls"] = pooled.cls; st["min_strong"] = pooled.min_strong; st["spread_strong"] = pooled.spread_strong
    st = st.sort_values(["cls", "min_strong"])
    st.to_csv(os.path.join(a.out, "seed_table.csv"))
    if a.anim and os.path.isdir(a.anim):
        write_decisive_html(a.anim, a.out, ep, st, names)
    rb = fb = None
    if "phase" in ep.columns:
        ep["seed_cls"] = ep.seed.map(pooled.cls)
        S = ep[ep.success]; F = ep[~ep.success]
        rb = pd.crosstab([S.name, S.seed_cls], S.pgrp, normalize="index").round(3)
        rb["n_succ"] = pd.crosstab([S.name, S.seed_cls], S.pgrp).sum(axis=1)
        rb.to_csv(os.path.join(a.out, "recovery_by_class.csv"))
        fb = pd.crosstab(F.name, F.phase); fb.to_csv(os.path.join(a.out, "failures_by_phase.csv"))

    # T3 paired comparisons (per-seed success probabilities)
    lines = []
    pairs = [p.split(":") for p in a.pairs.split(",") if ":" in p]
    if not pairs:
        names = list(t1.index); pairs = [(names[0], n) for n in names[1:]]
    for x, y in pairs:
        if x not in piv.columns or y not in piv.columns: continue
        d = (piv[x] - piv[y]).dropna(); nz = d[d != 0]
        wins = int((nz > 0).sum()); losses = int((nz < 0).sum())
        from scipy.stats import binomtest, wilcoxon
        pb = binomtest(min(wins, losses), wins + losses, 0.5).pvalue if wins + losses else float("nan")
        try: pw = wilcoxon(nz).pvalue if len(nz) >= 5 else float("nan")
        except Exception: pw = float("nan")
        lines.append(dict(a=x, b=y, seeds=len(d), mean_a=round(float(piv.loc[d.index, x].mean()), 3), mean_b=round(float(piv.loc[d.index, y].mean()), 3),
                          mean_diff=round(float(d.mean()), 3), seeds_a_better=wins, seeds_b_better=losses, sign_p=round(pb, 4), wilcoxon_p=round(pw, 4) if pw == pw else None,
                          a_better=[int(s) for s in nz[nz > 0].index], b_better=[int(s) for s in nz[nz < 0].index]))
    pd.DataFrame(lines).to_csv(os.path.join(a.out, "paired.csv"), index=False)

    # render list: frontier seeds only
    fr = set(pooled[pooled.cls != "easy"].index)
    ep[ep.seed.isin(fr)][["name", "rep", "seed", "cname", "success", "file"]].to_csv(os.path.join(a.out, "render_list.csv"), index=False)

    # markdown summary
    md = ["# Replay campaign summary", "", "## Per checkpoint (all replayed seeds)", "", md_table(t1), "",
          "## Seed classes over pooled reps: " + ", ".join("%s %d" % (k, v) for k, v in pooled.cls.value_counts().items()), "",
          "## Paired comparisons (per-seed success probability)", ""]
    for l in lines:
        md.append("- **%s vs %s** (n=%d common seeds): %.3f vs %.3f, mean diff %+.3f; seeds better under a: %d, under b: %d; sign p=%.4f%s" % (
            l["a"], l["b"], l["seeds"], l["mean_a"], l["mean_b"], l["mean_diff"], l["seeds_a_better"], l["seeds_b_better"], l["sign_p"],
            ("; wilcoxon p=%.4f" % l["wilcoxon_p"]) if l["wilcoxon_p"] is not None else ""))
        md.append("  - a better on seeds %s; b better on %s" % (l["a_better"], l["b_better"]))
    md += ["", "## Seed classes (strong = %s; weak = %s)" % (", ".join(strong), ", ".join(weak) or "none"), "",
           "easy = every checkpoint always solves it; strength = all strong checkpoints always solve it but a weak one fails it; "
           "frontier = a strong checkpoint fails it in some repetition; floor = no strong checkpoint ever solves it.", "",
           "## Decisive seeds (frontier + strength + floor): success probability per checkpoint", "",
           md_table(st[st.cls != "easy"].drop(columns=["mesh"]).round(2))]
    if rb is not None:
        md += ["", "## How successes are won, by seed class (share of successes per phase group)", "", md_table(rb.reset_index()),
               "", "## How failures end (phase of the failure)", "", md_table(fb.reset_index())]
    open(os.path.join(a.out, "SUMMARY.md"), "w").write("\n".join(md))
    print(t1.to_string()); print("\nseed classes:", pooled.cls.value_counts().to_dict())
    for l in lines: print("PAIR %s vs %s: %.3f vs %.3f  a-better %d  b-better %d  sign p=%.4f" % (l["a"], l["b"], l["mean_a"], l["mean_b"], l["seeds_a_better"], l["seeds_b_better"], l["sign_p"]))


if __name__ == "__main__":
    main()
