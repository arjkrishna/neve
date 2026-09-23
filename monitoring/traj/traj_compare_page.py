"""traj_compare_page.py -- two-model, seed-by-seed comparison page with film links.

For every seed where either model is below 100 % (or where model A formed a loop),
one section: a header with the two per-seed success probabilities, then two columns,
model A's replays on the left, model B's on the right, each replay with outcome, phase,
steps, max progress, catheter slack / lead, withdrawal, tip retreat, loop, and links to
the GIF, trace and key frames rendered by traj_render.py.

usage: python monitoring/traj/traj_compare_page.py <analysis_dir> <anim_dir> <name_a> <name_b>
           [--out compare_<a>_vs_<b>.html] [--seeds 16,31,...] [--title "..."]
"""
import os, sys, glob, csv, argparse, html
import pandas as pd

META = [("T", "steps", "%d"), ("phase", "phase", "%s"), ("p_max", "max progress", "%.2f"), ("cath_slack_max", "cath slack max", "%.0f mm"),
        ("cath_lead_max", "cath lead max", "%.0f mm"), ("withdrawn", "withdrawn", "%.0f mm"), ("tip_retreat_max", "tip retreat", "%.0f mm"),
        ("n_events", "stalls", "%d"), ("loop", "loop", "%s"), ("cname", "atlas type", "%s")]


def film_index(anim):
    idx = {}
    for f in glob.glob(os.path.join(anim, "*", "index.csv")):
        tag = os.path.basename(os.path.dirname(f))
        for r in csv.DictReader(open(f, encoding="utf-8")):
            try: idx[(tag, int(r["seed"]))] = r["dir"].replace("\\", "/")
            except (KeyError, ValueError): pass
    return idx


def fmt(v, f):
    try:
        if v != v: return "-"
        if f == "%s" and isinstance(v, (bool,)): return "yes" if v else "no"
        if f == "%d": return "%d" % int(v)
        return f % v
    except Exception:
        return str(v)


def cell(e, idx, rel):
    d = idx.get((e["tag"], int(e["seed"])))
    links = ("<a href='%s/%s/episode.gif'>gif</a> &middot; <a href='%s/%s/trace.png'>trace</a> &middot; <a href='%s/%s/key/'>frames</a>"
             % (rel, d, rel, d, rel, d)) if d else "<i>(not rendered yet)</i>"
    meta = " &middot; ".join("<b>%s</b> %s" % (lab, html.escape(fmt(e.get(k), f))) for k, lab, f in META if k in e)
    cls = "ok" if e["success"] else "fail"
    return "<div class='ep %s'><div class='hd'>rep %d &mdash; %s</div><div class='meta'>%s</div><div class='links'>%s</div></div>" % (
        cls, int(e["rep"]), "SUCCESS" if e["success"] else "FAIL", meta, links)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("analysis"); ap.add_argument("anim"); ap.add_argument("a"); ap.add_argument("b")
    ap.add_argument("--out", default=""); ap.add_argument("--seeds", default=""); ap.add_argument("--title", default="")
    ap.add_argument("--desc-a", default=""); ap.add_argument("--desc-b", default="")
    x = ap.parse_args()
    ep = pd.read_csv(os.path.join(x.analysis, "episodes.csv"))
    ep = ep[ep.name.isin([x.a, x.b])]
    if "loop" not in ep.columns: ep["loop"] = False
    out = x.out or os.path.join(x.analysis, "compare_%s_vs_%s.html" % (x.a, x.b))
    idx = film_index(x.anim)
    try: rel = os.path.relpath(x.anim, os.path.dirname(os.path.abspath(out))).replace("\\", "/")
    except ValueError: rel = "file:///" + os.path.abspath(x.anim).replace("\\", "/")
    p = ep.pivot_table(index="seed", columns="name", values="success", aggfunc="mean")
    lp = ep.pivot_table(index="seed", columns="name", values="loop", aggfunc="max")
    if x.seeds:
        seeds = [int(s) for s in x.seeds.split(",") if s]
    else:
        seeds = [s for s in p.index if p.loc[s, x.a] < 1 or p.loc[s, x.b] < 1 or bool(lp.loc[s, x.a])]
        seeds.sort(key=lambda s: (p.loc[s, x.a] - p.loc[s, x.b], p.loc[s, x.a]))
    mesh = ep.groupby("seed").mesh.first()
    top = lambda s: s.value_counts().index[0] if len(s) else "-"
    css = ("body{font:14px system-ui;margin:24px;max-width:1400px}h1{margin-bottom:4px}h2{margin:30px 0 4px}"
           ".sub{color:#444;margin:0 0 10px}table.sum{border-collapse:collapse;margin:10px 0 20px}table.sum td,table.sum th{padding:3px 10px;border-bottom:1px solid #ddd;text-align:left}"
           ".cols{display:grid;grid-template-columns:1fr 1fr;gap:14px}.col h3{margin:0 0 6px;font-size:14px}"
           ".ep{border:1px solid #ddd;border-radius:6px;padding:6px 10px;margin:0 0 8px}.ep.fail{background:#fdecea}.ep.ok{background:#eef8ee}"
           ".hd{font-weight:600}.meta{font-size:12.5px;color:#333;margin:3px 0}.links a{margin-right:4px}")
    title = x.title or "%s vs %s, seed by seed" % (x.a, x.b)
    H = ["<!doctype html><meta charset='utf-8'><title>%s</title><style>%s</style>" % (html.escape(title), css), "<h1>%s</h1>" % html.escape(title),
         "<p class='sub'>Left: <b>%s</b>%s. Right: <b>%s</b>%s. Each box is one replayed episode (same seed, independent device twist); "
         "red = failure, green = success. Numbers after the seed are the per-model success probabilities over the repetitions.</p>" % (
             x.a, (" (" + html.escape(x.desc_a) + ")") if x.desc_a else "", x.b, (" (" + html.escape(x.desc_b) + ")") if x.desc_b else "")]
    # summary table
    H.append("<table class='sum'><tr><th>seed</th><th>anatomy</th><th>%s</th><th>%s</th><th>%s: how it fails</th><th>%s: how it wins</th><th>%s: how it wins</th></tr>" % (x.a, x.b, x.a, x.a, x.b))
    for s in seeds:
        ea = ep[(ep.seed == s) & (ep.name == x.a)]; eb = ep[(ep.seed == s) & (ep.name == x.b)]
        H.append("<tr><td><a href='#s%d'>%d</a></td><td>%s</td><td>%.2f</td><td>%.2f</td><td>%s</td><td>%s</td><td>%s</td></tr>" % (
            s, s, mesh.get(s, ""), p.loc[s, x.a], p.loc[s, x.b], top(ea[~ea.success].phase), top(ea[ea.success].phase), top(eb[eb.success].phase)))
    H.append("</table>")
    for s in seeds:
        H.append("<h2 id='s%d'>seed %d &mdash; %s</h2><p class='sub'>%s %.2f &nbsp;&middot;&nbsp; %s %.2f</p><div class='cols'>" % (
            s, s, mesh.get(s, ""), x.a, p.loc[s, x.a], x.b, p.loc[s, x.b]))
        for nm in (x.a, x.b):
            H.append("<div class='col'><h3>%s</h3>" % nm)
            for _, e in ep[(ep.seed == s) & (ep.name == nm)].sort_values("rep").iterrows():
                H.append(cell(e, idx, rel))
            H.append("</div>")
        H.append("</div>")
    open(out, "w", encoding="utf-8").write("\n".join(H))
    print("wrote", out, "seeds:", seeds)


if __name__ == "__main__":
    main()
