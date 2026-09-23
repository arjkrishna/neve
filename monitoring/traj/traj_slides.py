"""traj_slides.py -- 16:9 PNG slides from the frontier replay campaign.

Pair slides put the carotid v3 best checkpoint (CB3) next to another model on the SAME seed,
zoomed into the nook where the other model fails, with the local vessel wall (collision mesh),
the planned path, both devices, and the progress / slack trace of each episode. Definition
slides show the behaviour phases on real frames; summary slides carry the numbers.

usage: python monitoring/traj/traj_slides.py [--out <dir>] [--only 04,05] [--half 45]
"""
import os, re, sys, glob, argparse, textwrap
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import traj_render as tr  # noqa: E402
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection  # noqa: E402

REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
CAMP = os.path.join(REPO, "saved", "eve_paper", "neurovascular", "full", "mesh_ben", "frontier_campaign")
ANAT = os.path.join(REPO, "carotid_data", "anatomies_v3")
MESH_BEN = os.path.join(CAMP, "slides", "assets", "mesh_ben.obj")

C = dict(bg="#f4f3ef", ink="#1d1f24", muted="#5f6670", rule="#d9d6cf",
         cb3="#1b6a8f", cb3_bg="#dfeaf1", other="#a63a3a", other_bg="#f3e3e1",
         wire="#00a34a", cath="#c62828", path="#f0a500", target="#ffd400", cl="#6b8bab", mesh="#8a97a5",
         stall="#f3d9a4", prog="#1d1f24", cslack="#c62828", gslack="#3b6ea5", ok="#2e7d32", fail="#a63a3a")
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": C["rule"], "text.color": C["ink"],
                     "axes.labelcolor": C["ink"], "xtick.color": C["muted"], "ytick.color": C["muted"]})

_MESH = {}


# ------------------------------------------------------------------ data access
def anat_dir(mesh_fp):
    m = re.match(r"case([a-z])(\d{3})(left|right)topcowmr(\d{3})(L?)$", mesh_fp)
    if not m: return None
    d = os.path.join(ANAT, "case_%s_%s_%s__topcow_mr_%s%s" % (m[1], m[2], m[3], m[4], "_L" if m[5] else ""))
    return d if os.path.isdir(d) else None


def load_obj(path):
    if path in _MESH: return _MESH[path]
    V, F = [], []
    for line in open(path):
        if line.startswith("v "): V.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("f "): F.append([int(t.split("/")[0]) - 1 for t in line.split()[1:4]])
    _MESH[path] = (np.asarray(V, float), np.asarray(F, int)); return _MESH[path]


def mesh_for(r, family):
    if family == "host": return load_obj(MESH_BEN) if os.path.exists(MESH_BEN) else None
    d = anat_dir(r["mesh_fp"])
    p = os.path.join(d, "vessel_architecture_collision.obj") if d else None
    return load_obj(p) if p and os.path.exists(p) else None


def episodes(analysis_dir):
    return pd.read_csv(os.path.join(analysis_dir, "episodes.csv"))


def record(df, name, rep, seed):
    row = df[(df.name == name) & (df.rep == rep) & (df.seed == seed)]
    if row.empty: raise KeyError("no episode %s rep%d seed%d" % (name, rep, seed))
    r = tr.load_record(row.iloc[0].file); r["row"] = row.iloc[0]
    _, cname, feat, ev = tr.type_record(r); r["ev"] = ev; r["feat"] = feat; r["cname"] = cname
    return r


# ------------------------------------------------------------------ geometry
def path_cum(r):
    P = np.asarray(r["path"], float)
    return P, np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])


def point_at(r, s):
    P, cum = path_cum(r); s = float(np.clip(s, 0, cum[-1])); i = int(np.searchsorted(cum, s))
    if i <= 0: return P[0]
    if i >= len(P): return P[-1]
    w = (s - cum[i - 1]) / max(cum[i] - cum[i - 1], 1e-6); return P[i - 1] + w * (P[i] - P[i - 1])


def nook_of(r):
    """3-D centre of the nook: onset of the episode's last stall, else the point of max progress."""
    ev = r["ev"]
    if ev:
        return point_at(r, float(ev[-1]["p0"])), float(ev[-1]["p0"])
    t = int(np.nanargmax(r["proj_s"])); s = float(r["proj_s"][t])
    return point_at(r, s), s


def view_for(r, center, half):
    """Look along the normal of the local path plane (so the bend is seen face-on)."""
    P = np.asarray(r["path"], float); sel = np.all(np.abs(P - center) <= half, axis=1)
    Q = P[sel] if sel.sum() >= 4 else P
    Q = Q - Q.mean(0); _, _, vt = np.linalg.svd(Q, full_matrices=False); n = vt[-1]
    if n[2] < 0: n = -n
    elev = float(np.degrees(np.arcsin(np.clip(n[2], -1, 1)))); azim = float(np.degrees(np.arctan2(n[1], n[0])))
    return min(elev, 55.0), azim


def clip(P, center, half):
    P = np.asarray(P, float).copy(); out = np.any(np.abs(P - center) > half, axis=1); P[out] = np.nan; return P


def draw_nook(ax, r, t, center, half, mesh, view, trail=25, mesh_alpha=0.22, zoom=1.75):
    if mesh is not None:
        V, F = mesh; cen = V[F].mean(1); sel = np.all(np.abs(cen - center) <= half * 1.1, axis=1)
        if sel.any():
            ax.add_collection3d(Poly3DCollection(V[F[sel]], facecolors=C["mesh"], alpha=mesh_alpha, edgecolors="#5b6673", linewidths=0.15))
    for _, c, on in r["cl"]:
        q = clip(c, center, half)
        if np.isfinite(q).any(): ax.plot(q[:, 0], q[:, 1], q[:, 2], color=C["cl"], lw=0.9 if on else 0.6, alpha=0.9 if on else 0.4)
    q = clip(r["path"], center, half)
    if np.isfinite(q).any(): ax.plot(q[:, 0], q[:, 1], q[:, 2], color=C["path"], lw=2.2, alpha=0.95)
    tg = np.asarray(r["target"], float)
    if np.isfinite(tg).all() and np.all(np.abs(tg - center) <= half):
        ax.scatter(*tg, color=C["target"], marker="X", s=140, edgecolors="k", linewidths=0.7, depthshade=False)
    a = max(0, t - trail); tp = clip(r["tip"][a:t + 1], center, half)
    if np.isfinite(tp).any(): ax.plot(tp[:, 0], tp[:, 1], tp[:, 2], color="#777", lw=0.9, alpha=0.7)
    for key, col, lw in (("cath", C["cath"], 3.4), ("gw", C["wire"], 2.6)):
        p = clip(r[key][t], center, half)
        if np.isfinite(p).any():
            ax.plot(p[:, 0], p[:, 1], p[:, 2], color=col, lw=lw, solid_capstyle="round")
            p0 = np.asarray(r[key][t][0], float)
            if np.all(np.abs(p0 - center) <= half):
                ax.scatter(*p0, color=col, s=46, edgecolors="k", linewidths=0.6, depthshade=False)
    ax.set_xlim(center[0] - half, center[0] + half); ax.set_ylim(center[1] - half, center[1] + half); ax.set_zlim(center[2] - half, center[2] + half)
    ax.set_box_aspect((1, 1, 1), zoom=zoom); ax.view_init(elev=view[0], azim=view[1]); ax.set_axis_off()


def draw_context(ax, r, center, half, mesh=None):
    if mesh is not None:
        V, F = mesh; step = max(1, len(F) // 4000)
        ax.add_collection3d(Poly3DCollection(V[F[::step]], facecolors=C["mesh"], alpha=0.06, linewidths=0))
    for _, c, on in r["cl"]:
        ax.plot(c[:, 0], c[:, 1], c[:, 2], color=C["cl"], lw=0.7 if on else 0.4, alpha=0.8 if on else 0.35)
    P = r["path"]; ax.plot(P[:, 0], P[:, 1], P[:, 2], color=C["path"], lw=1.8)
    tg = np.asarray(r["target"], float)
    if np.isfinite(tg).all(): ax.scatter(*tg, color=C["target"], marker="X", s=70, edgecolors="k", linewidths=0.5, depthshade=False)
    # the zoom cube
    c, h = np.asarray(center), half; corners = np.array([[sx, sy, sz] for sx in (-h, h) for sy in (-h, h) for sz in (-h, h)]) + c
    edges = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]
    ax.add_collection3d(Line3DCollection([[corners[i], corners[j]] for i, j in edges], colors=C["other"], linewidths=1.4))
    lo, hi = tr._bbox(r); ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect((1, 1, 1)); ax.set_axis_off()


def draw_trace(ax, r, marks, color):
    T = r["T"]; x = np.arange(T); pl = float(r["path_len"]); prog = r["proj_s"] / pl
    for e in r["ev"]:
        a = max(int(e["first"]) - 1, 0); b = int(e["close"]) - 1 if e["close"] > 0 else T - 1
        ax.axvspan(a, b, color=C["stall"], alpha=0.7, lw=0)
    ax.plot(x, prog, color=C["prog"], lw=1.6)
    ax.set_ylim(-0.02, 1.06); ax.set_xlim(0, max(T - 1, 1)); ax.set_ylabel("progress along path", fontsize=10)
    ax.set_xlabel("step", fontsize=10); ax.tick_params(labelsize=9)
    ax2 = ax.twinx(); cs = np.asarray(r["cath_slack"], float); gs = np.asarray(r["ins_gw"], float) - np.asarray(r["proj_s"], float)
    if np.isfinite(cs).any(): ax2.plot(x, cs, color=C["cslack"], lw=1.0, ls="--", alpha=0.85, label="catheter slack")
    ax2.plot(x, gs, color=C["gslack"], lw=1.0, ls=":", alpha=0.9, label="guidewire slack")
    ax2.set_ylabel("slack (mm)", fontsize=10); ax2.tick_params(labelsize=9)
    ymax = max(float(np.nanmax(gs)), float(np.nanmax(cs)) if np.isfinite(cs).any() else 0.0, 30.0); ax2.set_ylim(min(-20, float(np.nanmin(gs)) - 5), ymax * 1.1)
    ax2.legend(fontsize=8, frameon=False, loc="upper left")
    for k, (t, lab) in enumerate(marks):
        ax.axvline(t, color=color, lw=1.4, alpha=0.9)
        ax.text(t, 0.10 + 0.09 * (k % 2), " " + lab, color=color, fontsize=8.5, ha="left" if t < 0.8 * T else "right", va="bottom",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))
    for s in ("top",): ax.spines[s].set_visible(False); ax2.spines[s].set_visible(False)


# ------------------------------------------------------------------ frame choice
def frames_fail(r):
    T = r["T"]; ev = r["ev"]
    if ev:
        e = ev[-1]; f = max(int(e["first"]) - 1, 0)
        row = r.get("row"); fl = row.get("coilr_first_loop_step") if row is not None else None
        if fl is not None and np.isfinite(fl) and bool(row.get("loop", False)) and f < int(fl) < T - 1:
            mid = (int(fl), "loop forms")
        else:
            lead = np.asarray(r["ins_cath"][f:T], float) - np.asarray(r["ins_gw"][f:T], float)
            pk = f + int(np.argmax(lead)) if np.isfinite(lead).any() else (f + T - 1) // 2
            mid = (pk, "catheter lead peaks") if pk not in (f, T - 1) else ((f + T - 1) // 2, "mid-stall")
        return [(f, "stall onset"), mid, (T - 1, "end")]
    t1 = int(np.nanargmax(r["proj_s"])); return [(max(t1 // 2, 0), "advancing"), (t1, "furthest point"), (T - 1, "end")]


def frames_pass(r, nook_s, half=45.0):
    """Frames for the side that passes. Entries are (step, label, centre-override or None).
    If its nearest stall is at the nook: onset / withdrawal / escape there. If it stalls only
    elsewhere: approach and pass the nook, then its own stall's escape at that stall's centre."""
    T = r["T"]; ev = [e for e in r["ev"] if e["close"] > 0]
    ps = np.asarray(r["proj_s"], float); pl = float(r["path_len"])
    if ev:
        e = min(ev, key=lambda e: abs(float(e["p0"]) - nook_s))
        ph = [p for p in tr.phases(r, r["ev"]) if abs(p["p0"] - float(e["p0"])) < 1e-6]
        if ph:
            p = ph[0]
            mid = (p["withdraw"], "withdrawn %.0f mm" % p["retract_mm"]) if p["retract_mm"] >= 1.0 else (p["peak"], "peak load")
            if abs(float(e["p0"]) - nook_s) <= half:
                return [(p["first"], "stall onset", None), (mid[0], mid[1], None), (min(p["escape"] + 2, T - 1), "escape", None)]
            own = point_at(r, float(e["p0"]))
            a = np.nonzero(ps >= nook_s - 20)[0]; b = np.nonzero(ps >= nook_s + 20)[0]
            ta = int(a[0]) if len(a) else 0; tb = int(b[0]) if len(b) else int(np.nanargmax(ps))
            return [(ta, "approach (no stall here)", None), (tb, "past the nook", None),
                    (min(p["escape"] + 2, T - 1), "its own stall at %.0f %%: escape" % (100 * float(e["p0"]) / pl), own)]
    a = np.nonzero(ps >= nook_s - 20)[0]; b = np.nonzero(ps >= nook_s + 20)[0]
    ta = int(a[0]) if len(a) else 0; tb = int(b[0]) if len(b) else int(np.nanargmax(ps))
    return [(ta, "approach", None), (tb, "past the nook", None), (T - 1, "end", None)]


def caption(r, t):
    pl = float(r["path_len"]); prog = 100 * float(r["proj_s"][t]) / pl
    fed = float(r["ins_gw"][t]); lead = float(r["ins_cath"][t] - r["ins_gw"][t]); cs = float(r["cath_slack"][t])
    s = "step %d  ·  progress %.0f %%\nwire fed %.0f mm\ncatheter %+.0f mm vs wire" % (t, prog, fed, lead)
    if np.isfinite(cs) and abs(cs) >= 10: s += "\ncatheter slack %.0f mm" % cs
    return s


# ------------------------------------------------------------------ slide furniture
def new_slide():
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100); fig.patch.set_facecolor(C["bg"]); return fig


def header(fig, title, claim=None, tag=None):
    fig.text(0.035, 0.945, title, fontsize=25 if len(title) <= 64 else 21, weight="bold", color=C["ink"], va="center")
    if claim: fig.text(0.035, 0.897, textwrap.fill(claim, 150), fontsize=14.5, color=C["muted"], va="center")
    if tag: fig.text(0.965, 0.945, tag, fontsize=11, color=C["muted"], va="center", ha="right",
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C["rule"]))
    fig.add_artist(plt.Line2D([0.035, 0.965], [0.868, 0.868], color=C["rule"], lw=1.2, transform=fig.transFigure))


def footer(fig, text):
    fig.text(0.035, 0.028, text, fontsize=10, color=C["muted"], va="bottom")


def column_label(fig, x0, x1, y, text, role):
    col = C[role]; bg = C[role + "_bg"]
    fig.patches.append(FancyBboxPatch((x0, y - 0.022), x1 - x0, 0.046, boxstyle="round,pad=0.004,rounding_size=0.008",
                                      fc=bg, ec=col, lw=1.2, transform=fig.transFigure, figure=fig))
    fig.text(x0 + 0.012, y, text, fontsize=13.5, weight="bold", color=col, va="center")


def pair_slide(out, title, claim, other, cb3, family, half=45, notes="", tag=None, footer_text=""):
    """other / cb3: dict(label=..., r=record, sub=...)."""
    rf, rp = other["r"], cb3["r"]
    center, nook_s = nook_of(rf); mesh = mesh_for(rf, family); view = view_for(rf, center, half)
    ff = [(t, lab, None) for t, lab in frames_fail(rf)]; fp = frames_pass(rp, nook_s, half)
    fig = new_slide(); header(fig, title, claim, tag)
    cols = [(0.035, 0.49, other, "other", ff, rf), (0.51, 0.965, cb3, "cb3", fp, rp)]
    for x0, x1, side, role, frames, r in cols:
        column_label(fig, x0, x1, 0.835, side["label"], role)
        fig.text(x0 + 0.012, 0.803, textwrap.fill(side["sub"], 105), fontsize=10.5, color=C["ink"], va="top", linespacing=1.3)
        w = (x1 - x0) / 3.0
        for k, (t, lab, cen) in enumerate(frames):
            wide = (role == "other" and k == len(frames) - 1)          # the failing side's last frame: wider, to show where the slack went
            h = half * 2 if wide else half
            c_k = center if cen is None else np.asarray(cen, float); v_k = view if cen is None else view_for(r, c_k, half)
            ax = fig.add_axes([x0 + k * w - 0.01, 0.445, w + 0.02, 0.32], projection="3d"); ax.set_facecolor(C["bg"])
            draw_nook(ax, r, t, c_k, h, mesh, v_k, zoom=1.9 if not wide else 1.7)
            fig.text(x0 + k * w + w / 2, 0.437, lab + (" (±%d mm)" % h if wide else ""), fontsize=11.5, weight="bold", color=C[role], ha="center", va="top")
            fig.text(x0 + k * w + w / 2, 0.414, caption(r, t), fontsize=8.3, color=C["muted"], ha="center", va="top", linespacing=1.25)
        axt = fig.add_axes([x0 + 0.045, 0.165, x1 - x0 - 0.09, 0.185]); axt.set_facecolor("white")
        draw_trace(axt, r, [(t, lab.split(":")[-1].strip() if ":" in lab else lab) for t, lab, _ in frames], C[role])
    # context: whole anatomy with the zoom cube
    axc = fig.add_axes([0.83, 0.005, 0.16, 0.13], projection="3d"); axc.set_facecolor(C["bg"]); draw_context(axc, rf, center, half, mesh)
    fig.text(0.91, 0.004, "zoom cube ±%d mm, on the planned path" % half, fontsize=8, color=C["muted"], ha="center")
    if notes: fig.text(0.035, 0.108, textwrap.fill(notes, 165), fontsize=11, color=C["ink"], va="top", linespacing=1.35)
    footer(fig, footer_text or "guidewire green, catheter red, planned path orange, target X, vessel wall grey (collision mesh), centerlines blue; shaded trace spans = stall events")
    fig.savefig(out, dpi=100, facecolor=fig.get_facecolor()); plt.close(fig); print("wrote", out)


def text_block(fig, x, y, lines, fontsize=13, color=None, dy=None):
    dy = dy or (fontsize * 1.75 / 1080)
    for i, l in enumerate(lines):
        fig.text(x, y - i * dy, l, fontsize=fontsize, color=color or C["ink"], va="top")


def table(fig, x0, y0, w, colw, rows, fontsize=12, header_color=None, row_h=None):
    row_h = row_h or (fontsize * 2.1 / 1080)
    for i, row in enumerate(rows):
        y = y0 - i * row_h; x = x0
        if i == 0:
            fig.patches.append(FancyBboxPatch((x0 - 0.004, y - row_h * 0.72), w + 0.008, row_h * 0.95, boxstyle="round,pad=0.002",
                                              fc="#e8e6e0", ec="none", transform=fig.transFigure, figure=fig))
        for j, cell in enumerate(row):
            fig.text(x, y - row_h * 0.25, str(cell), fontsize=fontsize, weight="bold" if i == 0 else "normal",
                     color=(header_color or C["ink"]) if i == 0 else C["ink"], va="center")
            x += colw[j]
        if i > 0:
            fig.add_artist(plt.Line2D([x0, x0 + w], [y - row_h * 0.72, y - row_h * 0.72], color=C["rule"], lw=0.6, transform=fig.transFigure))


def bars(ax, labels, a_vals, b_vals, a_name, b_name, unit="%"):
    y = np.arange(len(labels)); h = 0.36
    ax.barh(y + h / 2, a_vals, h, color=C["other"], alpha=0.9, label=a_name); ax.barh(y - h / 2, b_vals, h, color=C["cb3"], alpha=0.9, label=b_name)
    for yy, v in zip(y + h / 2, a_vals): ax.text(v + 1, yy, "%.0f%s" % (v, unit), va="center", fontsize=10, color=C["other"])
    for yy, v in zip(y - h / 2, b_vals): ax.text(v + 1, yy, "%.0f%s" % (v, unit), va="center", fontsize=10, color=C["cb3"])
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=11); ax.invert_yaxis(); ax.set_xlim(0, max(max(a_vals), max(b_vals)) * 1.25 + 5)
    ax.tick_params(axis="x", labelsize=9); ax.legend(fontsize=10, frameon=False, loc="lower right"); ax.set_facecolor("white")
    for s in ("top", "right"): ax.spines[s].set_visible(False)


def thumb(fig, rect, r, t, half, family, title, text, role):
    center = np.asarray(r["tip"][t], float)
    if not np.isfinite(center).all(): center = np.asarray(r["gw"][t][0], float)
    mesh = mesh_for(r, family); view = view_for(r, center, half)
    ax = fig.add_axes(rect, projection="3d"); ax.set_facecolor(C["bg"]); draw_nook(ax, r, t, center, half, mesh, view)
    x, y, w, h = rect
    fig.text(x + w / 2, y + h + 0.004, title, fontsize=13, weight="bold", color=C[role], ha="center", va="bottom")
    fig.text(x + w / 2, y - 0.006, textwrap.fill(text, 62), fontsize=9.2, color=C["ink"], ha="center", va="top", linespacing=1.25)


# ------------------------------------------------------------------ the deck
def build(out_dir, only, half):
    os.makedirs(out_dir, exist_ok=True)
    A1 = os.path.join(CAMP, "analysis_part1"); A2 = os.path.join(CAMP, "analysis_part2")
    A3c = os.path.join(CAMP, "analysis_part3", "carotid"); A3h = os.path.join(CAMP, "analysis_part3", "host"); A3p = os.path.join(CAMP, "analysis_part3", "procedural")
    e1, e2, e3c, e3h, e3p = (episodes(d) for d in (A1, A2, A3c, A3h, A3p))
    want = lambda n: (not only) or (n in only)
    CB3 = "CB3 = carotid v3 best, checkpoint 2 289 002 (2026-09-09 run)"

    # 01 title -----------------------------------------------------------------
    if want("01"):
        fig = new_slide()
        fig.text(0.06, 0.78, "What the carotid v3 best checkpoint does that the others do not", fontsize=34, weight="bold", va="center")
        fig.text(0.06, 0.70, "Recovery and coil avoidance at the frontier: CB3 against the actor-masked ablation, TopBrain v2 and the procedural policy",
                 fontsize=18, color=C["muted"], va="center")
        text_block(fig, 0.06, 0.58, [
            "Frontier replay campaign, 2026-09-20/21: 27 replay runs, 2 217 recorded and filmed episodes, hard seeds only, 1–4 repetitions per checkpoint.",
            "Same seed = same anatomy, same target, same planned path; only the device twist differs between repetitions.",
            "",
            "Arm A  (ablation)      CB3 vs CBA  = same recipe with the ten privileged dims masked from the actor, its best checkpoint (1.01 M)     → loops, recovery",
            "Arm B  (data)              CB3 vs TB2  = TopBrain v2 best checkpoint, on the real patient                                                            → coils, recovery",
            "Arm C  (data)              CB3 vs PROC = procedural v3c policy (AWAC, its best 62 % checkpoint), on carotid, patient and procedural anatomies → coils, catheter-led, recovery",
        ], fontsize=14.5)
        table(fig, 0.06, 0.30, 0.86, [0.30, 0.14, 0.14, 0.14, 0.14], [
            ["headline (success on the hard seeds of each arm)", "CB3", "other", "seed-paired", "other's loop rate"],
            ["A  carotid frontier set, 47 seeds", "97.3 %", "CBA 92.6 %", "6 – 2", "5.9 % (CB3 2.1 %)"],
            ["B  real patient, 85 hard seeds", "92.9 %", "TB2 56.5 %", "59 – 3", "54 % (CB3 0.4 %)"],
            ["C  carotid frontier set, 47 seeds", "97.3 %", "PROC 42.6 %", "27 – 0", "4 % (CB3 2.1 %)"],
            ["C  real patient, 98 seeds", "93.9 %", "PROC 46.9 %", "52 – 2", "4 % (CB3 0.4 %)"],
            ["C  procedural anatomies, 98 seeds", "58.2 %", "PROC 50.0 %", "8 – 0", "49 % (CB3 27 %)"],
        ], fontsize=13)
        footer(fig, CB3 + ".  Seed-paired = seeds where CB3 is better – seeds where the other is better, on per-seed success probability over the repetitions.")
        fig.savefig(os.path.join(out_dir, "01_title.png"), dpi=100, facecolor=fig.get_facecolor()); plt.close(fig); print("wrote 01")

    # 02 behaviours ---------------------------------------------------------------
    if want("02"):
        fig = new_slide(); header(fig, "Vocabulary: how an episode goes", "Every replay is labelled by its stall events (frontier stops advancing while the wire is pushed) and by what happens inside them")
        ex = [  # (record, step-picker, title, text, role)
            (record(e1, "car_best", 2, 154), lambda r: int(r["T"] * 0.6), "free run", "No stall: the wire leads and the frontier advances all the way. 79 % of CB3's successes on easy seeds, 40 % on frontier seeds.", "cb3"),
            (record(e1, "car_best", 1, 50), lambda r: int(r["T"] * 0.6), "catheter-led run", "The catheter runs 20–40 mm ahead and carries the wire; fast (22–37 steps). CB3 does this only on easy seeds; the procedural policy lives by it.", "cb3"),
            (record(e1, "car_best", 1, 154), lambda r: [p for p in tr.phases(r, r["ev"])][0]["withdraw"], "light recovery", "A stall, the insertion is withdrawn < 8 mm, the wire re-advances and escapes.", "cb3"),
            (record(e1, "car_best", 3, 134), lambda r: [p for p in tr.phases(r, r["ev"])][0]["withdraw"], "deep recovery", "A stall, > 8 mm withdrawn at the insertion (the tip retreats in ~2/3 of cases, else stored slack is released), re-advance, escape. Half of CB3's frontier successes.", "cb3"),
            (record(e2, "car_best_host", 3, 900029), lambda r: [p for p in tr.phases(r, r["ev"])][1]["escape"], "fight (3+ stalls escaped)", "Several stalls in a row, each escaped. 39 % of CB3's successes on the real patient.", "cb3"),
            (record(e1, "car_best", 4, 134), lambda r: r["T"] - 1, "doorstep stall", "Last stall still open at the end, within the last 10 % of the path, no loop. CB3's only failure mode on its own anatomies.", "other"),
            (record(e1, "abl_best", 1, 139), lambda r: r["T"] - 1, "shove-capped (coil)", "The catheter is driven > 80 mm ahead of a stalled wire; the slack coils. The ablation's and TopBrain v2's signature failure.", "other"),
            (record(e1, "abl_best", 3, 134), lambda r: r["T"] - 1, "coiled-stuck", "Catheter slack > 50 mm or wire slack > 100 mm (a loop three times out of four on the polylines) and no escape.", "other"),
            (record(e1, "abl_last", 1, 23), lambda r: r["T"] - 1, "stuck mid-path", "Last stall open in the middle third of the path. The decayed ablation's failure; the procedural policy's on foreign anatomies.", "other"),
        ]
        cw, ch = 0.27, 0.175; xs = [0.055, 0.365, 0.675]; ys = [0.625, 0.365, 0.105]
        for k, (r, pick, ttl, txt, role) in enumerate(ex):
            fam = "host" if r["seed"] >= 900000 else "carotid"
            thumb(fig, [xs[k % 3], ys[k // 3], cw, ch], r, int(pick(r)), 38, fam, ttl, txt, role)
        fig.text(0.035, 0.012, "Frames zoomed ±38 mm around the guidewire tip; guidewire green, catheter red, planned path orange, target X, vessel wall grey.", fontsize=9.5, color=C["muted"], va="bottom")
        fig.savefig(os.path.join(out_dir, "02_behaviours.png"), dpi=100, facecolor=fig.get_facecolor()); plt.close(fig); print("wrote 02")

    # 02a / 02b / 02c: the two labelling methods --------------------------------------
    from matplotlib.patches import FancyArrowPatch
    def box(fig, x, y, w, h, text, fc="white", ec=C["rule"], fs=13, title=None, tc=None, lw=1.2, align="center", pad=0.012):
        fig.patches.append(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.01", fc=fc, ec=ec, lw=lw,
                                          transform=fig.transFigure, figure=fig, zorder=-1))
        ty = y + h - pad
        if title:
            fig.text(x + w / 2 if align == "center" else x + pad, ty, title, fontsize=fs + 1, weight="bold", color=tc or C["ink"],
                     ha=align, va="top"); ty -= (fs + 1) * 2.0 / 1080
        if text:
            fig.text(x + w / 2 if align == "center" else x + pad, ty, text, fontsize=fs, color=C["ink"], ha=align, va="top", linespacing=1.35)

    def arrow(fig, x0, y0, x1, y1, color=None, lw=1.6):
        fig.patches.append(FancyArrowPatch((x0, y0), (x1, y1), transform=fig.transFigure, arrowstyle="-|>", mutation_scale=16,
                                           color=color or C["muted"], lw=lw, shrinkA=0, shrinkB=0, figure=fig))

    def pill(fig, x, y, text, role, fs=12, w=None):
        w = w or (len(text) * fs * 0.70 / 1920 + 0.022)
        fig.patches.append(FancyBboxPatch((x, y - 0.014), w, 0.028, boxstyle="round,pad=0.003,rounding_size=0.012",
                                          fc=C[role + "_bg"], ec=C[role], lw=1.0, transform=fig.transFigure, figure=fig))
        fig.text(x + w / 2, y, text, fontsize=fs, color=C[role], ha="center", va="center", weight="bold"); return w

    if want("02a"):
        fig = new_slide(); header(fig, "Method 1 · the atlas: 11 clusters found by k-means",
                                  "Unsupervised: describe every episode by its shape, let an algorithm group similar shapes, name the groups afterwards", tag="labelling")
        # pipeline of four boxes
        bx = [0.04, 0.29, 0.54, 0.77]; bw = [0.21, 0.21, 0.19, 0.19]; by, bh = 0.54, 0.27
        box(fig, bx[0], by, bw[0], bh, "explore + eval, all training runs\n(carotid, TopBrain, procedural,\nablations), 2026-07 → 09",
            title="~139 000 episodes", tc=C["cb3"], fs=12.5)
        box(fig, bx[1], by, bw[1], bh, "", title="78-number fingerprint", tc=C["cb3"], fs=12.5)
        box(fig, bx[2], by, bw[2], bh, "PCA → 15 components\nk-means → 4 broad groups\nsub-clusters 4 / 3 / 2 / 2",
            title="compress + cluster", tc=C["cb3"], fs=12.5)
        box(fig, bx[3], by, bw[3], bh, "each cluster named by looking\nat its typical member —\nthe name is a description,\nnot a rule",
            title="11 clusters", tc=C["cb3"], fs=12.5)
        for i in range(3): arrow(fig, bx[i] + bw[i] + 0.004, by + bh / 2, bx[i + 1] - 0.004, by + bh / 2)
        # fingerprint icons inside box 2: two tiny curves + a scalar list
        r0 = record(e1, "car_best", 3, 134); T = r0["T"]; xs = np.linspace(0, T - 1, 20).astype(int)
        for k, (arr, lab, col) in enumerate(((r0["proj_s"] / float(r0["path_len"]), "progress, 20 points", C["prog"]),
                                             (r0["ins_gw"] - r0["proj_s"], "slack, 20 points", C["gslack"]))):
            axi = fig.add_axes([bx[1] + 0.02, by + 0.165 - k * 0.08, 0.075, 0.058]); axi.set_facecolor("white"); axi.set_zorder(2)
            axi.plot(np.arange(T), arr, color="#cfd3d8", lw=1); axi.plot(xs, arr[xs], "o-", color=col, ms=3, lw=1.2)
            axi.set_xticks([]); axi.set_yticks([]); [axi.spines[s].set_visible(False) for s in ("top", "right")]
            fig.text(bx[1] + 0.105, by + 0.194 - k * 0.08, lab, fontsize=11.5, va="center", color=C["ink"])
        fig.text(bx[1] + 0.02, by + 0.038, "+ 38 scalars: episode length, stalls,\n   withdrawal, catheter lead, max slack …", fontsize=11.5, va="center", color=C["ink"])
        # the eleven clusters as pills
        fig.text(0.04, 0.42, "The 11 clusters", fontsize=14, weight="bold", color=C["ink"], va="center")
        x = 0.04
        for t in ("clean", "clean-fast", "catheter-led sprint", "light recovery", "deep recovery"):
            x += pill(fig, x, 0.37, t, "cb3") + 0.012
        x = 0.04
        for t in ("shove-fight", "near-buckle fight", "mid-path thrash", "mega-coil", "shove-cap", "proximal thrash"):
            x += pill(fig, x, 0.325, t, "other") + 0.012
        fig.text(0.04, 0.285, "successes in blue, fights and failures in red", fontsize=10.5, color=C["muted"], va="center")
        # what it is good for / what it is
        box(fig, 0.04, 0.10, 0.44, 0.15, "•  a first map of everything the policies do, across all runs, with no categories decided in advance\n"
            "•  a rough one-word summary of an episode's shape", title="Good for", tc=C["ok"], fs=12.5, align="left", fc="#eef6ee", ec="#cfe3cf")
        box(fig, 0.52, 0.10, 0.44, 0.15, "•  a cluster is 'whatever lies near this average shape': its edges fall where k-means puts them\n"
            "•  the fingerprint includes episode length, so the groups partly just separate success from failure", title="Keep in mind", tc=C["other"], fs=12.5, align="left", fc=C["other_bg"], ec="#e3c7c4")
        footer(fig, "monitoring/traj/traj_cluster.py · model saved in saved/traj/cluster_model.npz")
        fig.savefig(os.path.join(out_dir, "02a_method_atlas.png"), dpi=100, facecolor=fig.get_facecolor()); plt.close(fig); print("wrote 02a")

    if want("02b"):
        fig = new_slide(); header(fig, "Method 2 · phases: 12 labels set by explicit rules",
                                  "Rule-based: find the stalls first, then label the episode by what happened inside them", tag="labelling")
        # root
        box(fig, 0.36, 0.72, 0.28, 0.11, "the tip stops moving forward along the path\nwhile the policy keeps pushing the wire",
            title="1 · find the stalls", tc=C["cb3"], fs=12, fc="#eef3f7")
        # three branch boxes
        cols = [(0.04, "no stall at all"), (0.365, "stalls, all escaped"), (0.69, "last stall never escaped  →  failure")]
        for x, t in cols:
            box(fig, x, 0.555, 0.27, 0.065, "", title="2 · " + t, tc=C["ink"], fs=12, fc="#faf9f6")
            arrow(fig, 0.50, 0.72, x + 0.135, 0.622)
        # leaves
        def leaf(x, y, cond, name, role):
            fig.text(x, y, cond, fontsize=11.5, color=C["ink"], va="center", ha="left")
            pill(fig, x, y - 0.036, name, role, fs=12)
        leaf(0.05, 0.50, "catheter > 30 mm ahead of the wire", "catheter-led run", "cb3")
        leaf(0.05, 0.415, "wire hardly pushed at all", "idle creep", "cb3")
        leaf(0.05, 0.33, "otherwise", "free run", "cb3")
        leaf(0.375, 0.50, "3 or more stalls escaped", "fight (near / far)", "cb3")
        leaf(0.375, 0.415, "1–2 stalls, withdrawn ≤ 8 mm", "light recovery", "cb3")
        leaf(0.375, 0.33, "1–2 stalls, withdrawn > 8 mm", "deep recovery", "cb3")
        fig.text(0.70, 0.515, "checked in this order:", fontsize=11, color=C["muted"], va="center", style="italic")
        leaf(0.70, 0.48, "①  catheter > 80 mm ahead of the stalled wire", "shove-capped", "other")
        leaf(0.70, 0.41, "②  catheter slack > 50 mm or wire slack > 100 mm", "coiled-stuck", "other")
        leaf(0.70, 0.34, "③  stuck in the last 10 % of the path", "doorstep stall", "other")
        leaf(0.70, 0.27, "④  stuck in the first third  /  anywhere else", "stuck proximal  ·  stuck mid-path", "other")
        # step 3
        box(fig, 0.04, 0.08, 0.60, 0.13, "loop check on the device shape (does the wire or catheter cross itself?)   ·   catheter lead   ·   tip retreat\n"
            "where along the path the episode ended   ·   withdrawal depth and stall count", title="3 · measurements recorded next to the label", tc=C["cb3"], fs=12, align="left", fc="#eef3f7")
        box(fig, 0.68, 0.08, 0.28, 0.13, "no episode length anywhere in the rules;\nevery threshold is a physical quantity\nanyone can read off the trace", title="By design", tc=C["ok"], fs=12, align="left", fc="#eef6ee", ec="#cfe3cf")
        footer(fig, "monitoring/traj/secondpass/phase_taxonomy.py and indicators.py")
        fig.savefig(os.path.join(out_dir, "02b_method_phases.png"), dpi=100, facecolor=fig.get_facecolor()); plt.close(fig); print("wrote 02b")

    if want("02c"):
        fig = new_slide(); header(fig, "Why the phases (method 2) are used for every comparison",
                                  "The clusters describe shapes; the phases describe what the device did, with rules anyone can check", tag="labelling")
        # two column headers
        column_label(fig, 0.30, 0.62, 0.815, "Atlas clusters · method 1", "other")
        column_label(fig, 0.65, 0.97, 0.815, "Phases · method 2", "cb3")
        rows = [("episode length", "in the fingerprint — failures all last 600 steps,\nso groups partly just split success from failure", "never used"),
                ("boundaries", "wherever k-means puts them", "physical thresholds: 8 mm withdrawal,\n80 mm catheter lead, last 10 % of the path"),
                ("catheter-led", "'sprint' cluster also swallowed fast clean\nruns with only 8–29 mm of lead", "an explicit lead measurement (> 30 mm)"),
                ("coils", "inferred from the shape of the slack curve", "slack thresholds + a loop check\non the device shape"),
                ("stability", "same seed, model and outcome:\nlabel changes on ~40 % of replays", "fixed rules — changes only\nwhen the behaviour changes"),
                ("one episode", "needs the whole fitted model", "read it off the trace")]
        y = 0.755
        for name, a, b in rows:
            fig.text(0.04, y, name, fontsize=13.5, weight="bold", color=C["ink"], va="center")
            fig.text(0.31, y, "✗", fontsize=15, color=C["other"], va="center", weight="bold"); fig.text(0.33, y, a, fontsize=12, color=C["ink"], va="center", linespacing=1.3)
            fig.text(0.66, y, "✓", fontsize=15, color=C["ok"], va="center", weight="bold"); fig.text(0.68, y, b, fontsize=12, color=C["ink"], va="center", linespacing=1.3)
            y -= 0.085
            fig.add_artist(plt.Line2D([0.04, 0.965], [y + 0.042, y + 0.042], color=C["rule"], lw=0.7, transform=fig.transFigure))
        # blind check tiles
        fig.text(0.04, 0.215, "Blind check of the phases · 26 episodes judged from pictures only, labels hidden", fontsize=13.5, weight="bold", color=C["ink"], va="center")
        for k, (big, lab) in enumerate((("26 / 26", "outcome"), ("20 / 20", "end location"), ("23 / 26", "loops (the 3 misses were tiny tip hooks)"))):
            x = 0.04 + k * 0.31
            box(fig, x, 0.06, 0.29, 0.12, "", fc="#eef3f7", ec="#cfdbe5")
            fig.text(x + 0.02, 0.125, big, fontsize=26, weight="bold", color=C["cb3"], va="center")
            fig.text(x + 0.13, 0.125, lab, fontsize=12, color=C["ink"], va="center", wrap=True)
        footer(fig, "Limits: a stall is only detected while the policy is pushing; a success ending in the very step the tip breaks free is relabelled as a recovery.  The atlas type is kept as a shape summary; every number in this deck uses the phases.")
        fig.savefig(os.path.join(out_dir, "02c_why_phases.png"), dpi=100, facecolor=fig.get_facecolor()); plt.close(fig); print("wrote 02c")

    # 03 trace + seed classes --------------------------------------------------------
    if want("03"):
        fig = new_slide(); header(fig, "Reading a trace, and which seeds measure the frontier", "Recovery is read off progress and slack; the seed classes come from repeated replays of the strong checkpoints")
        r = record(e1, "car_best", 3, 134); ph = tr.phases(r, r["ev"])[0]
        ax = fig.add_axes([0.06, 0.50, 0.55, 0.32]); ax.set_facecolor("white")
        draw_trace(ax, r, [(ph["first"], "stall onset"), (ph["peak"], "peak load"), (ph["withdraw"], "withdrawn %.0f mm" % ph["retract_mm"]), (ph["escape"], "escape")], C["cb3"])
        ax.set_title("CB3, seed 134, replay 3: a deep recovery (success in %d steps)" % r["T"], fontsize=12, loc="left")
        text_block(fig, 0.06, 0.44, textwrap.wrap(
            "Stall = the frontier tip stops advancing while the wire is commanded forward (shaded).  Load = the insertion keeps growing and slack builds.  "
            "Recovery = insertion withdrawn (light < 8 mm, deep > 8 mm), then re-advance and the frontier moves again.  Loop = catheter slack > 50 mm or wire slack > 100 mm, "
            "or a polyline self-crossing.  Catheter-led = catheter tip > 20 mm ahead of the wire while progress accrues.  Withdrawal is measured at the insertion; "
            "the tip itself retreats > 8 mm in about two thirds of CB3's deep recoveries.", 105), fontsize=11.5)
        text_block(fig, 0.66, 0.82, ["Seed classes (per arm, from the strong checkpoints' repetitions)", "",
                                     "easy      every checkpoint solves it every time",
                                     "strength  all strong checkpoints always solve it, a weak one fails it",
                                     "frontier  a strong checkpoint fails it in some repetition",
                                     "floor     no strong checkpoint ever solves it"], fontsize=12)
        axb = fig.add_axes([0.68, 0.36, 0.28, 0.30]); axb.set_facecolor("white")
        fams = ["carotid 47\n(arm A)", "patient 85\n(arm B)", "carotid 47\n(arm C)", "patient 98\n(arm C)", "procedural 98\n(arm C)"]
        easy = [14, 18, 20, 44, 49]; strength = [23, 57, 24, 39, 0]; frontier = [10, 23, 3, 15, 8]; floor = [0, 0, 0, 0, 41]
        x = np.arange(len(fams)); b = np.zeros(len(fams))
        for vals, col, lab in ((easy, "#c9d6df", "easy"), (strength, "#8fb3c9", "strength"), (frontier, C["cb3"], "frontier"), (floor, "#444", "floor")):
            axb.bar(x, vals, 0.62, bottom=b, color=col, label=lab); b += np.asarray(vals)
        axb.set_xticks(x); axb.set_xticklabels(fams, fontsize=9); axb.legend(fontsize=9, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.14))
        axb.tick_params(axis="y", labelsize=9); [axb.spines[s].set_visible(False) for s in ("top", "right")]
        text_block(fig, 0.66, 0.30, ["Arm A frontier seeds (10):  23, 31, 55, 93, 134, 167, 8, 16, 139, 154", "On them, half of every strong checkpoint's successes are recoveries", "(14 % on the easy seeds)."], fontsize=11.5)
        footer(fig, "Decisive seeds = frontier + strength + floor. Every decisive seed's replays and films: analysis_part*/decisive.html.")
        fig.savefig(os.path.join(out_dir, "03_trace_and_seeds.png"), dpi=100, facecolor=fig.get_facecolor()); plt.close(fig); print("wrote 03")

    # 04 / 05 arm A pair slides -----------------------------------------------------
    if want("04"):
        pair_slide(os.path.join(out_dir, "04_armA_seed134.png"), "Arm A · CB3 vs Ablation · seed 134",
                   "Same nook at 97 % of the path: the ablation keeps feeding and the catheter is shoved over a coiling wire; CB3 unloads and re-advances",
                   dict(label="CBA — ablation best (1.01 M, actor mask), replay 1: FAIL, shove-capped coil", r=record(e1, "abl_best", 1, 134),
                        sub="anatomy casew007 left topcow mr013 L · 600 steps · max progress 97 % · 348 mm withdrawn in total but the tip only retreats after the coil · catheter slack peaks at 294 mm · per-seed success 2/4"),
                   dict(label="CB3 — carotid best (2.29 M), replay 3: SUCCESS by Recovery", r=record(e1, "car_best", 3, 134),
                        sub="123 steps · one stall · 61 mm withdrawn, tip retreats 32 mm, re-advance, escape · catheter slack stays < 27 mm · per-seed success 3/4"),
                   "carotid", half=half, tag="arm A · loops / recovery",
                   notes="Both models reach the same bend and stall there. The ablation's answer is to push: the catheter runs 112 mm ahead of the wire and the slack behind it coils (loop held 174 steps). CB3's answer is to unload 61 mm and re-advance; the frontier clears the bend on the second try. Over the 47 hard seeds the ablation loops in 5.9 % of episodes and is stuck in half of them; CB3 loops in 2.1 % and escaped every one.")
    if want("05"):
        pair_slide(os.path.join(out_dir, "05_armA_seed167.png"), "Arm A · CB3 vs Ablation · seed 167",
                   "No coil this time: the ablation parks at the doorstep and pushes and pulls 200 mm without ever escaping; CB3 converts the same stall in 105 steps",
                   dict(label="CBA — ablation best, replay 1: FAIL, doorstep stall (no loop)", r=record(e1, "abl_best", 1, 167),
                        sub="anatomy casem030 right topcow mr013 L · 600 steps · max progress 100 % but never registers · 202 mm withdrawn in total, tip retreats 4 mm · catheter slack < 19 mm · per-seed success 1/4"),
                   dict(label="CB3 — carotid best, replay 1: SUCCESS by Recovery", r=record(e1, "car_best", 1, 167),
                        sub="105 steps · one stall · 48 mm withdrawn, tip retreats 8 mm, escape · per-seed success 4/4 (three deep recoveries and one free run)"),
                   "carotid", half=half, tag="arm A · recovery without a loop",
                   notes="The ablation's 14 failures on the hard seeds split 5 coils and 9 stalls like this one (seeds 16, 31, 167), all of which CB3 solves 4/4. Both models face the same doorstep; the difference is whether the withdrawal is large enough and followed by a clean re-advance. Loops explain part of the gap, the unconverted recoveries the rest.")

    # 06 arm A summary ------------------------------------------------------------------
    if want("06"):
        fig = new_slide(); header(fig, "Arm A in numbers: CB3 vs Ablation (47 hard carotid seeds, 4 replays each)",
                                  "The ablation is 5 points behind on the same seeds; its failures are coils and unconverted stalls", tag="arm A")
        table(fig, 0.05, 0.80, 0.50, [0.28, 0.11, 0.11], [
            ["", "CB3 2.29 M", "CBA best 1.01 M"],
            ["success (47 hard seeds)", "97.3 %", "92.6 %"],
            ["loop rate, all episodes", "2.1 %", "5.9 %"],
            ["loops escaped", "100 %", "55 %"],
            ["failures with a loop", "0 %", "36 %"],
            ["successes by recovery, frontier seeds", "51 %", "62 %"],
            ["failures ending before ⅔ of the path", "0 %", "0 %"],
            ["seed-paired vs CB3 (better – worse)", "—", "2 – 6  (p = 0.29)"],
        ], fontsize=12.5)
        axb = fig.add_axes([0.66, 0.50, 0.30, 0.32]); bars(axb, ["loop rate", "failures\nwith a loop", "loops\nescaped", "recovery share,\nfrontier seeds"], [5.9, 36, 55, 62], [2.1, 0, 100, 51], "CBA best", "CB3")
        text_block(fig, 0.05, 0.36, [
            "•  The ablation's 14 failures: 5 coils after a doorstep stall (seeds 55, 134 ×2, 139, 154; catheter driven 70–650 mm ahead, loop held 140–460 steps)",
            "    and 9 plain doorstep stalls with 7–19 mm of catheter slack (seeds 16 ×2, 31 ×4, 167 ×3). CB3 solves those three seeds 4/4 each.",
            "•  CB3's 5 failures are all doorstep stalls at 90–94 % of the path (seeds 23, 55 ×3, 134): recoveries that did not convert, never a coil.",
            "•  Seed 55 goes the other way (ablation 3/4, CB3 1/4). Over all 47 seeds: CB3 better on 6, worse on 2.",
            "•  Coil avoidance appears in the baseline by 1.55 M steps (loop rate 23 % → 2 %); the ablation never gets below 6 %.",
        ], fontsize=12)
        footer(fig, "Films: analysis_part1/compare_abl_best_vs_car_best.html (every replay of the nine seeds that separate the two models).")
        fig.savefig(os.path.join(out_dir, "06_armA_summary.png"), dpi=100, facecolor=fig.get_facecolor()); plt.close(fig); print("wrote 06")

    # 07 / 08 arm B pair slides ---------------------------------------------------------
    if want("07"):
        pair_slide(os.path.join(out_dir, "07_armB_seed900029.png"), "Arm B · CB3 vs TopBrain v2 · patient seed 900029",
                   "TopBrain v2 fails this seed 3/3 by coiling at 60 % of the path; CB3 solves it 3/3 by fighting through three stalls at the same place",
                   dict(label="TB2 — TopBrain v2 best, replay 2: FAIL, coiled-stuck", r=record(e2, "tbv2_best_host", 2, 900029),
                        sub="600 steps · max progress 60 % · catheter slack peaks at 105 mm · 242 mm withdrawn, tip retreats 71 mm, no escape · per-seed success 0/3"),
                   dict(label="CB3 — carotid best, replay 3: SUCCESS by Recovery", r=record(e2, "car_best_host", 3, 900029),
                        sub="171 steps · three stalls escaped in turn · 151 mm withdrawn in total, tip retreats 13 mm · catheter slack < 12 mm · per-seed success 3/3"),
                   "host", half=half, tag="arm B · coils",
                   notes="TopBrain v2 forms a loop in 54 % of its episodes on the hard patient seeds and 69 % of its failures are coils (51 shove-capped, 22 coiled-stuck of 111). CB3 looped once in 268 patient episodes and escaped it; it wins 77 % of its patient successes by recovery or fight, at the price of longer episodes (median 186 steps vs 73 for the ablation and TopBrain v3).")
    if want("08"):
        pair_slide(os.path.join(out_dir, "08_armB_seed900030.png"), "Arm B · CB3 vs TopBrain v2 · patient seed 900030",
                   "The same stall answered two ways: TopBrain v2 shoves the catheter 225 mm of slack ahead and coils; CB3 unloads and re-advances",
                   dict(label="TB2 — TopBrain v2 best, replay 1: FAIL, shove-capped coil", r=record(e2, "tbv2_best_host", 1, 900030),
                        sub="600 steps · max progress 60 % · catheter slack peaks at 225 mm · 226 mm withdrawn, tip retreats 141 mm · per-seed success 2/3"),
                   dict(label="CB3 — carotid best, replay 3: SUCCESS by Recovery", r=record(e2, "car_best_host", 3, 900030),
                        sub="236 steps · three stalls · 237 mm withdrawn in total, tip retreats 70 mm on the decisive one · catheter slack < 11 mm · per-seed success 3/3"),
                   "host", half=half, tag="arm B · recovery",
                   notes="On the 85 hard patient seeds CB3 is better on 59 and worse on 3 (56.5 % vs 92.9 %). TopBrain v3 (257 k) and the masked ablation also beat TopBrain v2 there and score like CB3, but every one of their few failures is a loop; CB3's 18 patient failures contain none.")

    # 09 arm B summary --------------------------------------------------------------------
    if want("09"):
        fig = new_slide(); header(fig, "Arm B in numbers: the real patient, 85 seeds TopBrain v2 finds hard",
                                  "CB3 reproduces the ~95 % on all 98 seeds and is the only model on the patient with no looping failure", tag="arm B")
        table(fig, 0.05, 0.80, 0.52, [0.28, 0.12, 0.12], [
            ["", "CB3", "TB2"],
            ["replays", "3", "3"],
            ["success, 85 hard seeds", "92.9 %", "56.5 %"],
            ["success, all 98 seeds", "95.9 %", "~72 % (in-run)"],
            ["loop rate, all episodes", "0.4 %", "54 %"],
            ["failures with a loop", "0 %", "69 %"],
            ["successes by recovery or fight", "77 %", "23 %"],
            ["median steps", "186", "389"],
            ["seed-paired vs TB2 (better – worse)", "59 – 3", "—"],
        ], fontsize=12.5)
        axb = fig.add_axes([0.70, 0.50, 0.27, 0.32]); bars(axb, ["loop rate", "failures\nwith a loop", "failures before\n⅔ of the path", "recovery or fight\nshare of successes"], [54, 69, 31, 23], [0.4, 0, 0, 77], "TB2", "CB3")
        text_block(fig, 0.05, 0.36, [
            "•  TopBrain v2's 111 failures: 51 shove-capped, 22 coiled-stuck, 33 stuck mid-path; loops form by step 64 (median) and are escaped 44 % of the time.",
            "•  CB3's 18 failures: 7 doorstep stalls, 9 mid-path stalls past ⅔ of the path, 1 shove — no loop. It wins 4 % of patient seeds as free runs, 38 % by recovery, 39 % by fight.",
            "•  TopBrain v2's three replays scored 41, 60 and 43 of 85 on the same seeds: a run-level variance twist noise does not explain (CB3 moved 77 → 79). All three are pooled.",
        ], fontsize=12)
        footer(fig, "Films: analysis_part2/compare_tbv2_best_host_vs_car_best_host.html; per-seed table analysis_part2/seed_table.csv.")
        fig.savefig(os.path.join(out_dir, "09_armB_summary.png"), dpi=100, facecolor=fig.get_facecolor()); plt.close(fig); print("wrote 09")

    # 10 / 11 arm C pair slides -----------------------------------------------------------
    if want("10"):
        pair_slide(os.path.join(out_dir, "10_armC_seed8.png"), "Arm C · CB3 vs procedural policy · carotid seed 8",
                   "The procedural policy stalls at 28 % of the path with the catheter 42 mm ahead and never recovers; CB3 passes that spot in a free run and wins the seed by deep recovery at the doorstep",
                   dict(label="PROC — procedural v3c (AWAC), replay 1: FAIL, stuck proximal", r=record(e3c, "proc_on_carotid", 1, 8),
                        sub="anatomy casew013 right topcow mr013 L · 600 steps · max progress 28 % · catheter runs up to 42 mm ahead of the wire · no recovery · 27 of 47 seeds lost this way (median max progress 20 %)"),
                   dict(label="CB3 — carotid best, replay 1: SUCCESS by Recovery", r=record(e1, "car_best", 1, 8),
                        sub="passes the procedural policy's nook without a stall, then one deep recovery near the target · per-seed success 4/4"),
                   "carotid", half=half, tag="arm C · catheter-led vs recovery",
                   notes="On the 47 carotid frontier seeds the procedural policy scores 42.6 % against CB3's 97.3 %, better on none (27 – 0). Every one of its 20 successes is catheter-led (catheter 55 mm ahead at the end, median 50 steps) and only 5 % involve a recovery. CB3 uses the catheter-led sprint on 21 % of its successes here, all on easy seeds; on the frontier seeds it wins by recovery.")
    if want("11"):
        pair_slide(os.path.join(out_dir, "11_armC_seed900085.png"), "Arm C · CB3 vs procedural policy · patient seed 900085",
                   "Mid-path, the procedural policy's catheter runs 66 mm ahead and the pair stalls for good; CB3 stalls at the same place and recovers",
                   dict(label="PROC — procedural v3c, replay 1: FAIL, stuck mid-path", r=record(e3h, "proc_on_host", 1, 900085),
                        sub="600 steps · max progress 55 % · catheter up to 66 mm ahead of the wire · 30 of its 52 patient failures end like this (median max progress 57 %)"),
                   dict(label="CB3 — carotid best, replay 2: SUCCESS by Recovery", r=record(e2, "car_best_host", 2, 900085),
                        sub="137 steps · 110 mm withdrawn in total · per-seed success 3/3 (light, deep, deep recovery)"),
                   "host", half=half, tag="arm C · catheter-led vs recovery",
                   notes="On the patient the procedural policy scores 46.9 % (CB3 93.9 %, 52 – 2), with 52 % of its successes catheter-led and 13 % recoveries. On its own procedural anatomies it scores 50.0 % and CB3 58.2 % with a superset of its wins (8 – 0); there the procedural policy fails 35 of 49 times by coiling.")

    # 12 arm C summary + closing --------------------------------------------------------------
    if want("12"):
        fig = new_slide(); header(fig, "Arm C in numbers, and what separates CB3 across all three arms",
                                  "Catheter-led advance is the procedural policy's trait, not a frontier trait; recovery and coil avoidance are", tag="arm C · closing")
        table(fig, 0.05, 0.80, 0.90, [0.22, 0.09, 0.09, 0.09, 0.16, 0.15, 0.12], [
            ["anatomy set", "CB3", "PROC", "paired", "PROC: cath-led successes", "PROC: recovery share", "PROC: loop rate"],
            ["carotid frontier seeds (47)", "97.3 %", "42.6 %", "27 – 0", "100 %", "5 %", "4 %"],
            ["real patient (98)", "93.9 %", "46.9 %", "52 – 2", "52 %", "13 %", "2 %"],
            ["procedural anatomies (98)", "58.2 %", "50.0 %", "8 – 0", "92 %", "20 %", "49 %"],
        ], fontsize=12.5)
        text_block(fig, 0.05, 0.55, [
            "1.  The last four points of carotid validation are recovery.  Ten seeds decide 93 % vs 97 %; on them every strong checkpoint wins half or more of its",
            "     successes by unload-and-repush, and CB3's remaining failures are recoveries that did not convert at the last bend. Nothing there is won by a sprint.",
            "",
            "2.  Coil avoidance is learned by 1.55 M steps and is what the actor-masked ablation lacks.  CB3 stops feeding before a loop can form (marginal loops, undone",
            "     in 1–3 steps); the ablation answers the same doorstep stall by shoving and coiling. On the patient the same property makes CB3 the only model with no",
            "     looping failure, where TopBrain v2 loses two thirds of its failures to coils.",
            "",
            "3.  Catheter-led advance does not separate the frontier models.  CB3 shows it only on easy seeds late in training; the procedural policy that lives by it",
            "     fails 57 % of the carotid frontier seeds and 53 % of the patient seeds with no recovery to fall back on, and on its own anatomies CB3's wins are a superset of its own.",
        ], fontsize=12.5)
        footer(fig, "Full report: frontier_campaign/REPORT.md.  Recommended carotid frontier evaluation: replay the 33 decisive seeds four times instead of the 98 once.")
        fig.savefig(os.path.join(out_dir, "12_armC_and_closing.png"), dpi=100, facecolor=fig.get_facecolor()); plt.close(fig); print("wrote 12")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=os.path.join(CAMP, "slides"))
    ap.add_argument("--only", default=""); ap.add_argument("--half", type=float, default=45.0)
    a = ap.parse_args(); build(a.out, [s for s in a.only.split(",") if s], a.half)


if __name__ == "__main__":
    main()
