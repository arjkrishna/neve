"""traj_render.py -- key frames, trace and movie for recorded episodes.

Input: per-episode ``.npz`` records written by ``training _scripts/util/traj_record.py``
(one directory per validation block / replay session). Output, per
monitoring/traj/ANIM_PIPELINE.md section 4:

    <out>/<block>/<type>/s<seed>_<mesh>_<T>st_<success|fail>/
        key/k0_start_s036.png  k1_stall_e1_s059.png  k2_load_e1_s102.png
            k3_withdraw_e1_s106.png  k4_escape_e1_s121.png  k9_end_s142.png
        trace.png        progress / slack trace with stall windows
        episode.gif      the sequence (5-step cadence, 2-step inside stalls)
        events.json      detector events, phase steps, frame -> step map, type
    <out>/<block>/index.csv
    <out>/by_seed/s<seed>.html      the same seed across blocks

Typing reuses the atlas: traj_extract.core_features -> nearest centre in
saved/traj/cluster_model.npz. Frame schedule from the measured recovery phase
timing (ANIM_PIPELINE section 2).

usage:
  python monitoring/traj/traj_render.py <records_dir> <out_dir>
         [--types recovery|failures|all|3,4,6] [--cadence 5] [--stall-cadence 2]
         [--movie gif|none] [--dump-frames] [--jobs 4] [--limit N] [--seeds 44,95]
"""
import os, sys, io, json, glob, argparse, csv
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from traj_extract import core_features, detect, CANON  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402

REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
MODEL = os.path.join(REPO, "saved", "traj", "cluster_model.npz")
NAMES = os.path.join(REPO, "saved", "traj", "cluster_names.json")
TYPE_FOLDER = {0: "0_clean", 1: "1_clean_fast", 2: "2_cathled", 3: "3_light_recovery", 4: "4_deep_recovery",
               5: "5_shove_fight", 6: "6_near_fight", 7: "7_midpath_thrash", 8: "8_coil", 9: "9_shove_cap",
               10: "10_prox_thrash"}
GROUPS = {"recovery": {3, 4}, "failures": {5, 6, 7, 8, 9, 10}, "all": set(TYPE_FOLDER)}
COL = dict(wire="#00b050", cath="#cc2222", path="#ffaa00", target="#ffd400", cl_on="#1f77b4", cl_off="#bdbdbd",
           stall="#f3d9a4", prog="#1c2229", slack="#3b6ea5", cslack="#cc2222")


# ------------------------------------------------------------------ frames
def _rot_matrix(image_rot_zx):
    """numpy port of eve.util.coordtransform._get_rot_matrix (eve is not on the host)."""
    rz = -float(image_rot_zx[0]) * np.pi / 180; rx = -float(image_rot_zx[1]) * np.pi / 180
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    return Rz @ Rx


def tracking3d_to_vessel_cs(arr, image_rot_zx, image_center):
    """The STEP log's tip3d is in the tracking frame; records are in vessel CS."""
    R = _rot_matrix(image_rot_zx); c = R @ np.asarray(image_center, dtype=np.float64)
    a = np.asarray(arr, dtype=np.float64)
    return ((a + c) @ R).astype(np.float32)          # == (R.T @ (a + c).T).T


# ------------------------------------------------------------------ records
def load_record(path):
    z = np.load(path, allow_pickle=False)
    r = {k: z[k] for k in z.files}
    r["gw"] = r["gw"].astype(np.float32); r["cath"] = r["cath"].astype(np.float32)
    r["T"] = int(len(r["proj_s"]))
    r["seed"] = int(r["meta"][0]); r["pid"] = int(r["meta"][1]); r["ep"] = int(r["meta"][2])
    r["mesh_fp"] = str(r["mesh_fp"]); r["reason"] = str(r["reason"])
    r["success"] = bool(r["term"][-1] and not r["trunc"][-1]) or bool(r["success"].any())
    r["cl"] = [(str(n), r["cl_coords"][a:b], bool(o)) for n, a, b, o in
               zip(r["cl_names"], r["cl_offsets"][:-1], r["cl_offsets"][1:], r["cl_onpath"])]
    r["file"] = path
    return r


def load_log_episode(row, series, idx, scenery=None):
    """Tier 1: build a record from a log-derived episode (traj_extract output).
    The logs carry the guidewire TIP only -- no device bodies, no planned path,
    no centerlines -- so the 3-D panel shows the tip track; `scenery` (a dict
    mesh_fp -> (cl, path, target) harvested from tier-2 records of the same
    anatomy) restores the vessel drawing where available."""
    off = series["offsets"]; a, b = int(off[idx]), int(off[idx + 1]); T = b - a
    nan3 = np.full((T, 64, 3), np.nan, np.float32)
    tip = series["tip"][a:b].astype(np.float32) / 10.0
    tip[tip <= -3000] = np.nan
    r = dict(T=T, gw=nan3, cath=nan3.copy(), tip=tip,
             ins_gw=series["gw"][a:b].astype(np.float32) / 10.0, ins_cath=series["cath"][a:b].astype(np.float32) / 10.0,
             proj_s=series["proj"][a:b].astype(np.float32) / 10.0, fold=series["fold"][a:b].astype(np.int16),
             cmd=np.column_stack([series["cmd0"][a:b].astype(np.float32) / 100.0] + [np.full(T, np.nan, np.float32)] * 3),
             cath_slack=np.full(T, np.nan, np.float32), on_path=series["onpath"][a:b], phys=series["phys"][a:b],
             xt=series["xt"][a:b].astype(np.float32) / 100.0, d_tgt=series["dtgt"][a:b].astype(np.float32) / 10.0,
             reward=series["rew"][a:b].astype(np.float32) / 1000.0,
             term=np.zeros(T, bool), trunc=np.zeros(T, bool), success=np.zeros(T, bool),
             path=np.zeros((0, 3), np.float32), path_len=np.float32(row.get("pl") or np.nan),
             target=np.full(3, np.nan, np.float32), cl=[], seed=int(row.get("seed") or -1) if row.get("seed") is not None else -1,
             pid=int(row.get("pid") or 0), ep=int(row.get("ep") or 0), mesh_fp=str(row.get("mesh_fp") or ""),
             reason=str(row.get("reason") or ""), file="%s|%s|%s" % (row.get("tag"), row.get("pid"), row.get("ep")))
    r["success"] = bool(row.get("success"))
    r["term"][-1] = r["success"]; r["trunc"][-1] = not r["success"]
    csm = row.get("cath_slack_max")
    r["cath_slack_max_hint"] = float(csm) if csm is not None and csm == csm else 0.0
    r["frame_ok"] = False
    if scenery and r["mesh_fp"] in scenery:
        cl, path, target, frame = scenery[r["mesh_fp"]]
        r["cl"] = cl
        if frame is not None:   # map the log's tracking-frame tip into vessel CS
            ok = np.isfinite(r["tip"]).all(1)
            if ok.any(): r["tip"][ok] = tracking3d_to_vessel_cs(r["tip"][ok], frame[0], frame[1])
            r["frame_ok"] = True
        if row.get("target") and path is not None and target is not None:
            try:  # same target -> same planned path (the logged target is in the tracking frame too)
                tg = np.array([float(x) for x in str(row["target"]).strip("()").split(",")])
                if frame is not None: tg = tracking3d_to_vessel_cs(tg[None], frame[0], frame[1])[0]
                if np.linalg.norm(tg - target) < 3.0:
                    r["path"], r["target"] = path, target.astype(np.float32)
            except ValueError:
                pass
    return r


def build_scenery(records_dir, default_frame=None):
    """mesh_fp -> (centerlines, planned path, target, frame) from tier-2 records;
    the first record of each anatomy wins for centerlines, paths are per target.
    `frame` = (image_rot_zx, image_center) stored by the recorder, else
    `default_frame` (the carotid / TopBrain family uses (20, 5) / (0, 0, 0))."""
    out = {}
    for root, _, fs in os.walk(records_dir):
        for fn in fs:
            if not fn.endswith(".npz"): continue
            try:
                r = load_record(os.path.join(root, fn))
            except Exception:
                continue
            if r["mesh_fp"] and r["mesh_fp"] not in out and r["cl"]:
                frame = (r["image_rot_zx"], r["image_center"]) if "image_rot_zx" in r else default_frame
                out[r["mesh_fp"]] = (r["cl"], r["path"] if len(r["path"]) else None,
                                     r["target"] if np.isfinite(r["target"]).all() else None, frame)
    return out


def type_record(r):
    """Atlas type via the saved cluster model. Returns (cluster_id, name, features, events)."""
    pl = float(r["path_len"]) if np.isfinite(r["path_len"]) and r["path_len"] > 0 else None
    proj, gw, cath = r["proj_s"].astype(float), r["ins_gw"].astype(float), r["ins_cath"].astype(float)
    cmd0 = np.nan_to_num(r["cmd"][:, 0].astype(float)); fold = r["fold"].astype(int)
    f, ev = core_features(proj, gw, cath, cmd0, fold, pl)
    cs = r["cath_slack"].astype(float)
    f["cath_slack_max"] = float(np.nanmax(cs)) if np.isfinite(cs).any() else float(r.get("cath_slack_max_hint") or 0.0)
    f["steps"] = r["T"]
    try:
        import pandas as pd
        from traj_cluster import feature_matrix
        M = np.load(MODEL, allow_pickle=True)
        X, _ = feature_matrix(pd.DataFrame([f]))
        p = ((X - M["mu"]) / M["sd"]) @ M["comps"].T
        j = int(np.argmin(((M["centers"] - p) ** 2).sum(1))); cid = int(M["remap"][j])
        names = {int(k): v for k, v in json.load(open(NAMES)).items()}
        return cid, names[cid], f, ev
    except Exception as e:  # model missing: still render, untyped
        sys.stderr.write("[traj_render] typing failed (%s); rendering untyped\n" % e)
        return -1, "untyped", f, ev


# ------------------------------------------------------------------ schedule
def phases(r, ev):
    """Per closed event: first / onset / peak-load / withdrawal / escape steps (0-based)."""
    gw = r["ins_gw"].astype(float); out = []
    for i, e in enumerate(ev):
        if e["close"] <= 0: continue
        first, onset, close = e["first"] - 1, e["onset"] - 1, e["close"] - 1
        first = max(first, 0)
        pk = first + int(np.argmax(gw[first:close + 1]))
        mn = pk + int(np.argmin(gw[pk:close + 1]))
        out.append(dict(i=i + 1, kind=e["k"], first=first, onset=onset, peak=pk, withdraw=mn, escape=close,
                        retract_mm=float(gw[pk] - gw[mn]), p0=float(e["p0"])))
    return out


def key_frames(r, ph):
    pl = float(r["path_len"]) if np.isfinite(r["path_len"]) and r["path_len"] > 0 else max(float(r["proj_s"].max()), 1.0)
    T = r["T"]; keys = []
    hit = np.nonzero(r["proj_s"] >= 0.25 * pl)[0]
    keys.append(("k0_start", int(hit[0]) if len(hit) else 0, "approach"))
    for p in ph:
        keys.append(("k1_stall_e%d" % p["i"], p["first"], "stall e%d: frontier stops" % p["i"]))
        keys.append(("k2_load_e%d" % p["i"], p["peak"], "stall e%d: peak load" % p["i"]))
        if p["retract_mm"] >= 1.0 and p["withdraw"] != p["peak"]:
            keys.append(("k3_withdraw_e%d" % p["i"], p["withdraw"], "stall e%d: withdrawn %.0f mm" % (p["i"], p["retract_mm"])))
        keys.append(("k4_escape_e%d" % p["i"], p["escape"], "stall e%d: escape" % p["i"]))
    keys.append(("k9_end", T - 1, "end: %s" % ("success" if r["success"] else r["reason"] or "failed")))
    return keys


def sequence(r, ph, cadence, stall_cadence, keys, dense_halfwidth=15):
    """Base cadence everywhere; dense cadence only around the phase points
    (stall onset, peak load, withdrawal, escape). A 230-step stall at 2-step
    cadence alone is 115 frames and a 7 MB GIF; the action is at the phase
    points, the middle of a long stall is the wire not moving."""
    T = r["T"]; steps = set(range(0, T, cadence)) | {T - 1} | {s for _, s, _ in keys}
    for p in ph:
        for c in (p["first"], p["peak"], p["withdraw"], p["escape"]):
            steps |= set(range(max(0, c - dense_halfwidth), min(T, c + dense_halfwidth + 1), stall_cadence))
    return sorted(steps)


# ------------------------------------------------------------------ drawing
def _bbox(r):
    pts = [r["path"]] if len(r["path"]) else []
    pts += [r["tip"][np.isfinite(r["tip"]).all(1)]]
    P = np.concatenate([p for p in pts if len(p)], 0)
    lo, hi = P.min(0) - 25, P.max(0) + 25
    c = (lo + hi) / 2; half = float((hi - lo).max()) / 2
    return c - half, c + half


def _draw_scene(ax, r, t, lims, trail=30, zoom=None):
    lo, hi = lims
    for name, c, on in r["cl"]:
        ax.plot(c[:, 0], c[:, 1], c[:, 2], color=COL["cl_on"] if on else COL["cl_off"], lw=1.0 if on else 0.6,
                alpha=0.8 if on else 0.35)
    if len(r["path"]):
        ax.plot(r["path"][:, 0], r["path"][:, 1], r["path"][:, 2], color=COL["path"], lw=2.0, label="planned path")
    tg = r["target"]
    if np.isfinite(tg).all():
        ax.scatter(*tg, color=COL["target"], marker="X", s=90, edgecolors="k", linewidths=0.6, label="target", depthshade=False)
    a = max(0, t - trail); tr = r["tip"][a:t + 1]; tr = tr[np.isfinite(tr).all(1)]
    if len(tr) > 1:
        ax.plot(tr[:, 0], tr[:, 1], tr[:, 2], color="#888", lw=0.8, alpha=0.7)
    drew_body = False
    for key, col, lab in (("cath", COL["cath"], "catheter"), ("gw", COL["wire"], "guidewire")):
        p = r[key][t]; p = p[np.isfinite(p).all(1)]
        if len(p):
            ax.plot(p[:, 0], p[:, 1], p[:, 2], color=col, lw=2.4, label=lab)
            ax.scatter(p[0, 0], p[0, 1], p[0, 2], color=col, s=26, edgecolors="k", linewidths=0.4, depthshade=False)
            drew_body = True
    if not drew_body:   # log-only tier: no device bodies, mark the guidewire tip
        tp = r["tip"][t]
        if np.isfinite(tp).all():
            ax.scatter(tp[0], tp[1], tp[2], color=COL["wire"], s=40, edgecolors="k", linewidths=0.5, depthshade=False,
                       label="guidewire tip (logs carry no body)")
    if zoom is not None:
        c = zoom; h = 40.0
        ax.set_xlim(c[0] - h, c[0] + h); ax.set_ylim(c[1] - h, c[1] + h); ax.set_zlim(c[2] - h, c[2] + h)
    else:
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    ax.set_xlabel("x (mm)", fontsize=7); ax.set_ylabel("y (mm)", fontsize=7); ax.set_zlabel("z (mm)", fontsize=7)
    ax.tick_params(labelsize=6)


def _draw_trace(axp, axs, r, ph, t):
    T = r["T"]; x = np.arange(T)
    pl = float(r["path_len"]) if np.isfinite(r["path_len"]) and r["path_len"] > 0 else max(float(r["proj_s"].max()), 1.0)
    for p in ph:
        for ax in (axp, axs):
            ax.axvspan(p["first"], p["escape"], color=COL["stall"], alpha=0.6, lw=0)
    axp.plot(x, r["proj_s"] / pl, color=COL["prog"], lw=1.4)
    axp.set_ylabel("progress (proj_s / path_len)", fontsize=7); axp.set_ylim(-0.02, 1.08)
    slack = r["ins_gw"] - r["proj_s"]
    axs.plot(x, slack, color=COL["slack"], lw=1.2, label="guidewire slack (mm)")
    cs = r["cath_slack"]
    if np.isfinite(cs).any():
        axs.plot(x, cs, color=COL["cslack"], lw=1.0, alpha=0.8, label="catheter slack (mm)")
    axs.axhline(0, color="#999", lw=0.6)
    axs.set_ylabel("slack (mm)", fontsize=7); axs.set_xlabel("step", fontsize=7)
    axs.legend(fontsize=6, frameon=False, loc="upper left")
    for ax in (axp, axs):
        ax.axvline(t, color="#b23a48", lw=1.2); ax.set_xlim(0, T - 1); ax.tick_params(labelsize=6)
        for s in ("top", "right"): ax.spines[s].set_visible(False)


def render_frame(r, ph, t, lims, title, key=False, dpi=110):
    fig = plt.figure(figsize=(11.5, 5.6), dpi=dpi)
    gs = GridSpec(2, 2, width_ratios=[1.25, 1], height_ratios=[1, 1], figure=fig, wspace=0.28, hspace=0.35)
    ax3 = fig.add_subplot(gs[:, 0], projection="3d")
    _draw_scene(ax3, r, t, lims)
    ax3.legend(loc="upper left", fontsize=6, frameon=False)
    if key:
        tip = r["tip"][t]
        if np.isfinite(tip).all():
            axi = fig.add_axes([0.30, 0.56, 0.20, 0.30], projection="3d")
            _draw_scene(axi, r, t, lims, trail=15, zoom=tip)
            axi.set_xticks([]); axi.set_yticks([]); axi.set_zticks([])
            axi.set_xlabel(""); axi.set_ylabel(""); axi.set_zlabel("")
            axi.text2D(0.5, -0.02, "tip ±40 mm", transform=axi.transAxes, fontsize=6, ha="center", va="top")
    axp = fig.add_subplot(gs[0, 1]); axs = fig.add_subplot(gs[1, 1], sharex=axp)
    _draw_trace(axp, axs, r, ph, t)
    fig.suptitle(title, fontsize=9)
    return fig


def fig_to_png(fig):
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=fig.dpi); plt.close(fig); buf.seek(0); return buf


def render_trace_only(r, ph, out_path, title):
    fig, (axp, axs) = plt.subplots(2, 1, figsize=(7, 4.2), sharex=True, dpi=120)
    _draw_trace(axp, axs, r, ph, r["T"] - 1)
    for p in ph:
        for s, lab in ((p["first"], "stall"), (p["peak"], "load"), (p["escape"], "escape")):
            axp.annotate(lab, (s, float(r["proj_s"][s]) / max(float(r["path_len"]), 1.0)), fontsize=6, xytext=(0, 6),
                         textcoords="offset points", ha="center")
    fig.suptitle(title, fontsize=9); fig.tight_layout(); fig.savefig(out_path); plt.close(fig)


# ------------------------------------------------------------------ episode
def process(args):
    path, out_root, block, opts = args
    try:
        r = load_record(path) if isinstance(path, str) else path
        path = r["file"]
    except Exception as e:
        return dict(file=str(path), error="load: %s" % e)
    if r["T"] < 3:
        return dict(file=path, error="too short")
    cid, cname, f, ev = type_record(r)
    if opts["types"] is not None and cid not in opts["types"]:
        return dict(file=path, seed=r["seed"], mesh=r["mesh_fp"], type=cid, cname=cname, steps=r["T"],
                    success=r["success"], skipped=True)
    ph = phases(r, ev); keys = key_frames(r, ph)
    seq = sequence(r, ph, opts["cadence"], opts["stall_cadence"], keys) if opts["movie"] != "none" or opts["dump"] else []
    sid = ("s%04d" % r["seed"]) if r["seed"] >= 0 else "sNA_pid%d_ep%d" % (r["pid"], r["ep"])
    ep_dir = os.path.join(out_root, block, TYPE_FOLDER.get(cid, "untyped"),
                          "%s_%s_%dst_%s" % (sid, r["mesh_fp"] or "unknown", r["T"], "success" if r["success"] else "fail"))
    os.makedirs(os.path.join(ep_dir, "key"), exist_ok=True)
    lims = _bbox(r)
    base = "seed %d · %s · %s · %s" % (r["seed"], r["mesh_fp"], cname, "success" if r["success"] else (r["reason"] or "fail"))
    keymap = {}
    for name, s, lab in keys:
        fn = "%s_s%03d.png" % (name, s + 1)
        fig = render_frame(r, ph, s, lims, "%s\nstep %d/%d — %s" % (base, s + 1, r["T"], lab), key=True, dpi=opts["dpi"])
        fig.savefig(os.path.join(ep_dir, "key", fn)); plt.close(fig); keymap[name] = dict(step=s + 1, file="key/" + fn, label=lab)
    render_trace_only(r, ph, os.path.join(ep_dir, "trace.png"), base)
    n_frames = 0
    if seq:
        from PIL import Image
        frames = []
        stall_steps = set()
        for p in ph: stall_steps |= set(range(p["first"], p["escape"] + 1))
        if opts["dump"]: os.makedirs(os.path.join(ep_dir, "seq"), exist_ok=True)
        for s in seq:
            lab = "stall" if s in stall_steps else ""
            fig = render_frame(r, ph, s, lims, "%s\nstep %d/%d %s" % (base, s + 1, r["T"], lab), key=False, dpi=max(60, int(opts["dpi"] * 0.75)))
            buf = fig_to_png(fig); im = Image.open(buf)
            if opts["gif_width"] and im.width > opts["gif_width"]:
                im.thumbnail((opts["gif_width"], 10000))
            im = im.convert("P", palette=Image.ADAPTIVE, colors=128)
            frames.append(im)
            if opts["dump"]:
                with open(os.path.join(ep_dir, "seq", "f%04d_s%03d.png" % (len(frames), s + 1)), "wb") as fh: fh.write(buf.getvalue())
        if opts["movie"] == "gif" and frames:
            frames[0].save(os.path.join(ep_dir, "episode.gif"), save_all=True, append_images=frames[1:],
                           duration=int(1000 / opts["fps"]), loop=0, optimize=False)
        n_frames = len(frames)
    json.dump(dict(file=os.path.basename(path), seed=r["seed"], mesh_fp=r["mesh_fp"], pid=r["pid"], ep=r["ep"],
                   steps=r["T"], success=r["success"], reason=r["reason"], type=cid, type_name=cname,
                   path_len=float(r["path_len"]), events=ev, phases=ph, key_frames=keymap,
                   sequence_steps=[s + 1 for s in seq], n_frames=n_frames,
                   features={k: v for k, v in f.items() if not isinstance(v, list)}),
              open(os.path.join(ep_dir, "events.json"), "w"), indent=1, default=float)
    return dict(file=path, seed=r["seed"], mesh=r["mesh_fp"], type=cid, cname=cname, steps=r["T"], success=r["success"],
                n_events=len(ev), n_key=len(keys), n_frames=n_frames, dir=os.path.relpath(ep_dir, out_root))


def write_indexes(out_root, rows):
    by_block = {}
    for row in rows:
        if "error" in row: continue
        by_block.setdefault(row.get("block", ""), []).append(row)
    by_seed = {}
    for block, rs in by_block.items():
        with open(os.path.join(out_root, block, "index.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["seed", "mesh", "type", "cname", "steps", "success", "n_events", "n_key", "n_frames", "dir", "skipped"])
            w.writeheader()
            for r in sorted(rs, key=lambda x: (x["seed"], x["mesh"])):
                w.writerow({k: r.get(k, "") for k in w.fieldnames}); by_seed.setdefault(r["seed"], []).append((block, r))
    os.makedirs(os.path.join(out_root, "by_seed"), exist_ok=True)
    for seed, items in by_seed.items():
        rows_html = "".join(
            "<tr><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>" % (
                b, r["mesh"], r["cname"], r["steps"], "success" if r["success"] else "fail",
                ('<a href="../%s/episode.gif">movie</a> · <a href="../%s/trace.png">trace</a>' % (r["dir"].replace(os.sep, "/"), r["dir"].replace(os.sep, "/"))) if not r.get("skipped") else "(not rendered)")
            for b, r in sorted(items, key=lambda x: (x[0], x[1].get("mesh", ""), x[1].get("steps", 0))))
        open(os.path.join(out_root, "by_seed", "s%04d.html" % seed), "w").write(
            "<title>seed %d</title><style>body{font:14px system-ui;padding:20px}table{border-collapse:collapse}td,th{padding:4px 10px;border-bottom:1px solid #ddd}</style>"
            "<h2>seed %d across blocks</h2><table><tr><th>block</th><th>anatomy</th><th>type</th><th>steps</th><th>outcome</th><th></th></tr>%s</table>" % (seed, seed, rows_html))


def log_jobs(a, opts):
    """Tier 1 jobs from traj_extract output: <log_dir>/<tag>.features.jsonl + .series.npz."""
    tag = a.from_logs
    feats = os.path.join(a.log_dir, tag + ".features.jsonl"); ser = os.path.join(a.log_dir, tag + ".series.npz")
    series = np.load(ser)
    if "tip" not in series.files:
        sys.exit("series %s has no per-step tip: re-run traj_extract.py (it stores tip since Sep 19)" % ser)
    frame = None
    if a.frame:
        v = [float(x) for x in a.frame.split(",")]; frame = ((v[0], v[1]), tuple(v[2:5]))
    scenery = build_scenery(a.scenery, frame) if a.scenery else None
    if scenery is not None: print("scenery anatomies:", len(scenery))
    want = {s.strip() for s in a.log_select.split(",") if s.strip()}
    rows = []
    with open(feats) as fh:
        for idx, line in enumerate(fh):
            row = json.loads(line)
            if row.get("complete"): rows.append((idx, row))
    # validation block index per episode: eval episodes sorted by wall time, a
    # new block whenever the gap exceeds 30 min (same rule as traj_load.eval_blocks)
    val = sorted([(row["wt_start"], idx) for idx, row in rows if row.get("is_eval") and row.get("wt_start")])
    block_of = {}; b = 0
    for k, (wt, idx) in enumerate(val):
        if k and wt - val[k - 1][0] > 1800: b += 1
        block_of[idx] = b
    jobs = []
    for idx, row in rows:
        kind = "val" if row.get("is_eval") else "explore"
        if want and kind not in want and not (row.get("seed") is not None and str(row["seed"]) in want): continue
        r = load_log_episode(row, series, idx, scenery)
        block = "%s_val%02d" % (tag, block_of.get(idx, 0)) if kind == "val" else "%s_explore" % tag
        jobs.append((r, a.out_dir, block, opts))
    return jobs


def reindex(out_root):
    """Rebuild by_seed/*.html from every <block>/index.csv under out_root (each render
    invocation only knows its own block; this merges them across checkpoints/reps)."""
    rows = []
    for idx in sorted(glob.glob(os.path.join(out_root, "*", "index.csv"))):
        block = os.path.basename(os.path.dirname(idx))
        with open(idx) as fh:
            for r in csv.DictReader(fh):
                r["block"] = block; r["seed"] = int(r["seed"]); r["success"] = r["success"] == "True"
                r["skipped"] = r.get("skipped", "") == "True"; rows.append(r)
    by_seed = {}
    for r in rows: by_seed.setdefault(r["seed"], []).append((r["block"], r))
    os.makedirs(os.path.join(out_root, "by_seed"), exist_ok=True)
    for seed, items in by_seed.items():
        items.sort(key=lambda x: (x[0], x[1].get("mesh", "")))
        rows_html = "".join(
            "<tr><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>" % (
                b, r["mesh"], r["cname"], r["steps"], "success" if r["success"] else "fail",
                ('<a href="../%s/episode.gif">movie</a> · <a href="../%s/trace.png">trace</a> · <a href="../%s/key/">keys</a>' % ((r["dir"].replace(os.sep, "/"),) * 3)) if r.get("dir") and not r["skipped"] else "(not rendered)")
            for b, r in items)
        n_ok = sum(1 for _, r in items if r["success"])
        open(os.path.join(out_root, "by_seed", "s%04d.html" % seed), "w").write(
            "<title>seed %d</title><style>body{font:14px system-ui;padding:20px}table{border-collapse:collapse}td,th{padding:4px 10px;border-bottom:1px solid #ddd}</style>"
            "<h2>seed %d — %d/%d succeeded across %d blocks</h2><table><tr><th>block</th><th>anatomy</th><th>type</th><th>steps</th><th>outcome</th><th></th></tr>%s</table>" % (
                seed, seed, n_ok, len(items), len(items), rows_html))
    # overview
    seeds = sorted(by_seed); blocks = sorted({b for items in by_seed.values() for b, _ in items})
    cell = lambda s, b: next(("<td style='background:%s'>%s</td>" % ("#cfe8d5" if r["success"] else "#f3c6c6", r["cname"].split(":")[0]) for bb, r in by_seed[s] if bb == b), "<td></td>")
    html = "<title>frontier overview</title><style>body{font:13px system-ui;padding:16px}table{border-collapse:collapse}td,th{padding:2px 6px;border:1px solid #ddd;text-align:center}</style><h2>seed × block (cell = type id; green success, red fail)</h2><table><tr><th>seed</th>%s</tr>%s</table>" % (
        "".join("<th>%s</th>" % b for b in blocks),
        "".join("<tr><td><a href='by_seed/s%04d.html'>%d</a></td>%s</tr>" % (s, s, "".join(cell(s, b) for b in blocks)) for s in seeds))
    open(os.path.join(out_root, "overview.html"), "w").write(html)
    print("reindexed %d seeds over %d blocks -> by_seed/, overview.html" % (len(seeds), len(blocks)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("records_dir", nargs="?", default=None); ap.add_argument("out_dir")
    ap.add_argument("--reindex", action="store_true", help="only rebuild by_seed/ and overview.html from existing <out>/*/index.csv")
    ap.add_argument("--from-logs", default="", metavar="TAG", help="tier 1: render log-derived episodes of this run tag instead of records")
    ap.add_argument("--log-dir", default=os.path.join(HERE, "out"), help="where <TAG>.features.jsonl / .series.npz live")
    ap.add_argument("--log-select", default="val", help="tier 1: 'val', 'explore', 'val,explore' or a comma list of seeds")
    ap.add_argument("--scenery", default="", help="tier 1: records dir to borrow centerlines / planned paths from (same anatomies)")
    ap.add_argument("--block", default="", help="name the output block (default: the records' subfolder) -- e.g. ck1005189_peak")
    ap.add_argument("--frame", default="20,5,0,0,0", help="tier 1 fallback tracking->vessel frame 'rot_z,rot_x,cx,cy,cz' for scenery records that predate the recorder storing it ('' = none)")
    ap.add_argument("--types", default="recovery,failures", help="recovery | failures | all | comma-list of cluster ids")
    ap.add_argument("--cadence", type=int, default=5); ap.add_argument("--stall-cadence", type=int, default=2)
    ap.add_argument("--movie", default="gif", choices=["gif", "none"]); ap.add_argument("--fps", type=float, default=6.0)
    ap.add_argument("--dump-frames", action="store_true"); ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--seeds", default="")
    ap.add_argument("--dpi", type=int, default=100)
    ap.add_argument("--gif-width", type=int, default=900, help="downscale movie frames to this width (0 = keep)")
    a = ap.parse_args()
    if a.reindex:
        reindex(a.out_dir); return
    types = None
    if a.types != "all":
        types = set()
        for tok in a.types.split(","):
            tok = tok.strip()
            if tok in GROUPS: types |= GROUPS[tok]
            elif tok: types.add(int(tok))
    seeds = {int(s) for s in a.seeds.split(",") if s.strip()} if a.seeds else None
    opts = dict(types=types, cadence=a.cadence, stall_cadence=a.stall_cadence, movie=a.movie, fps=a.fps, dump=a.dump_frames, dpi=a.dpi, gif_width=a.gif_width)
    if a.from_logs:
        jobs = log_jobs(a, opts)
        if seeds is not None: jobs = [j for j in jobs if j[0]["seed"] in seeds]
    else:
        if not a.records_dir: sys.exit("records_dir required unless --from-logs")
        files = []
        for root, _, fs in os.walk(a.records_dir):
            for fn in fs:
                if fn.endswith(".npz"):
                    if seeds is not None:
                        try: sd = int(fn.split("_")[0][1:])
                        except ValueError: sd = -1
                        if sd not in seeds: continue
                    files.append(os.path.join(root, fn))
        files.sort()
        jobs = [(p, a.out_dir, a.block or (os.path.relpath(os.path.dirname(p), a.records_dir).replace(".", "") or "records"), opts) for p in files]
    if a.limit: jobs = jobs[:a.limit]
    print("episodes:", len(jobs), "| types:", sorted(types) if types else "all")
    if a.jobs > 1:
        from multiprocessing import Pool
        with Pool(a.jobs) as pool: rows = pool.map(process, jobs)
    else:
        rows = [process(j) for j in jobs]
    for j, row in zip(jobs, rows): row["block"] = j[2]
    n_ok = sum(1 for r in rows if "error" not in r and not r.get("skipped")); n_skip = sum(1 for r in rows if r.get("skipped"))
    n_err = sum(1 for r in rows if "error" in r)
    print("rendered %d, skipped %d (type filter), errors %d" % (n_ok, n_skip, n_err))
    for r in rows:
        if "error" in r: print("  ERROR", r["file"], r["error"])
    write_indexes(a.out_dir, rows)
    from collections import Counter
    print("types:", Counter(r.get("cname") for r in rows if "error" not in r).most_common())


if __name__ == "__main__":
    main()
