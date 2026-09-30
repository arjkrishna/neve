"""Render trees (HOST, any python >= 3.8, numpy only): the shadow APPSIM_OUT one run's figures and videos are drawn
through.

The renderers (animate_hybrid.py, animate_lumen.py, label_views.py, overlay_views.py) read the rest meshes from
<APPSIM_OUT>/hybrid/meshes/<body>/ and the device from <APPSIM_OUT>/hybrid/applicator/<part>.obj.  A run was
simulated on ITS scene root (a wall run loads meshes/_scene_<vagina_wall_dir>/, vagina_wall.scene_mesh_root) and
ITS applicator variant (cfg applicator_dir), so it has to be rendered through a tree whose junctions point there.
MEASURED why it matters (README, "Videos"): the lumen view indexes the wall mesh's grid and fails with an
IndexError on a mismatched wall (G11-G13 through the tet26u tree); a mismatched applicator draws the wrong device
without any error.  This builds that tree from the run's cfg.json instead of by hand:

    <tree>/inputs             -> <out>/inputs                     canal.npz, tandem_path.npz (landmarks, canal)
    <tree>/validation         -> <out>/validation                 alignment.json (bt_overlay / eval through the tree)
    <tree>/hybrid/meshes      -> <out>/hybrid/meshes/_scene_<vagina_wall_dir>   wall run   | <out>/hybrid/meshes
    <tree>/hybrid/applicator  -> <out>/hybrid/<applicator_dir>    (TF0: applicator_v4, the tandem body only)
    <tree>/hybrid/runs        -> <out>/hybrid/runs                or --runs DIR (e.g. a synthetic run for tests)
    <tree>/hybrid/eval        -> <out>/hybrid/eval                metrics.json / tf_metrics.json (numbers on figures)
    <tree>/hybrid/figs        -> <out>/hybrid/figs                or --figs DIR: a separate folder that receives the
                                                                  figures (tests); its overlay_prep/<tag>_PELVIS links
                                                                  the data's bt_overlay prep of the run, or --prep DIR
                                                                  (e.g. a prep written with bt_overlay.py --out)
    <tree>/hybrid/logs        -> <out>/hybrid/logs

    python -P hybrid/render_tree.py --tag TF0                     # tree at <APPSIM_SCRATCH>/render_trees/TF0
    python -P hybrid/render_tree.py --tag G32 --tree <dir> [--figs <dir>] [--runs <dir>] [--force]
    APPSIM_OUT=<tree> py -3.11 hybrid/label_views.py --tag TF0 ...    # then render through it

cfg device_part_files (G32: the 26 mm caps ovoid_L_d26.obj) is NOT a junction: the renderers read it from the run's
cfg.json themselves (animate_hybrid.part_obj), exactly as the scene and eval_hybrid.run_devsurf do.

Windows: directory junctions (_winapi.CreateJunction; `ln -s` in Git Bash COPIES a directory).  POSIX: symlinks.
An existing link pointing elsewhere is replaced only with --force, and only the LINK is removed (os.rmdir on a
junction / os.unlink on a symlink), never its target.  Checks printed after building: the tree's rest vagina has the
vertex count of the run's exported frames, the wall meta is present for a wall run, and every device part the run
loaded exists (a ring_phases run: both ring halves, and every exported frame logs both in its device json
"ring_halves", the pose the renderers draw each half at).  Nothing under <out> is written."""
import argparse
import json
import os
import sys

sys.dont_write_bytecode = True                      # never leave __pycache__ in the repo

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
PARENT = os.path.dirname(HERE)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)
import config  # noqa: E402


def _norm(p):
    p = str(p).replace("\\", "/")
    for pre in ("//?/", "/??/"):
        if p.startswith(pre):
            p = p[len(pre):]
    return os.path.normcase(os.path.abspath(p)).replace("\\", "/")


def link_target(p):
    """Target of a junction / symlink, else None (a real folder or nothing)."""
    try:
        return _norm(os.readlink(p))
    except (OSError, ValueError, NotImplementedError):
        return None


def make_link(link, target, force=False):
    """link -> target (junction on Windows).  Returns 'kept' | 'made' | 'replaced'."""
    link, target = os.path.abspath(link), os.path.abspath(target)
    if not os.path.isdir(target):
        raise SystemExit("render_tree: target %s does not exist" % target)
    cur = link_target(link)
    if cur is not None:
        if cur == _norm(target):
            return "kept"
        if not force:
            raise SystemExit("render_tree: %s already links to %s, not %s (pass --force to replace the LINK)"
                             % (link, cur, target))
        if os.name == "nt":
            os.rmdir(link)                              # removes the junction itself, never the target's contents
        else:
            os.unlink(link)
        how = "replaced"
    elif os.path.exists(link):
        raise SystemExit("render_tree: %s exists and is a real folder/file, not a link; refusing to touch it" % link)
    else:
        how = "made"
    os.makedirs(os.path.dirname(link), exist_ok=True)
    if os.name == "nt":
        import _winapi
        _winapi.CreateJunction(target, link)
    else:
        os.symlink(target, link, target_is_directory=True)
    return how


def run_cfg(out, tag, runs=None):
    fn = "%s/%s/cfg.json" % (runs or out + "/hybrid/runs", tag)
    if not os.path.exists(fn):
        raise SystemExit("render_tree: no %s" % fn)
    with open(fn) as fh:
        return json.load(fh)


def mesh_root(out, cfg):
    """The meshes folder the run's scene loaded (scene_hybrid.load_inputs): meshes/_scene_<vagina_wall_dir> for a wall
    run (its per-body folders are the refreshed copies the scene used), else meshes/."""
    m = out + "/hybrid/meshes"
    if cfg.get("vagina_model", "solid") == "wall":
        wd = cfg.get("vagina_wall_dir") or "vagina_wall"
        cand = "%s/_scene_%s" % (m, wd)
        if not os.path.isdir(cand):
            raise SystemExit("render_tree: wall run on %r but %s does not exist (the scene builds it on its first run)"
                             % (wd, cand))
        return cand
    return m


def n_obj_vertices(fn):
    n = 0
    with open(fn) as fh:
        for line in fh:
            if line.startswith("v "):
                n += 1
    return n


def check(tree, tag, cfg):
    """What a mismatched tree gets wrong, measured on the tree itself.  Returns a list of problems (empty = ok)."""
    hyb = tree + "/hybrid"
    bad, notes = [], []
    fr = "%s/runs/%s/frames/index.json" % (hyb, tag)
    if os.path.exists(fr):
        with open(fr) as fh:
            idx = json.load(fh)
        f0 = idx["frames"][0]["surfaces"]
        for b in sorted(f0):
            nf = n_obj_vertices("%s/runs/%s/frames/%s" % (hyb, tag, f0[b]))
            nr = n_obj_vertices("%s/meshes/%s/surface.obj" % (hyb, b))
            (notes if nf == nr else bad).append("%s: rest %d / frame %d vertices" % (b, nr, nf))
    else:
        notes.append("no frames/index.json (label/overlay stills of the final state only)")
    if cfg.get("vagina_model") == "wall":
        with open(hyb + "/meshes/vagina/meta.json") as fh:
            if "wall" not in json.load(fh):
                bad.append("meshes/vagina/meta.json has no 'wall' block (not a wall mesh)")
    pf = dict(cfg.get("device_part_files") or {})
    with open(hyb + "/applicator/applicator.json") as fh:
        prm = json.load(fh).get("params", {})
    tandem_only = bool((prm.get("tandem_only") or {}).get("value"))
    parts = ["tube", "shaft"]
    # as animate_hybrid.ovoid_body_on: ovoid_mode "off" still loads (and draws) the ovoid body, "none" does not
    if not tandem_only and cfg.get("device_ovoids", True) is not False and cfg.get("ovoid_mode") != "none":
        parts += ["ovoid_L", "ovoid_R"] + (["rod_L", "rod_R"] if cfg.get("device_rods") else []) \
            + (["packing"] if cfg.get("device_packing") else [])
    if cfg.get("device_parts"):
        parts = list(cfg["device_parts"])
    if cfg.get("ring_phases"):
        # S7f: the two ring halves are their own bodies (ovoid_mode "none"); the renderers draw each at its own pose
        # from the frames' device json "ring_halves" (animate_hybrid.part_pose), so every frame must carry them
        parts += [p for p in ("ovoid_L", "ovoid_R") if p not in parts]
        if os.path.exists(fr):
            want = {"ovoid_L", "ovoid_R"}
            miss = []
            for f in idx["frames"]:
                with open("%s/runs/%s/frames/%s" % (hyb, tag, f["device"])) as fh:
                    got = {h.get("part") for h in (json.load(fh).get("ring_halves") or {}).values()}
                if not want <= got:
                    miss.append(int(f["step"]))
            (bad if miss else notes).append(
                "ring_halves: %s" % (("missing in %d frames (first step %d)" % (len(miss), miss[0])) if miss else
                                     "both halves logged in all %d frames" % len(idx["frames"])))
    for p in parts:
        fn = "%s/applicator/%s.obj" % (hyb, pf.get(p, p))
        (notes if os.path.exists(fn) else bad).append("part %s -> %s%s" % (p, os.path.basename(fn),
                                                                           "" if os.path.exists(fn) else " MISSING"))
    notes.append("applicator tandem_only=%s" % tandem_only)
    return bad, notes


def build(tag, tree=None, figs=None, runs=None, force=False, out=None, prep_dir=None):
    P = config.paths()
    out = (out or P["out"]).replace("\\", "/")
    runs_src = (runs or out + "/hybrid/runs").replace("\\", "/")
    cfg = run_cfg(out, tag, runs_src)
    tree = (tree or "%s/render_trees/%s" % (P["scratch"], tag)).replace("\\", "/")
    if _norm(tree).startswith(_norm(out) + "/") and not _norm(tree).startswith(_norm(P["scratch"]) + "/"):
        raise SystemExit("render_tree: put the tree outside %s (or under its scratch/), not inside the data" % out)
    appd = "%s/hybrid/%s" % (out, cfg.get("applicator_dir") or "applicator")
    if figs:
        figs = figs.replace("\\", "/")
        os.makedirs(figs + "/overlay_prep", exist_ok=True)
    links = [("inputs", out + "/inputs"), ("validation", out + "/validation"),
             ("hybrid/meshes", mesh_root(out, cfg)), ("hybrid/applicator", appd), ("hybrid/runs", runs_src),
             ("hybrid/eval", out + "/hybrid/eval"), ("hybrid/logs", out + "/hybrid/logs"),
             ("hybrid/figs", figs or out + "/hybrid/figs")]
    # a separate figs folder links only THIS run's bt_overlay prep (if it exists): bt_overlay.py prep run through the
    # tree for a run without one then writes into the separate folder, never into the data's figs/overlay_prep
    prep = (prep_dir or "%s/hybrid/figs/overlay_prep/%s_PELVIS" % (out, tag)).replace("\\", "/")
    if figs and os.path.isdir(prep):
        links.append((None, prep))
    os.makedirs(tree + "/hybrid", exist_ok=True)
    done = []
    for rel, tgt in links:
        link = (figs + "/overlay_prep/%s_PELVIS" % tag) if rel is None else tree + "/" + rel
        if not os.path.isdir(tgt):
            done.append("%-18s (skipped: %s missing)" % (rel, tgt))
            continue
        done.append("%-18s %-8s -> %s" % (rel or "figs/overlay_prep/%s_PELVIS" % tag, make_link(link, tgt, force), tgt))
    bad, notes = check(tree, tag, cfg)
    return dict(tree=tree, cfg=cfg, links=done, problems=bad, notes=notes)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True, help="run tag: <runs>/<tag>/cfg.json names the wall and the applicator")
    ap.add_argument("--tree", default=None, help="tree folder (default <APPSIM_SCRATCH>/render_trees/<tag>)")
    ap.add_argument("--figs", default=None, help="a real folder for the figures instead of the data's hybrid/figs")
    ap.add_argument("--runs", default=None, help="runs folder instead of the data's hybrid/runs (synthetic runs)")
    ap.add_argument("--out", default=None, help="the real APPSIM_OUT (default: config.paths()['out'])")
    ap.add_argument("--prep", default=None, help="with --figs: the bt_overlay prep folder to link as "
                                                 "overlay_prep/<tag>_PELVIS (default: the data's, if it exists)")
    ap.add_argument("--force", action="store_true", help="replace links that point elsewhere (the links only)")
    a = ap.parse_args()
    r = build(a.tag, a.tree, a.figs, a.runs, a.force, a.out, a.prep)
    print("render tree for %s (applicator_dir %s, vagina_model %s, wall %s):" % (
        a.tag, r["cfg"].get("applicator_dir"), r["cfg"].get("vagina_model"), r["cfg"].get("vagina_wall_dir")))
    for s in r["links"]:
        print("  " + s)
    for s in r["notes"]:
        print("  ok   " + s)
    for s in r["problems"]:
        print("  FAIL " + s)
    print("APPSIM_OUT=%s" % r["tree"])
    if r["problems"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
