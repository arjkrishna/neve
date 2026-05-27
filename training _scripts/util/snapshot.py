"""End-of-episode snapshot helpers (RL_IMPROV_8).

Two backends, selected by ``SNAPSHOT_MODE`` env var:

  ``mesh``        — vessel surface mesh (translucent grey) + per-device wires
                    + planned-path polyline + target marker. Heavier; loads
                    the visu mesh once per worker and decimates it.
  ``centerlines`` — branch centerlines (light grey) + per-device wires +
                    planned-path polyline + target marker. Faster, no mesh.

Outputs land in
``${SNAPSHOT_DIR}/<target_branch>/<reason>/ep<N>_pid<P>_step<S>_<reason>.png``
where ``<target_branch>`` is the named daughter the episode's target lives
on (LCCA/LVA/RCCA/RVA, or ``unknown`` if classification fails) and
``<reason>`` is the termination cause. Designed to be invoked once per
episode at terminal/truncation only, so overhead is amortised across the
whole episode.

All rendering is matplotlib's ``Agg`` backend — works headless inside the
SOFA container and per-worker subprocess without any display setup.
"""

from __future__ import annotations

import os
import sys
import traceback
from typing import Optional

import numpy as np

# Force Agg backend BEFORE pyplot import — workers run headless.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

from eve.util.coordtransform import tracking3d_to_vessel_cs


# Cap face count when rendering the mesh — matplotlib's Poly3DCollection
# becomes unusably slow above ~10k triangles.
MESH_TARGET_FACES = 6000


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _load_decimated_mesh(visu_mesh_path: str):
    """Load the visu mesh via pyvista, decimate to MESH_TARGET_FACES,
    return (verts (V,3), faces (F,3)) as numpy arrays. Cached by caller.
    """
    import pyvista as pv

    mesh = pv.read(visu_mesh_path)
    # PolyData faces is a flat array [n0, i0_0, i0_1, ..., n1, i1_0, ...].
    # For triangular meshes, n=3 throughout. extract_surface() flattens
    # mixed-element grids if the loader returned a UnstructuredGrid.
    if not hasattr(mesh, "faces") or mesh.faces is None:
        mesh = mesh.extract_surface()
    n_faces = mesh.n_faces if hasattr(mesh, "n_faces") else 0
    if n_faces > MESH_TARGET_FACES:
        target_reduction = 1.0 - (MESH_TARGET_FACES / float(n_faces))
        try:
            mesh = mesh.triangulate().decimate(target_reduction)
        except Exception:
            pass

    verts = np.asarray(mesh.points, dtype=np.float32)
    faces_flat = np.asarray(mesh.faces, dtype=np.int64).reshape(-1)
    # Parse [3, i, j, k, 3, i, j, k, ...]
    tris = []
    i = 0
    while i < len(faces_flat):
        n = int(faces_flat[i])
        if n == 3 and i + 3 < len(faces_flat):
            tris.append(faces_flat[i + 1 : i + 4])
        i += n + 1
    if not tris:
        return verts, np.zeros((0, 3), dtype=np.int64)
    faces = np.asarray(tris, dtype=np.int64)
    return verts, faces


def _wire_polylines_vcs(env):
    """Return list of (label, polyline_vcs, color) for each device tracking.

    intervention.fluoroscopy.device_trackings3d returns one polyline per
    device in tracking3d coords, ordered by the device's actual node order
    (tip first after np.flip). Colours: device 0 = green (guidewire),
    device 1 = red (catheter).
    """
    out = []
    try:
        device_trackings = env.intervention.fluoroscopy.device_trackings3d
    except Exception:
        return out

    rot_zx = env.intervention.fluoroscopy.image_rot_zx
    img_center = env.intervention.fluoroscopy.image_center

    palette = [("guidewire", "#00b050"), ("catheter", "#cc2222")]
    for idx, track in enumerate(device_trackings):
        if track is None or len(track) == 0:
            continue
        track_vcs = tracking3d_to_vessel_cs(np.asarray(track), rot_zx, img_center)
        label, color = palette[idx] if idx < len(palette) else (
            f"device{idx}", f"C{idx}"
        )
        out.append((label, np.asarray(track_vcs), color))
    return out


def _target_vcs(env) -> Optional[np.ndarray]:
    try:
        t3 = env.intervention.target.coordinates3d
        if t3 is None:
            return None
        return tracking3d_to_vessel_cs(
            np.asarray(t3),
            env.intervention.fluoroscopy.image_rot_zx,
            env.intervention.fluoroscopy.image_center,
        )
    except Exception:
        return None


def _planned_path_vcs(env) -> Optional[np.ndarray]:
    try:
        return np.asarray(env.pathfinder.path_points_vessel_cs)
    except Exception:
        return None


def _branches_coords(env):
    """Return list of (name, coords (N,3)) for every centerline branch."""
    out = []
    try:
        for br in env.intervention.vessel_tree.branches:
            coords = np.asarray(br.coordinates)
            if coords.size == 0:
                continue
            out.append((getattr(br, "name", "?"), coords))
    except Exception:
        pass
    return out


def _set_equal_aspect(ax, all_pts: np.ndarray) -> None:
    """Equal-aspect 3D axis from a stacked (N,3) points array."""
    if all_pts.size == 0:
        return
    lo = np.min(all_pts, axis=0)
    hi = np.max(all_pts, axis=0)
    center = (lo + hi) / 2.0
    span = max(float(np.max(hi - lo)), 1.0)
    half = span / 2.0
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)


def _annotate_and_save(
    fig, ax, env, episode, ep_step, reason, reward, target_branch, out_path
):
    ax.set_xlabel("x (vessel-CS, mm)")
    ax.set_ylabel("y (vessel-CS, mm)")
    ax.set_zlabel("z (vessel-CS, mm)")
    title = (
        f"ep={episode} step={ep_step} pid={os.getpid()} | reason={reason}\n"
        f"target={target_branch} | reward={reward:.3f}"
    )
    ax.set_title(title, fontsize=9)
    ax.legend(loc="upper left", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _resolve_target_branch(env) -> str:
    """Best-effort string for the target branch name (for the title only)."""
    try:
        # Bench envs stash the option dict in self.intervention.target if
        # configured via target_branch — fall back to coords if not.
        tb = getattr(env.intervention.target, "branch_name", None)
        if tb:
            return str(tb).split(" - ")[-1].replace(".mrk", "")
    except Exception:
        pass
    try:
        t = env.intervention.target.coordinates3d
        if t is not None:
            return f"({t[0]:.0f},{t[1]:.0f},{t[2]:.0f})"
    except Exception:
        pass
    return "?"


def _draw_centerlines(ax, branches, target_path_set):
    """Plot every branch as a polyline; on-path branches highlighted."""
    for name, coords in branches:
        on_path = name in target_path_set
        color = "#1f77b4" if on_path else "#bdbdbd"
        alpha = 0.85 if on_path else 0.45
        lw = 1.4 if on_path else 0.8
        ax.plot(
            coords[:, 0], coords[:, 1], coords[:, 2],
            color=color, alpha=alpha, linewidth=lw,
        )


def _path_branch_names(env):
    """Names of branches the pathfinder considers on-path (best-effort)."""
    names = set()
    try:
        path_set = env.pathfinder.path_branch_set
        for br in path_set:
            n = getattr(br, "name", None)
            if n:
                names.add(n)
    except Exception:
        pass
    return names


def _render_mesh_view(env, episode, ep_step, reason, reward, out_path):
    visu_mesh_path = getattr(env.intervention.vessel_tree, "visu_mesh_path", None)
    if not visu_mesh_path or not os.path.exists(visu_mesh_path):
        # Fall back to centerlines if mesh isn't available.
        _render_centerlines_view(env, episode, ep_step, reason, reward, out_path)
        return

    cache = getattr(env, "_snapshot_mesh_cache", None)
    if cache is None or cache.get("path") != visu_mesh_path:
        verts, faces = _load_decimated_mesh(visu_mesh_path)
        cache = {"path": visu_mesh_path, "verts": verts, "faces": faces}
        env._snapshot_mesh_cache = cache
    verts, faces = cache["verts"], cache["faces"]

    fig = plt.figure(figsize=(8, 9))
    ax = fig.add_subplot(111, projection="3d")

    if faces.size > 0:
        tri_verts = verts[faces]
        coll = Poly3DCollection(
            tri_verts,
            facecolor=(0.65, 0.65, 0.7, 0.18),
            edgecolor=(0.4, 0.4, 0.45, 0.10),
            linewidth=0.2,
        )
        ax.add_collection3d(coll)

    # Planned path
    pp = _planned_path_vcs(env)
    if pp is not None and len(pp) > 1:
        ax.plot(pp[:, 0], pp[:, 1], pp[:, 2],
                color="#ffaa00", linewidth=2.2, label="planned path")

    # Wires
    wire_pts_for_bounds = []
    for label, poly, color in _wire_polylines_vcs(env):
        ax.plot(poly[:, 0], poly[:, 1], poly[:, 2],
                color=color, linewidth=2.2, label=label)
        # Tip marker (first point of device_trackings3d after np.flip = tip)
        ax.scatter(poly[0, 0], poly[0, 1], poly[0, 2],
                   color=color, s=22, edgecolors="black", linewidths=0.4)
        wire_pts_for_bounds.append(poly)

    # Target
    tgt = _target_vcs(env)
    if tgt is not None:
        ax.scatter(tgt[0], tgt[1], tgt[2],
                   color="#ffd400", s=80, marker="X",
                   edgecolors="black", linewidths=0.6, label="target")

    # Equal aspect using mesh bounds (more stable than wire-only).
    bounds = [verts] + wire_pts_for_bounds
    if pp is not None:
        bounds.append(pp)
    if tgt is not None:
        bounds.append(np.atleast_2d(tgt))
    _set_equal_aspect(ax, np.concatenate(bounds, axis=0))

    target_branch = _resolve_target_branch(env)
    _annotate_and_save(fig, ax, env, episode, ep_step, reason, reward,
                       target_branch, out_path)


def _render_centerlines_view(env, episode, ep_step, reason, reward, out_path):
    fig = plt.figure(figsize=(8, 9))
    ax = fig.add_subplot(111, projection="3d")

    branches = _branches_coords(env)
    on_path = _path_branch_names(env)
    _draw_centerlines(ax, branches, on_path)

    pp = _planned_path_vcs(env)
    if pp is not None and len(pp) > 1:
        ax.plot(pp[:, 0], pp[:, 1], pp[:, 2],
                color="#ffaa00", linewidth=2.2, label="planned path")

    wire_pts_for_bounds = []
    for label, poly, color in _wire_polylines_vcs(env):
        ax.plot(poly[:, 0], poly[:, 1], poly[:, 2],
                color=color, linewidth=2.4, label=label)
        ax.scatter(poly[0, 0], poly[0, 1], poly[0, 2],
                   color=color, s=22, edgecolors="black", linewidths=0.4)
        wire_pts_for_bounds.append(poly)

    tgt = _target_vcs(env)
    if tgt is not None:
        ax.scatter(tgt[0], tgt[1], tgt[2],
                   color="#ffd400", s=80, marker="X",
                   edgecolors="black", linewidths=0.6, label="target")

    bounds = [c for _, c in branches] + wire_pts_for_bounds
    if pp is not None:
        bounds.append(pp)
    if tgt is not None:
        bounds.append(np.atleast_2d(tgt))
    if bounds:
        _set_equal_aspect(ax, np.concatenate(bounds, axis=0))

    target_branch = _resolve_target_branch(env)
    _annotate_and_save(fig, ax, env, episode, ep_step, reason, reward,
                       target_branch, out_path)


def _classify_target_branch(env) -> str:
    """Return the LCCA / LVA / RCCA / RVA tag the schedule asked for at
    episode reset. ``BenchEnv5.reset()`` stashes this on
    ``env._target_branch_short`` from ``options["target_branch"]``;
    ``"unknown"`` if not provided (e.g. plain training with random
    target sampling).
    """
    return getattr(env, "_target_branch_short", "unknown") or "unknown"


def save_snapshot(
    env,
    episode,
    ep_step,
    reason,
    reward,
    phase=None,
    base_dir_override: Optional[str] = None,
    mode_override: Optional[str] = None,
) -> Optional[str]:
    """Render and save an end-of-episode snapshot.

    Mode and output dir are read from environment variables so the
    multiprocessing workers (which inherit os.environ at spawn time)
    pick up the same configuration as the parent.

      SNAPSHOT_MODE  = "mesh" | "centerlines" | "none" (default none)
      SNAPSHOT_DIR   = output directory (default ./snapshots under CWD)

    Output path is bucketed by training phase + target branch + reason:
      ``${SNAPSHOT_DIR}/<phase>/<target_branch>/<reason>/ep<N>_pid<P>_step<S>_R<reward>_<reason>.png``

    ``phase`` is one of ``seed`` / ``eval`` / ``explore`` (Plan v5 — lets
    prune_training_snapshots.py keep all seed+eval snapshots and only the
    10-best/10-worst per 100 explore episodes). When None, no phase
    sub-bucket is inserted (legacy heuristic-only-run layout). The episode
    reward is encoded in the filename (``R<+/-><value>``) so the pruner
    can rank without opening the PNGs.

    ``base_dir_override`` / ``mode_override`` (Plan v9 Change 8b) let
    callers (e.g. env5 restore-start verification) write to a separate
    folder + force rendering even when SNAPSHOT_MODE is globally off.

    Returns the path written, or None if disabled / errored.
    """
    mode = (mode_override or os.environ.get("SNAPSHOT_MODE", "none")).lower()
    if mode in ("none", "", "off", "false", "0"):
        return None

    if base_dir_override is not None:
        base_dir = base_dir_override
    else:
        base_dir = os.environ.get("SNAPSHOT_DIR") or os.path.join(os.getcwd(), "snapshots")
    sub = reason if reason else "unknown"
    target_branch = _classify_target_branch(env)
    if phase:
        out_dir = os.path.join(base_dir, str(phase), target_branch, sub)
    else:
        out_dir = os.path.join(base_dir, target_branch, sub)

    try:
        _ensure_dir(out_dir)
    except Exception:
        return None

    fname = (
        f"ep{episode:03d}_pid{os.getpid()}_step{ep_step}"
        f"_R{reward:+.2f}_{sub}.png"
    )
    out_path = os.path.join(out_dir, fname)

    try:
        if mode == "mesh":
            _render_mesh_view(env, episode, ep_step, reason, reward, out_path)
        elif mode == "centerlines":
            _render_centerlines_view(env, episode, ep_step, reason, reward, out_path)
        else:
            return None
    except Exception:
        # Snapshots must NEVER break a run. Log to stderr and move on.
        sys.stderr.write(
            f"[snapshot] failed (mode={mode}, ep={episode}, reason={reason}):\n"
            f"{traceback.format_exc()}\n"
        )
        sys.stderr.flush()
        return None

    return out_path
