"""Second animation of a WALL run (cfg vagina_model = wall): does the device travel INSIDE the vaginal lumen?
py -3.11 (pyvista + matplotlib + PIL).  Nothing here touches the simulation; it reads runs/<tag>/frames (the exported
surfaces and device poses) and runs/<tag>/log.jsonl (the run's own per-station containment measurement).

    APPSIM_OUT=... py -3.11 hybrid/animate_lumen.py render --tag Z7S             # PNG per frame + GIF
    python hybrid/animate_hybrid.py mp4 --tag Z7S --name Z7S_lumen --pk <pk>      # the MP4 (py3.13, as for the main one)

Each frame (1920 x 800) has three parts:
  LEFT    a see-through side view: the vaginal wall is translucent, so the tube/shaft (black) and the ovoids (grey)
          are seen inside it; the cervix and corpus are faint; four station rings are highlighted in colour.
  MIDDLE  a straightened coronal SILHOUETTE along the lumen: at every station the left-right extent of the wall and
          of the device, each station drawn about its own lumen centre, so the wall becomes a straight band and what
          moves is the device relative to the lumen.  Dashed = rest state (the lumen narrows, it never widens).
  RIGHT   four true cross-sections (planes normal to the lumen axis) at the highlighted stations: the wall annulus,
          the rest annulus dashed, the device section filled, and a verdict read off THESE sections (inside the
          lumen / crossing the wall / outside / covering the whole section), so picture and words cannot disagree.
          The run's own logged shaft containment (nearest-vertex rule, see scene_hybrid.wall_metrics) is quoted
          underneath for the record.
Units mm; frame preBT world RAS.  Sections use the fixed lumen axis (meshes/vagina/meta.json axis), LR = patient right,
AP = anterior; the lumen bends by < 3 mm over its length, so a fixed axis is an honest projection.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.dont_write_bytecode = True
os.environ.setdefault("MPLBACKEND", "Agg")

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
for _p in (HERE, os.path.dirname(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import geom  # noqa: E402
import animate_hybrid as AH  # noqa: E402
from animate_hybrid import (P, load_json, load_index, frames_dir, run_dir, device_world, poly, overlay,  # noqa: E402
                            save_gif, TANDEM, OVOIDS, COL_TANDEM, COL_OVOID)

STATIONS = (4, 11, 18, 25)                                   # of the 28 wall stations (introitus = 0, apex = 27)
ST_COL = ("#1b9e77", "#d95f02", "#7570b3", "#e7298a")
LIM = 23.0                                                   # mm half-width of the section panels (ovoids 39.4 across)
PURPLE = (0.55, 0.30, 0.70)
SIDE_W, FULL_W, H = 704, 1920, 800                           # multiples of 16 keep libx264 happy


# ================================================================================= geometry helpers
def wall_rings(meta):
    """surface.obj vertex index of every (station, theta) node on the lumen (inner) and outer surface."""
    w = meta["wall"]
    gi = np.asarray(w["grid_index"], int)
    v2n = np.asarray(meta["surface_obj_vertex_to_tet_node"], int)
    n_ax, n_th, n_rad = int(w["n_axial"]), int(w["n_theta"]), int(w["n_radial"])
    inner = np.full((n_ax, n_th), -1, int)
    outer = inner.copy()
    for vi, node in enumerate(v2n):
        st, lay, th = gi[node]
        if lay == 0:
            inner[st, th] = vi
        elif lay == n_rad:
            outer[st, th] = vi
    if (inner < 0).any() or (outer < 0).any():
        raise RuntimeError("the vagina surface does not carry every inner/outer ring node")
    return inner, outer, np.asarray(w["s"], float)


def slice_points(V, F, n, d):
    """Points where the triangle mesh (V, F) crosses the plane n.x = d."""
    sd = V @ n - d
    pts = []
    for i, j in ((0, 1), (1, 2), (2, 0)):
        a, b = F[:, i], F[:, j]
        m = (sd[a] * sd[b]) < 0
        if m.any():
            t = sd[a[m]] / (sd[a[m]] - sd[b[m]])
            pts.append(V[a[m]] + t[:, None] * (V[b[m]] - V[a[m]]))
    return np.vstack(pts) if pts else np.zeros((0, 3))


def hull2d(p):
    """Convex hull of 2-D points, CCW (monotone chain); None if degenerate."""
    p = np.unique(np.round(np.asarray(p, float), 6), axis=0)
    if len(p) < 3:
        return None
    p = p[np.lexsort((p[:, 1], p[:, 0]))]

    def half(pts):
        h = []
        for q in pts:
            while len(h) >= 2 and (h[-1][0] - h[-2][0]) * (q[1] - h[-2][1]) - (h[-1][1] - h[-2][1]) * (q[0] - h[-2][0]) <= 0:
                h.pop()
            h.append(q)
        return h

    lo, hi = half(p), half(p[::-1])
    h = np.array(lo[:-1] + hi[:-1])
    return h if len(h) >= 3 else None


def signed_area(p):
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


# ================================================================================= the per-frame measurements
class Lumen:
    def __init__(self, tag):
        self.tag = tag
        self.idx = load_index(tag)
        self.fd = frames_dir(tag)
        self.R = np.asarray(self.idx["R_rows"], float)
        meta = load_json(P["meshes"] + "/vagina/meta.json")
        if "wall" not in meta:
            raise SystemExit("meshes/vagina is not a wall mesh (no 'wall' in meta.json); point APPSIM_OUT at the "
                             "wall-run tree")
        self.inner, self.outer, self.s = wall_rings(meta)
        a = geom.unit(np.asarray(meta["axis"]["axis"], float))
        x = np.array([1.0, 0.0, 0.0])
        self.a = a
        self.e_lr = geom.unit(x - (x @ a) * a)                         # patient right, normal to the lumen axis
        self.e_ap = np.cross(a, self.e_lr)                             # anterior, normal to both
        self.dev_app = {p: geom.read_obj("%s/%s.obj" % (P["applicator"], p)) for p in TANDEM + OVOIDS}
        self.rest = {b: geom.read_obj("%s/%s/surface.obj" % (P["meshes"], b)) for b in ("vagina", "cervix", "corpus")
                     if b in self.idx["bodies"]}
        self.rest_sec = self.wall_sections(self.rest["vagina"][0])
        self.log = {}
        lp = run_dir(tag) + "/log.jsonl"
        if os.path.exists(lp):
            with open(lp) as fh:
                for line in fh:
                    r = json.loads(line)
                    self.log[int(r["step"])] = r
        self.n_last = self.idx["frames"][-1]["step"]

    def to2d(self, pts, C):
        q = np.asarray(pts, float) - C
        return np.c_[q @ self.e_lr, q @ self.e_ap]

    def wall_sections(self, Vv):
        """Per station: lumen centre, the two rings in 2-D about that centre, their mean radii, and their LR extents
        ext = [L_out, L_in, R_in, R_out] (the coronal SILHOUETTE of the wall at that level)."""
        Pin, Pout = Vv[self.inner], Vv[self.outer]
        C = Pin.mean(1)
        out = dict(C=C, rings=[], ext=np.zeros((len(C), 4)), r_in=np.zeros(len(C)), r_out=np.zeros(len(C)))
        for k in range(len(C)):
            i2, o2 = self.to2d(Pin[k], C[k]), self.to2d(Pout[k], C[k])
            out["rings"].append((i2, o2))
            out["ext"][k] = [o2[:, 0].min(), i2[:, 0].min(), i2[:, 0].max(), o2[:, 0].max()]
            out["r_in"][k] = np.linalg.norm(i2, axis=1).mean()
            out["r_out"][k] = np.linalg.norm(o2, axis=1).mean()
        return out

    def device_sections(self, dev, C):
        """Per part and station: the convex section polygon (2-D about the lumen centre) and its LR extent."""
        parts = {}
        for p in TANDEM + OVOIDS:
            V, F = self.dev_app[p]
            o = dev["ovoid_origin_mm"] if p in OVOIDS else dev["flange_mm"]
            Vw = device_world(V, o, self.R)
            F = np.asarray(F, int)
            hulls, ext = [], np.full((len(C), 2), np.nan)
            for k in range(len(C)):
                pts = slice_points(Vw, F, self.a, float(C[k] @ self.a))
                h = hull2d(self.to2d(pts, C[k])) if len(pts) else None
                hulls.append(h)
                if h is not None:
                    ext[k] = [h[:, 0].min(), h[:, 0].max()]
            parts[p] = dict(hulls=hulls, ext=ext, world=(Vw, F))
        return parts

    @staticmethod
    def classify(hulls, in_ring, out_ring):
        """Where a group of section polygons sits relative to the wall annulus at one level:
        absent | inside (every polygon within the lumen) | outside (no overlap with the wall or lumen at all) |
        wall (crossing the wall) | engulfs (the whole annulus lies inside the device solid)."""
        from matplotlib.path import Path
        hs = [h for h in hulls if h is not None]
        if not hs:
            return "absent"
        ip, op = Path(in_ring), Path(out_ring)
        covered = np.zeros(len(out_ring), bool)                       # the two ovoid halves meet at the midline:
        for h in hs:                                                  # test the UNION of the group's sections
            covered |= Path(h).contains_points(out_ring)
        if covered.all():
            return "engulfs"
        if all(ip.contains_points(h).all() for h in hs):
            return "inside"
        touch = any(op.contains_points(h).any() or Path(h).contains_points(out_ring).any()
                    or Path(h).contains_points(in_ring).any() for h in hs)
        return "wall" if touch else "outside"

    def verdict(self, sec, parts, k):
        """(text, colour) for the shaft group and the ovoid group at station k, from the SAME sections that are drawn."""
        i2, o2 = sec["rings"][k]
        rin, rout = sec["r_in"][k], sec["r_out"][k]
        out = []                                     # the tube (above the flange, NOT a wall-contact part in Z7S) and
        for names, what in ((["tube"], "tube"), (["shaft"], "shaft"), (OVOIDS, "ovoids")):   # the shaft are judged apart
            hs = [parts[p]["hulls"][k] for p in names]
            cls = self.classify(hs, i2, o2)
            rr = np.concatenate([np.linalg.norm(h, axis=1) for h in hs if h is not None]) if cls != "absent" else None
            num = "" if rr is None else "  (r %.1f-%.1f | lumen %.1f / wall %.1f)" % (rr.min(), rr.max(), rin, rout)
            if cls == "absent":
                out.append(("%s: not at this level" % what, "0.45"))
            elif cls == "inside":
                out.append(("%s: INSIDE the lumen%s" % (what, num), "#1a9641"))
            elif cls == "wall":
                out.append(("%s: crossing the wall%s" % (what, num), "#d98c00"))
            elif cls == "outside":
                out.append(("%s: OUTSIDE, beside the wall%s" % (what, num), "#d7191c"))
            else:
                out.append(("%s: wall INSIDE the ovoid solid%s" % (what, num), "#d7191c"))
        return out

    def log_line(self, step):
        """The run's own logged shaft containment for this step (nearest-vertex rule), quoted for the record."""
        r = self.log.get(step)
        c = (r or {}).get("wall", {}).get("containment") if r else None
        if not c or "shaft" not in c["parts"]:
            return "run log: no containment record for this step"
        s = c["parts"]["shaft"]
        return ("run log, shaft at this step: inside the lumen at %d of %d stations seen, %d grazing the wall, "
                "%d outside (nearest-vertex rule)" % (s["n_inside_lumen"], s["n_stations_seen"],
                                                        s["n_embedded_in_wall"], s["n_outside_wall"]))


# ================================================================================= the see-through side view (pyvista)
class Side:
    def __init__(self, L, size=(SIDE_W, H)):
        pv = self.pv = AH._pv()
        self.L = L
        self.pl = pv.Plotter(off_screen=True, window_size=list(size))
        self.pl.set_background("white")
        try:
            self.pl.enable_depth_peeling(number_of_peels=10, occlusion_ratio=0.0)
        except Exception:
            pass
        col = {b: tuple(load_json(P["meshes"] + "/bodies.json")["bodies"][b]["color"]) for b in L.rest}
        self.col = col
        for b, (V, F) in L.rest.items():
            self.pl.add_mesh(poly(pv, V, F), style="wireframe", color=col[b], opacity=0.22 if b == "vagina" else 0.10,
                             line_width=1, name="rest_" + b)
        # ---- the camera box: the vagina (rest and final), the cervix, the seated ovoids, the flange
        last = L.idx["frames"][-1]
        dev_last = load_json(L.fd + "/" + last["device"])
        pts = [L.rest["vagina"][0], geom.read_obj(L.fd + "/" + last["surfaces"]["vagina"])[0]]
        if "cervix" in L.rest:
            pts += [L.rest["cervix"][0], geom.read_obj(L.fd + "/" + last["surfaces"]["cervix"])[0]]
        oc = np.asarray(dev_last["ovoid_centres_mm"], float)
        pts += [oc - 24.0, oc + 24.0, np.atleast_2d(dev_last["flange_mm"])]
        V = np.vstack(pts)
        lo, hi = V.min(0) - 8.0, V.max(0) + 8.0
        hi[2] += 0.16 * (hi[2] - lo[2])                                # headroom for the overlay text
        self.c = 0.5 * (lo + hi)
        self.pscale = max(0.5 * (hi[2] - lo[2]), 0.5 * (hi[1] - lo[1]) / (size[0] / size[1]))
        self.cam = [(self.c[0] - 600.0, self.c[1], self.c[2]), tuple(self.c), (0.0, 0.0, 1.0)]
        self._shown = False

    def draw(self, fr, Vv, Fv, parts, sec, txt):
        pv, pl, L = self.pv, self.pl, self.L
        for b in ("cervix", "corpus"):
            if b in L.rest:
                V, F = geom.read_obj(L.fd + "/" + fr["surfaces"][b])
                pl.add_mesh(poly(pv, V, F), color=self.col[b], opacity=0.22 if b == "cervix" else 0.14,
                            smooth_shading=True, name="body_" + b)
        m = poly(pv, Vv, Fv)
        pl.add_mesh(m, color=self.col["vagina"], opacity=0.30, smooth_shading=True, specular=0.1, name="body_vagina")
        pl.add_mesh(m, style="wireframe", color=self.col["vagina"], opacity=0.30, line_width=1, name="wire_vagina")
        for p in TANDEM + OVOIDS:
            Vw, F = parts[p]["world"]
            pl.add_mesh(poly(pv, Vw, F), color=COL_OVOID if p in OVOIDS else COL_TANDEM,
                        opacity=0.45 if p in OVOIDS else 1.0, smooth_shading=True, name="dev_" + p)
        lab_pts, labs = [], []
        for j, k in enumerate(STATIONS):
            ring = Vv[L.inner[k]]
            line = pv.lines_from_points(np.vstack([ring, ring[:1]]))
            pl.add_mesh(line, color=ST_COL[j], line_width=5, render_lines_as_tubes=True, name="ring_%d" % j)
            lab_pts.append(sec["C"][k] - 16.0 * L.e_ap)                 # label posterior of the ring
            labs.append("S%d" % (j + 1))
        try:
            pl.add_point_labels(np.array(lab_pts), labs, font_size=15, text_color="black", shape=None,
                                show_points=False, always_visible=True, name="st_labels")
        except Exception:
            pass
        pl.add_text(txt, position="upper_left", font_size=10, color="black", name="info")
        pl.add_text("see-through side view from the patient's left (ANTERIOR left, SUPERIOR up), nothing cut:\n"
                    "vaginal wall translucent purple, tube/shaft black, ovoids grey, cervix/corpus faint,\n"
                    "rest state = wireframe; coloured rings = the section levels S1-S4 of the right-hand panels",
                    position="lower_left", font_size=8, color="black", name="cap")
        pl.camera_position = self.cam
        pl.camera.parallel_projection = True
        pl.camera.parallel_scale = self.pscale
        pl.renderer.reset_camera_clipping_range()
        if not self._shown:
            pl.show(auto_close=False)
            self._shown = True
        pl.render()
        return pl.screenshot(return_img=True)

    def close(self):
        self.pl.close()


# ================================================================================= the section panels (matplotlib)
def draw_sections(L, dev, sec, parts, txt):
    import matplotlib.pyplot as plt
    from matplotlib.patches import PathPatch, Polygon
    from matplotlib.path import Path

    s = L.s
    fig = plt.figure(figsize=((FULL_W - SIDE_W) / 100.0, H / 100.0), dpi=100)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 1.0], left=0.06, right=0.985, top=0.885, bottom=0.10,
                          wspace=0.30, hspace=0.34)
    axL = fig.add_subplot(gs[:, 0])
    axs = [fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 2])]
    fig.text(0.06, 0.975, txt.split("\n")[0] + "\n" + txt.split("\n")[1], ha="left", va="top", fontsize=9.5)

    # ---- the straightened coronal silhouette: LR extents of the wall and of the device at every level
    ex, rx = sec["ext"], L.rest_sec["ext"]
    axL.fill_betweenx(s, ex[:, 0], ex[:, 1], color=PURPLE, alpha=0.55, lw=0, label="vaginal wall (now)")
    axL.fill_betweenx(s, ex[:, 2], ex[:, 3], color=PURPLE, alpha=0.55, lw=0)
    for j in range(4):
        axL.plot(rx[:, j], s, ls="--", lw=0.9, color=PURPLE, alpha=0.8, label="wall at rest" if j == 0 else None)
    axL.axvline(0.0, color="0.5", lw=0.6, ls=":")
    labelled = set()
    for p in OVOIDS + TANDEM:                                          # ovoids first so the shaft is drawn on top
        e = parts[p]["ext"]
        okp = np.isfinite(e).all(1)
        if okp.any():
            grp = "ovoids" if p in OVOIDS else "tube / shaft"
            axL.fill_betweenx(s, e[:, 0], e[:, 1], where=okp, color=COL_OVOID if p in OVOIDS else COL_TANDEM,
                              alpha=0.95 if p in TANDEM else 0.70, lw=0, label=None if grp in labelled else grp)
            labelled.add(grp)
    for j, k in enumerate(STATIONS):
        axL.axhline(s[k], color=ST_COL[j], lw=1.2, ls="-", alpha=0.9)
        axL.text(-LIM + 0.6, s[k] + 0.5, "S%d" % (j + 1), color=ST_COL[j], fontsize=9, fontweight="bold", va="bottom")
    axL.set_xlim(-LIM, LIM)
    axL.set_ylim(s[0] - 3.0, s[-1] + 3.0)
    axL.set_aspect("equal")
    axL.set_xlabel("distance from the lumen centre, mm\n(patient left  <-   ->  patient right)", fontsize=8.5)
    axL.set_ylabel("position along the vagina, mm   (introitus  ->  apex)", fontsize=8.5)
    axL.set_title("straightened coronal silhouette\n(left-right extent at every level,\neach level about its own "
                  "lumen centre)", fontsize=9)
    axL.tick_params(labelsize=8)
    axL.legend(loc="lower right", fontsize=7.5, framealpha=0.9)

    # ---- the four cross-sections
    for j, k in enumerate(STATIONS):
        ax = axs[j]
        i2, o2 = sec["rings"][k]
        ri, ro = L.rest_sec["rings"][k]
        if signed_area(o2) < 0:
            o2 = o2[::-1]
        if signed_area(i2) > 0:
            i2 = i2[::-1]
        verts = np.vstack([o2, o2[:1], i2, i2[:1]])
        codes = ([Path.MOVETO] + [Path.LINETO] * (len(o2) - 1) + [Path.CLOSEPOLY]
                 + [Path.MOVETO] + [Path.LINETO] * (len(i2) - 1) + [Path.CLOSEPOLY])
        ax.add_patch(PathPatch(Path(verts, codes), facecolor=PURPLE, alpha=0.55, edgecolor=PURPLE, lw=1.0))
        for r in (ri, ro):
            ax.plot(np.r_[r[:, 0], r[0, 0]], np.r_[r[:, 1], r[0, 1]], ls="--", lw=0.9, color=PURPLE, alpha=0.8)
        for p in TANDEM + OVOIDS:
            h = parts[p]["hulls"][k]
            if h is not None:
                ax.add_patch(Polygon(h, closed=True, facecolor=COL_OVOID if p in OVOIDS else COL_TANDEM,
                                     alpha=0.75 if p in OVOIDS else 0.95, edgecolor="black", lw=0.6))
        ax.plot([0.0], [0.0], "+", color="0.3", ms=8, mew=0.8)
        ax.set_xlim(-LIM, LIM)
        ax.set_ylim(-LIM, LIM)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7.5)
        for sp in ax.spines.values():
            sp.set_edgecolor(ST_COL[j])
            sp.set_linewidth(2.0)
        ax.set_title("S%d   station %d,  s = %+.0f mm along the vagina" % (j + 1, k, s[k]), color=ST_COL[j],
                     fontsize=9, fontweight="bold")
        ax.set_xlabel("patient right ->  (mm)", fontsize=7.5)
        ax.set_ylabel("anterior ->  (mm)", fontsize=7.5)
        y = 0.975
        for t, c in L.verdict(sec, parts, k):                          # verdict on one line, its numbers on the next
            head, _, num = t.partition("  (")
            ax.text(0.02, y, head + ("\n      " + num.rstrip(")") if num else ""), transform=ax.transAxes,
                    fontsize=6.8, color=c, va="top", linespacing=1.15, fontweight="bold" if c != "0.45" else "normal",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
            y -= 0.115 if num else 0.06
    fig.text(0.06, 0.008, "cross-sections: planes normal to the lumen axis at S1-S4; purple = vaginal wall now, "
                          "dashed = wall at rest (both about their own lumen centre); black = tube/shaft, "
                          "grey = ovoids.\nverdicts are read off these same sections (r = distance of the device "
                          "section from the lumen centre).\n" + L.log_line(int(dev["step"])),
             fontsize=7.5, va="bottom")
    return fig


# ================================================================================= render
def render(tag, name=None, every=1, limit=None, fps=9.0, hold_last_s=1.5, gif_max_mb=12.0):
    import matplotlib.pyplot as plt
    from PIL import Image

    L = Lumen(tag)
    name = name or (tag + "_lumen")
    fr = L.idx["frames"]
    if every > 1:
        fr = fr[::every] + ([fr[-1]] if (len(fr) - 1) % every else [])
    if limit:
        fr = fr[:limit]
    od = P["anim"] + "/frames_" + name
    os.makedirs(od, exist_ok=True)
    side = Side(L)
    pngs = []
    for i, f in enumerate(fr):
        dev = load_json(L.fd + "/" + f["device"])
        Vv, Fv = geom.read_obj(L.fd + "/" + f["surfaces"]["vagina"])
        sec = L.wall_sections(Vv)
        parts = L.device_sections(dev, sec["C"])
        txt, _ = overlay(tag, dev, list(L.idx["bodies"]), L.n_last, False, True)
        left = side.draw(f, Vv, Fv, parts, sec, txt)
        fig = draw_sections(L, dev, sec, parts, txt)
        right = od + "/_sections.png"
        fig.savefig(right, dpi=100, facecolor="white")
        plt.close(fig)
        im = Image.new("RGB", (FULL_W, H), "white")
        im.paste(Image.fromarray(np.asarray(left)[:, :, :3]).resize((SIDE_W, H)), (0, 0))
        im.paste(Image.open(right).convert("RGB").resize((FULL_W - SIDE_W, H)), (SIDE_W, 0))
        p = od + "/step_%04d.png" % dev["step"]
        im.save(p)
        pngs.append(p)
        print("frame %d/%d  step %d %s u=%.2f" % (i + 1, len(fr), dev["step"], dev["phase"], dev["u"]), flush=True)
    side.close()
    if os.path.exists(od + "/_sections.png"):
        os.remove(od + "/_sections.png")
    gif = save_gif(pngs, P["anim"] + "/insertion_%s.gif" % name, fps, hold_last_s, gif_max_mb, width0=1200)
    meta = dict(tag=tag, name=name, bodies=["vagina", "cervix", "corpus"], zoom=True, clip=False, backdrop=False,
                kind="lumen (see-through side view + straightened coronal section + 4 cross-sections)",
                stations=list(STATIONS), station_s_mm=[round(float(L.s[k]), 2) for k in STATIONS],
                pngs=[os.path.basename(p) for p in pngs], fps=fps, hold_last_s=hold_last_s, size=[FULL_W, H],
                png_size_mb=round(sum(os.path.getsize(p) for p in pngs) / 1e6, 1), gif=gif)
    with open(od + "/frames.json", "w") as fh:
        json.dump(meta, fh, indent=1)
    print("[render] %d PNG in %s (%.1f MB), gif %s" % (len(pngs), od, meta["png_size_mb"], gif["path"]))
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["render"])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--name", default=None, help="output basename (default <tag>_lumen)")
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--fps", type=float, default=9.0)
    ap.add_argument("--hold", type=float, default=1.5)
    ap.add_argument("--gif-max-mb", type=float, default=12.0)
    a = ap.parse_args()
    os.makedirs(P["anim"], exist_ok=True)
    render(a.tag, name=a.name, every=a.every, limit=a.limit, fps=a.fps, hold_last_s=a.hold, gif_max_mb=a.gif_max_mb)


if __name__ == "__main__":
    main()
