"""ApplicatorController (CONTAINER): kinematic tandem(-and-ovoid) driver, distributed node ties (DNT),
sphere-SDF contact (M2), per-step logging, convergence/abort checks and export.  Units mm, mN, s.

DNT (spec 5.1): canal point i has fixed trilinear weights w_ij on its rest hexa (SparseGrid cells are
axis-aligned at rest, so this equals BarycentricMapping).  c_i = sum_j w_ij x_j.  For target t_i and stiffness
k_i the controller writes, on the FEM nodes themselves (LDL-safe),
    k_j = sum_i k_i w_ij ,   tie_tgt_j = x_j + sum_i k_i w_ij (t_i - c_i) / k_j
so the nodal force equals J^T of the mapped spring force at the start of the step; the assembled stiffness is
its diagonal (lumped) approximation.  Held across steps, c_i -> t_i (fixed point = the mapped-spring solution).
"""
import json
import os
import time

import numpy as np

import geom
import Sofa  # noqa: F401
import Sofa.Core


def _rot_about(axis, ang_rad):
    K = geom.skew(geom.unit(axis))
    return np.eye(3) + np.sin(ang_rad) * K + (1 - np.cos(ang_rad)) * K @ K


def build_schedule(ctx):
    """Phases A (approach, no engagement), T (insertion path), D (M2 ovoid inflation). H is open-ended."""
    cfg = ctx["cfg"]; inp = ctx["inp"]; pose = ctx["pose"]
    L = float(inp["app"]["L_iu_mm"])
    a0 = np.array(inp["poses"]["a0"], float); O = np.array(inp["poses"]["O_pre"], float)
    x0 = geom.ortho([1.0, 0.0, 0.0], a0)
    sched = []
    nA = int(cfg["phaseA_steps"]); d0 = float(cfg["phaseA_start_mm"])
    for k in range(1, nA + 1):
        tip = O - d0 * a0 + (d0 * k / nA) * a0
        sched.append(dict(phase="A", u=0.0, F=tip - L * a0, a=a0.copy(), x=x0.copy(), tip=tip, r_scale=0.0))
    path = geom.pose_path(O, a0, x0, np.asarray(pose["F"], float), np.asarray(pose["a"], float), L,
                          cfg["step_mm"], cfg["max_rot_deg_per_step"])
    # x_app is parallel-transported; a spin about the axis (blended with the same smoothstep) makes x(1) = pose x
    a_end = path[-1]["a"]; x_end = path[-1]["x"]; xf = geom.ortho(pose["x"], a_end)
    phi = float(np.arctan2(np.dot(np.cross(x_end, xf), a_end), np.dot(x_end, xf)))
    travel = bool(cfg["m2"]) and cfg.get("m2_mode", "inflate") == "travel"
    for p in path[1:]:
        w = geom.smoothstep((p["u"] - 0.3) / 0.7)
        x = geom.ortho(_rot_about(p["a"], phi * w) @ p["x"], p["a"])
        # travel: the full-size ovoid/cap spheres ride with the device from the first insertion step
        sched.append(dict(phase="T", u=p["u"], F=p["F"], a=p["a"], x=x, tip=p["tip"], r_scale=1.0 if travel else 0.0))
    if cfg["m2"] and not travel:
        nD = int(cfg["phaseD_steps"])
        for j in range(1, nD + 1):
            q = dict(sched[-1]); q.update(phase="D", r_scale=j / float(nD)); sched.append(q)
    return sched


class ApplicatorController(Sofa.Core.Controller):
    def __init__(self, *args, **kw):
        Sofa.Core.Controller.__init__(self, *args, **kw)
        self.ctx = kw["ctx"]; self.sched = kw["schedule"]; self.out_dir = kw.get("out_dir")
        c = self.ctx; cfg = c["cfg"]; self.cfg = cfg
        self.L = float(c["inp"]["app"]["L_iu_mm"]); self.r_rod = float(c["inp"]["app"]["r_tandem_mm"])
        self.n = len(c["canal"])
        self.is_tip = np.zeros(self.n, bool); self.is_tip[-int(cfg["tip_push_n_pts"]):] = True
        self.engaged = np.zeros(self.n, bool); self.released = np.zeros(self.n, bool)
        self.eng_step = np.full(self.n, -1, int)
        # persistent (non-AL) tie targets, world mm.  Spec 5.2 caps the MOTION of a target per step (lateral 1 mm
        # toward the rod axis, axial 1 mm for the tip push); a new tie starts at the tissue point (zero force).
        # (An earlier version capped the target's distance from the CURRENT canal point instead, which saturates
        # the tie force at k*cap and let the canal end lag the rod tip by up to 6 mm; fixed 2026-09-11.)
        self.t_prev = np.zeros((self.n, 3)); self.cap_done = np.zeros(self.n, bool)
        # augmented-Lagrangian target corrections (mm): lateral = equality constraint (canal on the rod axis);
        # axial (tip points only) = UNILATERAL constraint s_i >= L_iu + margin, multiplier projected to >= 0
        self.lam_lat = np.zeros((self.n, 3)); self.lam_ax = np.zeros(self.n)
        self.X0 = c["X0"]; self.X_prev = self.X0.copy()
        self.V0, _ = geom.hexa_volumes(self.X0, c["hexa"])
        self.signs, self.size = geom.hexa_corner_signs(self.X0, c["hexa"])
        sph = c["inp"]["app"]["spheres"]
        self.sph_c_app = np.array([s["center_app_mm"] for s in sph], float)
        self.sph_r = np.array([s["r_mm"] for s in sph], float)
        self.cs = c.get("contact_surface"); self.cs_lin = None; self.cs_n = 0     # stage-2 surface contact (pyff)
        self.k = 0; self.hold = 0; self.done = False; self.status = "running"
        self.rows = []; self.override = None; self.dx_grow = 0; self.last_dx = None
        self.dx_hist = []; self.ok_hist = []; self.fres_prev = None; self.drift_est = None; self.rho = None
        self.anchor = c.get("anchor")
        if self.anchor is not None:
            self.X0s = self.X0[c["ns"]["sup_nodes"]]
        self.t_step0 = None
        self._log = None
        if self.out_dir:
            self._log = open(os.path.join(self.out_dir, "log.jsonl"), "w")

    # ------------------------------------------------------------------ pose / RL hook
    def current_pose(self):
        if self.override is not None:
            return self.override
        if self.k < len(self.sched):
            return self.sched[self.k]
        p = dict(self.sched[-1]); p["phase"] = "H"
        return p

    def step_action(self, d_depth_mm, d_angles_deg):
        """M5 hook (not used by the M1/M2 protocol): advance the applicator by d_depth_mm along its axis and tilt
        it by (rx, ry) deg about its own x/y axes (pivot at the flange).  Subsequent steps use this pose."""
        p = dict(self.current_pose()); R = geom.pose_frame(p)
        Rx = _rot_about(R[0], np.radians(d_angles_deg[0])); Ry = _rot_about(R[1], np.radians(d_angles_deg[1]))
        a = geom.unit(Ry @ Rx @ p["a"]); x = geom.ortho(Ry @ Rx @ p["x"], a)
        F = np.asarray(p["F"]) + float(d_depth_mm) * a
        self.override = dict(phase="RL", u=p.get("u", 1.0), F=F, a=a, x=x, tip=F + self.L * a, r_scale=p.get("r_scale", 0.0))

    # ------------------------------------------------------------------ helpers
    def X(self):
        return np.array(self.ctx["dofs"].position.value, dtype=float, copy=True)

    def canal_pos(self, X):
        """Current position of the TIE points (canal points, or per-cell groups of them)."""
        return np.einsum("ij,ijk->ik", self.ctx["W"], X[self.ctx["cnodes"]]) + self.ctx["c_off"]

    def canal_full(self, X):
        """Current position of every canal point (1 mm spacing; read-out only)."""
        c = self.ctx
        return np.einsum("ij,ijk->ik", c["W_full"], X[c["cnodes_full"]]) + c["c_off_full"]

    def cg_iterations(self):
        """CG iterations of the last linear solve, parsed from CGLinearSolver 'graph' ("Error e0 e1 .. Denominator
        d1 d2 ..."; the physics review's method).  None if not CG or unreadable."""
        ls = self.ctx.get("ls")
        if ls is None or self.cfg["linear_solver"] != "CGLinearSolver":
            return None
        try:
            ser, cur = {}, None
            for tok in ls.findData("graph").getValueString().split():
                try:
                    float(tok); ser[cur] = ser.get(cur, 0) + 1
                except ValueError:
                    cur = tok
            if "Denominator" in ser:
                return int(ser["Denominator"])
            return int(ser.get("Error", 1) - 1)
        except Exception:
            return None

    def _elem_loc(self, X, j, F, a):
        """Where hexa j sits relative to the rod (axial s, radial r, mm), and how many of its nodes are boundary."""
        cc = X[self.ctx["hexa"][j]].mean(0) - F; s = float(cc @ a)
        nb = int(np.isin(self.ctx["hexa"][j], self.ctx["ns"]["boundary"]).sum())
        return dict(hexa=j, s_mm=round(s, 2), r_mm=round(float(np.linalg.norm(cc - s * a)), 2), n_boundary_nodes=nb)

    def anchor_positions(self, w):
        """Support anchors for the device-rigid anchoring at progress w in [0, 1]."""
        an = self.anchor
        R = _rot_about(an["axis"], np.radians(an["angle_deg"] * w))
        return an["pivot"] + (self.X0s - an["pivot"]) @ R.T + w * an["shift"]

    def desired(self, c, F, a):
        s = (c - F) @ a
        s_des = np.where(self.is_tip, np.maximum(s, self.L + self.cfg["tip_push_margin_mm"]), s)
        return F + np.outer(s_des, a), s

    def spheres_world(self, pose, scale):
        R = geom.pose_frame(pose)
        return np.asarray(pose["F"]) + self.sph_c_app @ R, self.sph_r * scale

    # ------------------------------------------------------------------ step begin
    def onAnimateBeginEvent(self, ev):
        try:
            self._begin(ev)
        except Exception:
            import traceback
            traceback.print_exc(); raise

    def onAnimateEndEvent(self, ev):
        try:
            self._end(ev)
        except Exception:
            import traceback
            traceback.print_exc(); raise

    def _begin(self, ev):
        t0 = time.perf_counter()
        c = self.ctx; cfg = self.cfg
        pose = self.current_pose(); self.pose = pose
        F = np.asarray(pose["F"], float); a = geom.unit(pose["a"])
        X = self.X(); self.X_begin = X
        cp = self.canal_pos(X)
        s = (cp - F) @ a
        ph = pose["phase"]
        if ph == "H" and self.hold == 0 and c.get("ode") is not None:
            # quasi-static hold: full Newton solve per step (insertion uses cfg newton_iterations, default 1)
            c["ode"].newton_iterations.value = int(cfg.get("hold_newton_iterations", cfg["newton_iterations"]))
            if cfg.get("hold_cg_iterations") and cfg["linear_solver"] == "CGLinearSolver":
                c["ls"].iterations.value = int(cfg["hold_cg_iterations"])
                c["ls"].tolerance.value = float(cfg["hold_cg_tolerance"])
        if self.anchor is not None:                    # device-rigid support anchors follow the insertion progress
            w = 0.0 if ph == "A" else (float(pose.get("u", 1.0)) if ph == "T" else 1.0)
            c["found_tgt"].position.value = self.anchor_positions(w).tolist()
        if ph not in ("A",):
            lim = np.where(self.is_tip, self.L + cfg["tip_engage_tol_mm"], self.L)
            new = (~self.engaged) & (s <= lim) & (s >= -cfg["shaft_len_mm"])
            self.engaged |= new; self.eng_step[new] = self.k
            self.t_prev[new] = cp[new]
        if ph == "D" and not self.released.any():
            self.released = self.engaged & (~self.is_tip) & (s < 0)     # device body occupies the shaft space
        scale_now = float(pose.get("r_scale", 0.0))
        if cfg["m2"] and cfg.get("m2_mode", "inflate") == "travel" and scale_now > 0:
            # travel: a canal point inside an ovoid/cap sphere is released for good (the device occupies that space)
            cc, rr = self.spheres_world(pose, scale_now)
            occ = geom.sphere_sdf(cp, cc, rr).min(1) < 0
            self.released |= occ & (~self.is_tip)
        active = self.engaged & ~self.released
        # ---- capped target motion, expressed in the CURRENT rod frame (F, a)
        q = self.t_prev - F; s_prev = q @ a; lat_prev = q - np.outer(s_prev, a)
        ln = np.linalg.norm(lat_prev, axis=1)
        lat_new = lat_prev * np.clip(1.0 - cfg["lateral_cap_mm_per_step"] / np.maximum(ln, 1e-12), 0.0, 1.0)[:, None]
        s_push = self.L + cfg["tip_push_margin_mm"]
        # tip push is UNILATERAL: the target never sits behind the tissue point (no pull-back), and it advances at most
        # axial_cap per step toward s_push; lateral ties carry no axial restraint (frictionless sliding: s_tgt = s).
        s_tgt = np.where(self.is_tip, np.maximum(s, np.minimum(s_push, s_prev + cfg["axial_cap_mm_per_step"])), s)
        t_base = F + np.outer(s_tgt, a) + lat_new
        self.t_prev[active] = t_base[active]
        self.cap_done = (np.linalg.norm(lat_new, axis=1) < 1e-9) & ((~self.is_tip) | (s_tgt >= s_push - 1e-9) | (s >= s_push))
        t = t_base.copy()
        al_on = cfg["tie_al"] == "always" or (cfg["tie_al"] == "hold" and ph == "H")
        if al_on:
            lam_l = self.lam_lat - np.outer(self.lam_lat @ a, a)          # keep the lateral multiplier normal to the rod
            t = t + lam_l + np.outer(self.lam_ax, a)
        ramp = np.clip((self.k - self.eng_step + 1) / float(cfg["tie_ramp_steps"]), 0.0, 1.0)
        kscale = c["grp_n"] if cfg.get("tie_group_k", "sum") == "sum" else 1.0   # a cell tie carries its members' stiffness
        k_i = np.where(active, cfg["k_tie_mN_per_mm"] * ramp * kscale, 0.0)
        self.t_eff, self.k_i, self.s_begin = t, k_i, s
        # per-point projector: lateral ties act only normal to the rod (frictionless), the tip push acts in 3D.
        # With the isotropic RSSFF implementation the projection is applied to the target offset only.
        Plat = np.eye(3) - np.outer(a, a)
        self.Pi = np.where(self.is_tip[:, None, None], np.eye(3)[None], Plat[None])
        self.n_null = 0
        if ph == "A" and cfg.get("phaseA_null_ties", False):
            # null test of the coupling: every tie point tied in 3-D with full stiffness to its CURRENT position (zero
            # force, non-zero stiffness); the body must not move (umax < 1e-3 mm is checked by run_insertion)
            t = cp.copy(); k_i = np.full(self.n, float(cfg["k_tie_mN_per_mm"])); self.Pi = np.repeat(np.eye(3)[None], self.n, 0)
            self.t_eff, self.k_i = t, k_i; self.n_null = self.n
        # ---- stage-2 ovoid/cap contact on the TRUE body-surface vertices (frictionless, smooth sphere-union SDF):
        # vertex v at c_v = sum_j w_vj x_j; if phi(c_v) < 0 the penalty force k (-phi) n acts along the SDF normal
        # only (tangential sliding free); its linearisation k n n^T is lumped onto the 8 cell nodes like the ties.
        self.cs_lin = None; self.cs_n = 0
        if self.cs is not None and float(pose.get("r_scale", 0.0)) > 0:
            cs = self.cs
            cv = np.einsum("ij,ijk->ik", cs["W"], X[cs["nodes"]]) + cs["off"]
            cc, rr = self.spheres_world(pose, float(pose["r_scale"]))
            phi, nrm = geom.smooth_union_sdf(cv, cc, rr, float(cfg["contact_blend_mm"]))
            ins = phi < 0
            if ins.any():
                ks = float(cfg["k_s_mN_per_mm"]); dep = np.minimum(-phi[ins], float(cfg["contact_depth_cap_mm"]))
                nv = nrm[ins]
                fv = ks * dep[:, None] * nv; Kv = ks * nv[:, :, None] * nv[:, None, :]
                self.cs_lin = (cs["nodes"][ins], cs["W"][ins], fv, Kv)
                self.cs_f0, self.cs_K, self.cs_idx, self.cs_c0 = fv, Kv, np.nonzero(ins)[0], cv[ins]
            self.cs_n = int(ins.sum())
        self.write_dnt(X, cp, t, k_i)
        # ---- M2 sphere-SDF contact on candidate nodes (RSSFF 'con', LDL-safe)
        self.n_contacts = self.cs_n
        self.con_k = None
        if c["con"] is not None:
            scale = float(pose.get("r_scale", 0.0))
            if scale > 0:
                cc, rr = self.spheres_world(pose, scale)
                Xc = X[c["cand"]]
                phi = geom.sphere_sdf(Xc, cc, rr); kk = phi.argmin(1); pm = phi[np.arange(len(Xc)), kk]
                inside = pm < 0
                tg = Xc.copy()
                if inside.any():
                    # nearest exit from the sphere UNION: project onto every sphere surface and keep the closest
                    # projection that lies outside all spheres (projecting onto the deepest sphere alone can land
                    # inside a neighbouring, overlapping sphere and makes the contact chatter); fallback = deepest.
                    xi = Xc[inside]
                    v = xi[:, None, :] - cc[None]; nv = np.maximum(np.linalg.norm(v, axis=2), 1e-12)
                    cand_p = cc[None] + v / nv[..., None] * rr[None, :, None]                 # (m, K, 3)
                    phic = np.linalg.norm(cand_p[:, :, None, :] - cc[None, None], axis=3) - rr[None, None]
                    okc = phic.min(2) >= -1e-6
                    dst = np.where(okc, np.linalg.norm(cand_p - xi[:, None], axis=2), np.inf)
                    kb = dst.argmin(1); nofree = ~np.isfinite(dst.min(1)); kb[nofree] = kk[inside][nofree]
                    proj = cand_p[np.arange(len(xi)), kb]
                    tg[inside] = xi + cfg["contact_omega"] * (proj - xi)
                kc = np.where(inside, cfg["k_s_mN_per_mm"], 0.0)
                c["con_tgt"].position.value = tg.tolist()
                c["con"].stiffness.value = kc.tolist()
                self.n_contacts = int(inside.sum()); self.con_k, self.con_tg = kc, tg
            else:
                c["con"].stiffness.value = [0.0] * len(c["cand"])
        self.t_ctrl_ms = 1000 * (time.perf_counter() - t0)
        self.t_solve0 = time.perf_counter()

    def write_dnt(self, X, cp, t, k_i):
        c = self.ctx
        W = c["W"]; li = c["loc"][c["cnodes"]]                        # (n,8) local tie-node index
        kw = k_i[:, None] * W
        if c.get("tie_impl") == "pyff":
            nn = len(X); cn = c["cnodes"]                                                  # global node ids (n, M)
            fi = k_i[:, None] * np.einsum("nij,nj->ni", self.Pi, t - cp)                  # projected point force
            Kb = np.zeros((nn, 3, 3)); np.add.at(Kb, cn.ravel(), (kw[:, :, None, None] * self.Pi[:, None]).reshape(-1, 3, 3))
            F0 = np.zeros((nn, 3)); np.add.at(F0, cn.ravel(), (W[:, :, None] * fi[:, None, :]).reshape(-1, 3))
            if self.cs_lin is not None:                                                    # + surface contact
                sn, sW, fv, Kv = self.cs_lin
                np.add.at(Kb, sn.ravel(), (sW[:, :, None, None] * Kv[:, None]).reshape(-1, 3, 3))
                np.add.at(F0, sn.ravel(), (sW[:, :, None] * fv[:, None, :]).reshape(-1, 3))
            act = np.nonzero((np.abs(Kb).reshape(nn, -1).max(1) > 0) | (np.abs(F0).max(1) > 0))[0]
            xr = X[act]
            c["tie"].set_linearisation(act, xr, F0[act], Kb[act])
            self.Kb, self.F0, self.xref, self.lin_nodes = Kb[act], F0[act], xr, act
            self.kj, self.tgt_nodes = None, None
            return
        kj = np.zeros(len(c["tie_nodes"])); np.add.at(kj, li.ravel(), kw.ravel())
        num = np.zeros((len(c["tie_nodes"]), 3))
        np.add.at(num, li.ravel(), (kw[:, :, None] * (t - cp)[:, None, :]).reshape(-1, 3))
        Xt = X[c["tie_nodes"]]
        tgt = Xt + np.where(kj[:, None] > 0, num / np.maximum(kj, 1e-300)[:, None], 0.0)
        c["tie_tgt"].position.value = tgt.tolist()
        c["tie"].stiffness.value = kj.tolist()
        self.kj, self.tgt_nodes = kj, tgt

    # ------------------------------------------------------------------ step end
    def _end(self, ev):
        solve_ms = 1000 * (time.perf_counter() - self.t_solve0)
        c = self.ctx; cfg = self.cfg; pose = self.pose; ph = pose["phase"]
        F = np.asarray(pose["F"], float); a = geom.unit(pose["a"])
        X = self.X()
        finite = bool(np.all(np.isfinite(X)))
        cp = self.canal_pos(X)
        des, s = self.desired(cp, F, a)
        active = self.engaged & ~self.released
        err = np.linalg.norm(cp - des, axis=1)
        lam_prev = (self.lam_lat.copy(), self.lam_ax.copy())
        frozen = (cfg.get("hold_rule", "v1") == "v2" and ph == "H"
                  and self.hold >= int(cfg.get("al_freeze_after_hold_steps", 10 ** 9)))
        if (cfg["tie_al"] == "always" or (cfg["tie_al"] == "hold" and ph == "H")) and not frozen:
            # augmented-Lagrangian update of the target corrections (NUMERICAL device: removes the penalty error
            # F/k at convergence).  Only where the capped target has arrived, and at most tie_al_step_mm per step so
            # the multiplier never kicks the tissue (the old unlimited update jumped up to 6 mm at hold entry).
            stp = float(cfg.get("tie_al_step_mm", 0.5)); om = float(cfg.get("tie_al_relax", 1.0))
            upd = active & self.cap_done
            e = om * (des - cp); e_ax = e @ a; e_lat = e - np.outer(e_ax, a)
            el = np.linalg.norm(e_lat, axis=1, keepdims=True)
            e_lat = e_lat * np.minimum(1.0, stp / np.maximum(el, 1e-12))
            self.lam_lat[upd] += e_lat[upd]
            nl = np.linalg.norm(self.lam_lat, axis=1, keepdims=True)
            self.lam_lat *= np.minimum(1.0, cfg["tie_al_clip_mm"] / np.maximum(nl, 1e-12))
            tipa = upd & self.is_tip
            if cfg.get("tie_al_tip", True):
                g = np.clip(om * ((self.L + cfg["tip_push_margin_mm"]) - s), -stp, stp)   # > 0: canal end behind the tip
                self.lam_ax[tipa] = np.clip(self.lam_ax[tipa] + g[tipa], 0.0, cfg["tie_al_clip_mm"])
        # forces applied by the device on the tissue (mN), from the node springs after the solve
        if self.kj is None:     # pyff
            fnode = self.F0 - np.einsum("nij,nj->ni", self.Kb, X[self.lin_nodes] - self.xref)   # ties + contact
        else:
            fnode = self.kj[:, None] * (self.tgt_nodes - X[c["tie_nodes"]])
        f_i = self.k_i[:, None] * np.einsum("nij,nj->ni", self.Pi, self.t_eff - cp)   # per canal point (exact at convergence)
        tip = self.is_tip & active; lat_m = active & ~self.is_tip
        f_ax = float((f_i[tip] @ a).sum()) if tip.any() else 0.0
        f_lat_vec = f_i[lat_m] - np.outer(f_i[lat_m] @ a, a) if lat_m.any() else np.zeros((0, 3))
        mom = np.cross(cp[active] - F, f_i[active]).sum(0) if active.any() else np.zeros(3)
        u = np.linalg.norm(X - self.X0, axis=1)
        dX = X - self.X_prev; dxn = np.linalg.norm(dX, axis=1); jm = int(dxn.argmax()); dx = float(dxn[jm])
        rj = X[jm] - F; sj = float(rj @ a)
        dx_diag = dict(node=jm, s_mm=round(sj, 2), r_mm=round(float(np.linalg.norm(rj - sj * a)), 2),
                       axial_frac=round(float(abs(dX[jm] @ a) / max(dx, 1e-12)), 3),
                       to_tip_mm=round(float(np.linalg.norm(X[jm] - np.asarray(pose["tip"], float))), 2),
                       n_over_tol=int((dxn > cfg["hold_dx_tol_mm"]).sum()))
        V, tv = geom.hexa_volumes(X, c["hexa"])
        vr = V / self.V0; tvr = tv.min(1) / (self.V0 / 5.0)
        lam_max = geom.principal_stretches(geom.hexa_center_F(X, self.X0, c["hexa"], self.signs, self.size))[:, -1]
        pen = 0.0; f_ov_ax = 0.0; f_ov_lat = 0.0
        if c["con"] is not None and float(pose.get("r_scale", 0.0)) > 0:
            cc, rr = self.spheres_world(pose, float(pose["r_scale"]))
            pen = float(max(0.0, -geom.sphere_sdf(X[c["cand"]], cc, rr).min()))
            if self.con_k is not None:
                fc = (self.con_k[:, None] * (self.con_tg - X[c["cand"]])).sum(0)   # net ovoid force on tissue (mN)
                f_ov_ax = float(fc @ a); f_ov_lat = float(np.linalg.norm(fc - f_ov_ax * a))
        if self.cs is not None and float(pose.get("r_scale", 0.0)) > 0:
            cs = self.cs; cv = np.einsum("ij,ijk->ik", cs["W"], X[cs["nodes"]]) + cs["off"]
            cc, rr = self.spheres_world(pose, float(pose["r_scale"]))
            phi_e, _ = geom.smooth_union_sdf(cv, cc, rr, float(cfg["contact_blend_mm"]))
            pen = float(max(0.0, -phi_e.min()))                         # surface-vertex penetration after the solve
            if self.cs_lin is not None:                                   # linearised contact force at the end state
                fe = self.cs_f0 - np.einsum("nij,nj->ni", self.cs_K, cv[self.cs_idx] - self.cs_c0)
                fc = fe.sum(0); f_ov_ax = float(fc @ a); f_ov_lat = float(np.linalg.norm(fc - f_ov_ax * a))
        e_act = err[active]
        cg_its = self.cg_iterations()                                   # CG iterations of the LAST Newton solve
        try:                                                            # nodal force residual (mN) at the last Newton
            fr = np.linalg.norm(np.asarray(c["dofs"].force.value, float), axis=1)   # iterate (before its correction)
            fres = float(fr.max()); fres_mean = float(fr.mean())
        except Exception:
            fres = fres_mean = None
        dlam = float(max(np.abs(self.lam_lat - lam_prev[0]).max(initial=0.0), np.abs(self.lam_ax - lam_prev[1]).max(initial=0.0)))
        al_lam_max = float(max(np.linalg.norm(self.lam_lat, axis=1).max(initial=0.0), self.lam_ax.max(initial=0.0)))
        s_end = float(s[-1])
        row = dict(step=self.k, phase=ph, u=round(float(pose.get("u", 0.0)), 5),
                   solve_ms=round(solve_ms, 2), ctrl_ms=round(self.t_ctrl_ms, 2), cg_its=cg_its,
                   cg_cap=int(c["ls"].iterations.value) if cg_its is not None else None,
                   f_res_max_mN=fres, f_res_mean_mN=fres_mean, al_dlam_mm=dlam, al_frozen=bool(frozen),
                   lam_max_mm=al_lam_max, n_lam_sat=int((np.linalg.norm(self.lam_lat, axis=1) >= float(cfg.get("lam_sat_frac", 0.9)) * cfg["tie_al_clip_mm"]).sum()),
                   canal_end_s_mm=s_end, rod_beyond_canal_end_mm=float(self.L - s_end), n_null_ties=int(self.n_null),
                   umax=float(u.max()), umean=float(u.mean()), dx_max=dx, dx_diag=dx_diag,
                   tie_err_mean=float(e_act.mean()) if len(e_act) else 0.0,
                   tie_err_max=float(e_act.max()) if len(e_act) else 0.0,
                   tie_err_lat_max=float(err[lat_m].max()) if lat_m.any() else 0.0,
                   tie_err_tip_max=float(err[tip].max()) if tip.any() else 0.0,
                   n_engaged=int(active.sum()), n_released=int(self.released.sum()), n_contacts=int(self.n_contacts), pen_max=pen,
                   F_ovoid_axial_mN=f_ov_ax, F_ovoid_lat_mN=f_ov_lat,
                   F_axial_mN=f_ax, F_lat_sum_mN=float(np.linalg.norm(f_lat_vec, axis=1).sum()),
                   F_lat_net_mN=float(np.linalg.norm(f_lat_vec.sum(0))) if len(f_lat_vec) else 0.0,
                   F_node_net_mN=float(np.linalg.norm(fnode.sum(0))),
                   M_F_mNmm=float(np.linalg.norm(mom)),
                   min_vol_ratio=float(vr.min()), min_tet_ratio=float(tvr.min()), max_stretch=float(lam_max.max()),
                   max_vol_ratio=float(vr.max()), n_vol_out_0p85_1p15=int(((vr < 0.85) | (vr > 1.15)).sum()),
                   n_neg_corner_tet=int((tvr < 0).sum()),
                   minV_at=self._elem_loc(X, int(vr.argmin()), F, a),
                   tip=np.round(np.asarray(pose["tip"], float), 3).tolist(), finite=finite)
        self.rows.append(row)
        self.X_prev = X
        # ---- stop / abort logic
        if not finite:
            self.done, self.status = True, "abort_nan"
        elif vr.min() < cfg["min_vol_ratio_abort"]:
            self.done, self.status = True, "abort_volume_ratio"
        elif ph == "H":
            self.hold += 1
            v2 = cfg.get("hold_rule", "v1") == "v2"
            fgrow = (fres is not None and self.fres_prev is not None and fres > self.fres_prev
                     and fres > float(cfg.get("hold_force_tol_mN", 1.0)))
            if self.last_dx is not None and dx > self.last_dx and dx > cfg["hold_dx_tol_mm"] and (fgrow or not v2):
                self.dx_grow += 1                        # v2: displacement AND force residual growing together
            else:
                self.dx_grow = 0
            self.fres_prev = fres
            ok = (dx < cfg["hold_dx_tol_mm"] and row["tie_err_mean"] < cfg["tie_err_mean_tol_mm"]
                  and row["tie_err_max"] < cfg["tie_err_max_tol_mm"] and pen < cfg["pen_tol_mm"])
            if v2:
                ok = ok and (fres is None or fres < float(cfg["hold_force_tol_mN"])) and dlam < float(cfg["al_dlam_tol_mm"])
                self.dx_hist.append(dx); self.ok_hist.append(bool(ok))
                n = int(cfg["hold_consec_steps"])
                if len(self.dx_hist) >= n + 1:
                    d = np.array(self.dx_hist[-(n + 1):])
                    self.rho = float(np.median(d[1:] / np.maximum(d[:-1], 1e-12)))
                    self.drift_est = float(dx * self.rho / (1 - self.rho)) if self.rho < 1 else float("inf")
                row.update(hold_ok=bool(ok), dx_ratio=self.rho, drift_est_mm=self.drift_est)
                conv = (len(self.ok_hist) >= n and all(self.ok_hist[-n:]) and self.drift_est is not None
                        and self.drift_est < float(cfg["hold_drift_tol_mm"]))
            else:
                conv = ok and self.hold >= 2
            if conv:
                self.done, self.status = True, "converged"
            elif self.dx_grow > cfg["residual_growth_abort_steps"]:
                self.done, self.status = True, "abort_residual_growth"
            elif self.hold >= cfg["hold_max_steps"]:
                self.done, self.status = True, "hold_not_converged"
        if self._log:
            self._log.write(json.dumps(row) + "\n"); self._log.flush()
        self.last_dx = dx if ph == "H" else None
        self.k += 1

    def close(self):
        if self._log:
            self._log.close(); self._log = None
