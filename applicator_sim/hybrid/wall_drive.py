"""S7b of the G32 fix plan: the DEVICE-DRIVEN opening of the v5 vaginal wall.  numpy only (the SOFA container's
python 3.8 and the host); no SOFA here.  Unit tests: hybrid/test_wall_drive.py (synthetic geometry).

WHAT IT DOES.  The v5 wall (vagina_wall_tet.py tet26v5 / tet26v5p) has two shapes over the same tets: the REST shape
(tets.vtk: stress-free, seated around applicator_v5 at device_final) and the collapsed START (start.vtk: the preBT label
below the HR-CTV, a slit lumen, the top ring ON the cervix surface).  The drive moves every wall node kinematically,
step by step, from the start towards the rest shape, and ONLY as the device parts arrive (fix plan S7b):

  floor     the collapsed section, carried: each node keeps its start in-plane offset q_s from its start slab centre
            C_s (the label's centre curve; slab planes normal to the vaginal axis a) and its start height, and
            - STRAIGHTENING: the slab centre moves onto the device line D(h) where the tandem body occupies the slab,
              C = C_s + omega (D - C_s), omega = smoothstep of the tip's travel past the slab (straight_lead_mm before
              the tip arrives to straight_ramp_mm after), monotone;
            - VAULT ON THE PORTIO: the top ring (node set vault_top) is glued to the cervix surface points it starts on
              (meta wall.v5.pairing: cervix triangle + barycentric weights + the 0.2 mm start offset); the nodes within
              glue_blend_mm (start arclength) below the top follow the top ring's displacement with a weight falling
              from 1 to 0 -- the vagina ends at the cervix (the physician's comment).  This holds through V, C, K1 and
              R only; what happens in K2 depends on k2_vault (below).  Glued
              nodes are not straightened (the portio carries them), and a glued node (g >= glue_push_g) is not pushed
              by the tandem where the rod at its height lies on the CERVIX side of its paired portio triangle (the rod
              is then in the tissue, not in the lumen; MEASURED on TF1v's cervix: in C the rod crosses the top ring's
              slab 6-10 mm from the junction slit);
            - K1 (lift): heights move from the start heights to the rest heights with the lift weight.
  need      the lumen must contain every device part + clear_mm: a lumen node is pushed outward along its own ray from
            the current slab centre to the far side of the device's section (tandem body: the plane section of the
            dilated rod, an ellipse; a nose of lead_mm ahead of the tip opens the slit before the tip arrives; ring
            halves: a filleted half-disc + socket boss + rod, dilated by ring_clear_mm, swept ring_lead_mm ahead of the
            half -- at its full size (ring_lead_slope None, the default), or, with ring_lead_slope s, shrunk by s mm per
            mm of approach still to go, so the lumen opens only as fast as the rate allows ahead of the half's own
            surface (see ring_lead_slope); the glued vault top is not pushed by the halves: it rides the portio, which
            the ring pushes by contact); the outer sheet moves with its lumen partner.  Nothing else opens:
            R = max(need + c, floor).
  rate      the push-out changes by at most rate_mm per step (opening and closing): the slot closes behind the tandem
            and behind each half at <= rate_mm / step.
  K2        the packing front: from the vault downwards every node blends from the driven position to its K2 target
            (lambda = smoothstep((front - depth) / k2_blend_mm), depth = rest centreline arclength below the top).
            k2_vault "rest" (DEFAULT): the target is the REST shape and the glue is ignored, so at the end of K2 the wall
            IS the rest shape bit for bit -- and the vagina-cervix junction BREAKS wherever the live portio is not where
            the rest top ring expects it (MEASURED, TF2c_c / TF2p_b / TF2c_nu25_b end of K2: 18-19 of 49 top-ring nodes
            per sheet more than 3 mm off the cervix, up to 9 mm posteriorly, some 4.1 mm inside it; 85 wall nodes inside
            the cervix).  k2_vault "live": the target is the rest shape with its VAULT re-derived on the live cervix
            (vault_band / vault_pair / vault_live): at the first K2 row each top-ring generator (the rest polyline at
            the node's angle through the vault_band_mm below the top, extended vault_reach_mm beyond it) is followed up
            from the band's anchor station to where it first meets the cervix surface; that point becomes the node's
            new junction, a material point of the cervix (triangle + barycentric weights, offset vault_offset_mm along
            the outward normal), so the junction follows the portio from then on, through the settle too.  The band's
            nodes are re-spaced along their generators to end there (vagina ends where it meets the cervix, as the
            v5 start was built), plus the junction's drift since the pairing, falling to 0 at the anchor; below the
            band the target is the rest shape unchanged.  While a node blends, the top ring is kept ON the cervix
            surface (closest point + offset) and any other node found inside the cervix is put back on its surface.

Every quantity above is derived at run time from the wall meta, the applicator json and the cervix state: no patient
constant lives here."""
import numpy as np

DRIVE_DEFAULTS = dict(
    clear_mm=0.75,              # c: lumen clearance over the tandem body (plan S7b: c = 0.5-1 mm)
    rate_mm=0.7,                # per-step bound on each node's push-out change, opening and closing (plan S7b)
    speed_mm=0.7,               # per-step bound on each driven node's total motion (S7c(5)); None = off
    lead_mm=7.0,                # the tandem nose: the slit opens this far ahead of the tip (plan: >= 7 mm)
    straight_lead_mm=20.0,      # straightening of a slab starts when the tip is this far below it ...
    straight_ramp_mm=10.0,      # ... and completes over this much tip travel (<= 10 steps at <= 1 mm / step), i.e.
                                #   BEFORE the nose (lead_mm + r + c above the hemisphere centre) reaches the slab: the
                                #   push rays then start on the device line inside the slit, never beside it
    glue_blend_mm=8.0,          # nodes this far (start arclength) below the top ring follow the portio, weight 1 -> 0
    glue_push_g=0.5,            # glued nodes (weight >= this) are not pushed by a rod lying on the cervix side of ...
    glue_push_mm=0.5,           # ... their portio triangle by more than this
    cos_min=0.3,                # the rod's tilt to the slab normal is capped (section semi-axis <= r / cos_min)
    ring_clear_mm=1.0,          # clearance over each ring half (the rest vault = ring outline + 1 mm, S7a)
    ring_lead_mm=21.0,          # the lumen opens this far ahead of an approaching half (plan: 21-36 mm)
    ring_lead_samples=4,        # the half is tested at this many positions over its lead
    ring_ray_mm=45.0,           # ray length searched beyond a node's floor radius for the half's far side
    ring_ray_step_mm=0.5,
    k2_blend_mm=15.0,           # K2: the packing front's ramp length (rest centreline arclength)
    k2_vault="rest",            # K2's target at the vault (module docstring, K2): "rest" (DEFAULT, bit for bit as TF2) or
                                #   "live" (the junction re-derived on the live cervix at the first K2 row and followed
                                #   from then on; needs the cervix surface triangles, WallDrive(cervix_tri=...))
    vault_band_mm=15.0,         # "live": the vault band (rest centreline arclength below the top) re-spaced along its
                                #   generators; the station just below it is the anchor that stays at rest
    vault_reach_mm=25.0,        # "live": how far beyond the rest top a generator is extended to find the cervix
    vault_offset_mm=0.2,        # "live": the junction's offset off the cervix surface (the v5 start pairing: ~0.2 mm)
    ring_lead_slope=None,       # None (DEFAULT): each lead pose counts the half at its full size (TF2: the lumen opened to
                                #   ring width up to 41 mm ahead of the half, review of TF2c_c).  s > 0: a lead pose dd mm
                                #   further along the approach counts the half's exit radius minus s * dd -- the lumen
                                #   ahead of the half opens in a cone of slope s (mm of push-out per mm of approach) off
                                #   the half's own surface, i.e. no earlier than the rate limit needs: with s * (approach
                                #   step) <= rate_mm the push target never grows faster than the lumen may open once the
                                #   half is under way (TF1r2 crushed the rectum when a half outran it).  It does NOT keep
                                #   the half inside the sheet everywhere: MEASURED (TF3c/TF3p, code review) the opening
                                #   lags 7-14 mm on each half's first row and a transient containment gap of up to
                                #   9.4 mm occurs (TF3c step 396); TF3 slope 0.8, lead 28.
    ring_smooth_st=0,           # > 0: the ring halves' push-out is smoothed over the lumen grid before it is applied: a
    ring_smooth_th=0,           #   max filter over +/- ring_smooth_st stations and +/- ring_smooth_th angle cells (wrapped),
                                #   then the mean of that over the same window (>= the unsmoothed push everywhere, so the
                                #   half stays contained).  MEASURED why (TF1r2 renders): per-node rays that graze a half
                                #   get very different exits from their neighbours' -- the lumen became a comb of radial
                                #   fins.  0 / 0 (DEFAULT) = off, bit for bit.
)


def unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-12)


def smoothstep(x):
    x = np.clip(np.asarray(x, float), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


TF3_KEYS = ("k2_vault", "vault_band_mm", "vault_reach_mm", "vault_offset_mm", "ring_lead_slope")


def params_record(pr):
    """The drive parameters as the scene records them: every key, except the TF3 keys while at their defaults (so a
    TF2 cfg's scene summary stays bit for bit what it was)."""
    return {k: v for k, v in pr.items() if k not in TF3_KEYS or v != DRIVE_DEFAULTS[k]}


def drive_cfg(user=None):
    """DRIVE_DEFAULTS updated by a dict (cfg wall_drive_params); unknown keys are refused."""
    out = dict(DRIVE_DEFAULTS)
    for k, v in (user or {}).items():
        if k not in DRIVE_DEFAULTS:
            raise ValueError("wall_drive_params: unknown key %r (have %s)" % (k, sorted(DRIVE_DEFAULTS)))
        out[k] = v
    if out["k2_vault"] not in ("rest", "live"):
        raise ValueError("wall_drive_params k2_vault must be 'rest' or 'live' (got %r)" % (out["k2_vault"],))
    if out["ring_lead_slope"] is not None and not float(out["ring_lead_slope"]) > 0.0:
        raise ValueError("wall_drive_params ring_lead_slope must be null or > 0 (got %r)" % (out["ring_lead_slope"],))
    if out["k2_vault"] == "live" and not float(out["vault_band_mm"]) > 0.0:
        raise ValueError("wall_drive_params vault_band_mm must be > 0")
    return out


# ============================================================================================ per-node tables
def periodic_weights(th_q, th_ref):
    """Linear interpolation weights on a periodic angle grid: for each query angle, (j0, j1, w) with value =
    (1 - w) v[j0] + w v[j1]; th_ref need not be sorted (indices refer to its own order)."""
    th_ref = np.mod(np.asarray(th_ref, float), 2.0 * np.pi)
    o = np.argsort(th_ref, kind="stable")
    ts = th_ref[o]
    tq = np.mod(np.asarray(th_q, float), 2.0 * np.pi)
    n = len(ts)
    j1 = np.searchsorted(ts, tq, side="right") % n
    j0 = (j1 - 1) % n
    t0, t1 = ts[j0], ts[j1]
    span = np.mod(t1 - t0, 2.0 * np.pi)
    span = np.where(span <= 0.0, 2.0 * np.pi, span)
    w = np.clip(np.mod(tq - t0, 2.0 * np.pi) / span, 0.0, 1.0)
    return o[j0], o[j1], w


def node_tables(meta, X_rest, X_start):
    """Everything the drive needs per node, from the wall meta (meta.json of a v5 wall) and its two shapes.
    X_rest = tets.vtk (the scene's X0), X_start = start.vtk (same node order)."""
    w = meta["wall"]
    v5 = w.get("v5")
    if not v5:
        raise ValueError("wall_drive needs a v5 wall (meta wall.v5: lumen_profile 'packed', vagina_wall_tet26v5*)")
    X_rest, X_start = np.asarray(X_rest, float), np.asarray(X_start, float)
    n = len(X_rest)
    if X_start.shape != X_rest.shape:
        raise ValueError("start.vtk (%s) and tets.vtk (%s) differ in shape" % (X_start.shape, X_rest.shape))
    a = unit(np.asarray(meta["axis"]["axis"], float))
    c0 = np.asarray(meta["axis"]["centroid"], float)
    sheet = np.asarray(v5["node_sheet"], int)
    station = np.asarray(v5["node_station"], int)
    theta = np.asarray(v5["node_theta_rad"], float)
    sigma = np.asarray(v5["node_start_sigma_mm"], float)
    f = np.asarray(v5["node_f"], float)
    st = v5["stations"]["start"]
    lab_sig = np.asarray(st["label_sigma_mm"], float)
    lab_c = np.asarray(st["label_centre"], float)
    Cs = np.stack([np.interp(sigma, lab_sig, lab_c[:, j]) for j in range(3)], 1)
    q = X_start - Cs
    q = q - np.outer(q @ a, a)
    Cs = X_start - q                                    # exact: the start node = its slab centre + in-plane offset
    rho_s = np.linalg.norm(q, axis=1)
    E = unit(q)
    hs = (X_start - c0) @ a
    hr = (X_rest - c0) @ a
    gi = np.asarray(w["grid_index"], int)
    inner = np.nonzero(sheet == 0)[0]
    key = {(int(gi[i, 0]), int(gi[i, 2])): int(i) for i in inner}
    partner = np.array([key[(int(gi[i, 0]), int(gi[i, 2]))] for i in range(n)], int)
    # start arclength below the top ring along the node's generator: sigma_top - sigma = sigma (1 / f - 1)
    depth = np.where(f > 1e-9, sigma * (1.0 / np.maximum(f, 1e-9) - 1.0), 1e9)
    ns_ = meta["node_sets"]
    top = np.asarray(ns_["vault_top"], int)
    pr = v5["pairing"]
    if [int(i) for i in pr["vault_top"]] != [int(i) for i in top]:
        raise ValueError("meta wall.v5.pairing.vault_top is not node set vault_top in the same order")
    top_sheet = sheet[top]
    glue_w = {}
    for sh in (0, 2):
        ref = top[top_sheet == sh]
        idx = np.nonzero(sheet == sh)[0]
        j0, j1, ww = periodic_weights(theta[idx], theta[ref])
        glue_w[sh] = dict(idx=idx, ref=np.nonzero(top_sheet == sh)[0], j0=j0, j1=j1, w=ww)
    s_rest = np.asarray(w["s"], float)
    depth_rest = float(s_rest[-1]) - s_rest[station]
    return dict(n=n, a=a, c0=c0, sheet=sheet, station=station, theta=theta, sigma=sigma, f=f, Cs=Cs, q=q,
                rho_s=rho_s, E=E, hs=hs, hr=hr, partner=partner, inner=inner, outer=np.nonzero(sheet != 0)[0],
                depth=depth, top=top, top_sheet=top_sheet, glue_w=glue_w,
                pair_tri=np.asarray(pr["cervix_tri_nodes"], int), pair_bary=np.asarray(pr["cervix_tri_bary"], float),
                pair_off=np.asarray(pr["offset_start_mm"], float), depth_rest=depth_rest,
                X_rest=X_rest, X_start=X_start, grid_index=gi)


# ============================================================================================ the device
def tandem_geometry(app):
    """The tandem body as a centreline in the APPLICATOR frame, ordered from the tip end down: the tube (radius
    r_tandem) from its hemisphere centre (L_iu - r) to the tube bottom, then the vaginal tandem (landmarks
    shaft_centreline below the tube bottom, radius r_shaft).  Everything read from applicator.json."""
    prm = app["params"]
    val = lambda k: float(prm[k]["value"] if isinstance(prm[k], dict) else prm[k])  # noqa: E731
    L, rt = val("L_iu_mm"), val("r_tandem_mm")
    rs = val("r_shaft_mm") if "r_shaft_mm" in prm else rt
    lm = app["landmarks"]
    zb = float(lm["tube_bottom"][2]) if lm.get("tube_bottom") is not None else 0.0
    zc = L - rt
    nz = max(2, int(np.ceil((zc - zb) / 1.0)) + 1)
    zt = np.linspace(zc, zb, nz)
    tube = np.c_[np.zeros(nz), np.zeros(nz), zt]
    parts = [tube]
    rad = [np.full(nz, rt)]
    scl = lm.get("shaft_centreline")
    if scl is not None:
        S = np.asarray(scl, float)
        if S[0, 2] < S[-1, 2]:
            S = S[::-1]
        S = S[S[:, 2] < zb]
        if len(S):
            parts.append(S)
            rad.append(np.full(len(S), rs))
    return dict(cl=np.vstack(parts), rad=np.concatenate(rad), zc=zc, r_tip=rt, L_iu=L, r_shaft=rs, z_tube_bottom=zb)


def ring_geometry(app):
    """The two ring halves (applicator_v5 ring block) as analytic shapes in the APPLICATOR frame of their SEAT:
    a half of a filleted disc (ring frame: T_seat origin / rows, faces z_top / z_bottom, fillets, outer radius, the
    medial slot), a socket boss and the rod (capsules).  None when the applicator has no ring."""
    ring = app.get("ring")
    lg = (app.get("label_geometry") or {}).get("ring_halves") or {}
    if not ring or not lg:
        return None
    ts = ring["T_seat"]
    halves = {}
    for side, h in ring["halves"].items():
        halves[side] = dict(part=h["part"], n_out=unit(np.asarray(h["n_out_app"], float)),
                            rod=np.asarray(h["rod_pts_app"], float), boss=np.asarray(h["boss"], float))
    ap = ring.get("approach") or {}
    for side, h in (ap.get("halves") or {}).items():
        if side in halves:
            halves[side].update(e=unit(np.asarray(h["e_app"], float)), n_out_ap=unit(np.asarray(h["n_out_app"], float)),
                                D=float(h["D_mm"]), offset=float(h["offset_mm"]), click=float(h["click_mm"]),
                                p=float(h["click_shape_p"]), order=int(h.get("order", 0)))
    return dict(origin=np.asarray(ts["origin_app"], float), R=np.asarray(ts["R_rows_app"], float),
                z_top=float(lg["z_top"]), z_bot=float(lg["z_bottom"]), f_top=float(lg["fillet_top"]),
                f_bot=float(lg["fillet_bottom"]), R_out=float(lg["R_out"]),
                slot=float(lg.get("slot", 0.5)) if not isinstance(lg.get("slot"), dict) else 0.5,
                r_boss=float(lg.get("r_boss", 4.0)), r_rod=float(lg.get("r_rod", 3.12)), halves=halves,
                order=[s for s, _ in sorted(((s, h.get("order", 0)) for s, h in halves.items()), key=lambda t: t[1])])


def approach_disp(h, d):
    """A ring half's displacement from its seat (applicator frame) at approach distance d (applicator.json
    ring.approach: disp(d) = d e + o(d) n_out, o = offset for d >= click, offset (1 - (1 - d / click)^p) below)."""
    d = float(d)
    if d >= h["click"]:
        o = h["offset"]
    else:
        o = h["offset"] * (1.0 - (1.0 - max(d, 0.0) / h["click"]) ** h["p"])
    return d * h["e"] + o * h["n_out_ap"]


def device_world(geo, F, R_rows):
    """The tandem centreline in world: p = F + p_app @ R_rows."""
    return np.asarray(F, float) + geo["cl"] @ np.asarray(R_rows, float)


def device_crossing(h, CLw, rad, at, a, c0, clear, lead):
    """Where the tandem body crosses the slab planes at heights h (along a from c0): the centre D, the local unit
    direction d, and the DILATED section radius rA (radius + clear), for each h.  Above the tube's hemisphere centre
    the line continues along the tube axis at, and the section is the NOSE: an ellipsoid of lateral semi-axis
    r_tip + clear and axial semi-axis r_tip + clear + lead (so the slit opens `lead` ahead of the tip).  Below the
    centreline's lowest point: ok False.  Returns D, d, rA, ok, s_nose (the axial distance above the hemisphere centre,
    <= 0 on the body)."""
    h = np.atleast_1d(np.asarray(h, float))
    CLw = np.asarray(CLw, float)
    hc = (CLw - c0) @ a
    n = len(h)
    D = np.zeros((n, 3))
    d = np.tile(np.asarray(at, float), (n, 1))
    rA = np.zeros(n)
    ok = np.zeros(n, bool)
    s_nose = np.zeros(n)
    h0, h1 = hc[:-1], hc[1:]
    lo, hi = np.minimum(h0, h1), np.maximum(h0, h1)
    M = (h[:, None] >= lo[None, :]) & (h[:, None] <= hi[None, :])
    has = M.any(1)
    j = np.argmax(M, axis=1)
    dh = h1[j] - h0[j]
    t = np.where(np.abs(dh) > 1e-12, (h - h0[j]) / np.where(np.abs(dh) > 1e-12, dh, 1.0), 0.0)
    Db = CLw[j] + t[:, None] * (CLw[j + 1] - CLw[j])
    db = unit(CLw[j + 1] - CLw[j])
    rb = rad[j] + t * (rad[j + 1] - rad[j]) + float(clear)
    D[has], d[has], rA[has], ok[has] = Db[has], db[has], rb[has], True
    # above the top of the centreline: the line continues along the tube axis (D is defined there for the straightening,
    # which completes BEFORE the nose arrives); the section is the nose, then nothing
    ca = float(np.asarray(at, float) @ a)
    up = (~has) & (h > hc[0])
    if up.any() and ca > 1e-6:
        s = (h[up] - hc[0]) / ca
        rn = float(rad[0]) + float(clear)
        ax = rn + float(lead)
        D[up] = CLw[0] + np.outer(s, at)
        rA[up] = rn * np.sqrt(np.maximum(0.0, 1.0 - (s / ax) ** 2))
        ok[up] = rA[up] > 1e-6
        s_nose[up] = s
    # below the lowest point: the last segment continued (D only; no section)
    dn = (~has) & (h < hc[-1])
    if dn.any():
        e = unit(CLw[-1] - CLw[-2])
        ce = float(e @ a)
        s = (h[dn] - hc[-1]) / (ce if abs(ce) > 1e-6 else -1e-6)
        D[dn] = CLw[-1] + np.outer(s, e)
        d[dn] = e
    return D, d, rA, ok, s_nose


def ellipse_exit(O, E, D, d, rA, a, cos_min):
    """The far exit radius along rays O + rho E (in the slab plane, normal a) of the plane section of a cylinder of
    radius rA about the line (D, d): an ellipse centred on D with semi-axes rA / cos(phi) along the in-plane projection
    of d and rA across it (phi = the tilt of d to a, cos capped at cos_min).  Returns rho2 (-inf where the ray misses)."""
    O, E, D, d = (np.atleast_2d(np.asarray(v, float)) for v in (O, E, D, d))
    rA = np.asarray(rA, float)
    a = np.asarray(a, float)
    dd = d - np.outer(d @ a, a)
    ca = np.maximum(np.abs(d @ a), float(cos_min))
    m = dd / np.maximum(np.linalg.norm(dd, axis=1, keepdims=True), 1e-12)
    small = np.linalg.norm(dd, axis=1) < 1e-9
    if small.any():                                     # rod along the slab normal: circular section, any in-plane m
        e0 = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        m[small] = unit(e0 - (e0 @ a) * a)
    nv = np.cross(a[None, :], m)
    gm = (ca / np.maximum(rA, 1e-9)) ** 2
    gn = 1.0 / np.maximum(rA, 1e-9) ** 2
    dl = O - D
    dl = dl - np.outer(dl @ a, a)
    Em, En = np.einsum("ij,ij->i", E, m), np.einsum("ij,ij->i", E, nv)
    Dm, Dn = np.einsum("ij,ij->i", dl, m), np.einsum("ij,ij->i", dl, nv)
    A = gm * Em ** 2 + gn * En ** 2
    B = 2.0 * (gm * Em * Dm + gn * En * Dn)
    C = gm * Dm ** 2 + gn * Dn ** 2 - 1.0
    disc = B * B - 4.0 * A * C
    hit = (disc > 0.0) & (A > 1e-15) & (rA > 1e-9)
    out = np.full(len(A), -np.inf)
    out[hit] = (-B[hit] + np.sqrt(disc[hit])) / (2.0 * A[hit])
    return out


# ---- ring halves: inside tests in the half's applicator frame
def _capsule_dist(P, pts):
    """Distance of points P to the polyline pts."""
    best = np.full(len(P), np.inf)
    for p0, p1 in zip(pts[:-1], pts[1:]):
        ab = p1 - p0
        l2 = float(ab @ ab)
        t = np.clip(((P - p0) @ ab) / max(l2, 1e-12), 0.0, 1.0)
        best = np.minimum(best, np.linalg.norm(P - (p0 + t[:, None] * ab), axis=1))
    return best


def half_inside(P_app, rg, side, clear):
    """Points (applicator frame of the half's own pose) inside the half dilated by `clear`: the half-disc (ring frame
    x_r * sign >= slot / 2 - clear, radius <= the filleted profile + clear, z_bot - clear <= z <= z_top + clear), the
    socket boss or the rod."""
    h = rg["halves"][side]
    P = np.atleast_2d(np.asarray(P_app, float))
    q = (P - rg["origin"]) @ rg["R"].T                  # ring frame (rows of R = x_r, y_r, z)
    z = P[:, 2]
    sgn = 1.0 if float(h["n_out"] @ rg["R"][0]) > 0 else -1.0
    zc = np.clip(z, rg["z_bot"], rg["z_top"])
    ft, fb, Ro = rg["f_top"], rg["f_bot"], rg["R_out"]
    r = np.full(len(P), Ro)
    tt = zc > rg["z_top"] - ft
    r[tt] = Ro - ft + np.sqrt(np.maximum(0.0, ft ** 2 - (zc[tt] - (rg["z_top"] - ft)) ** 2))
    bb = zc < rg["z_bot"] + fb
    r[bb] = np.minimum(r[bb], Ro - fb + np.sqrt(np.maximum(0.0, fb ** 2 - (zc[bb] - (rg["z_bot"] + fb)) ** 2)))
    rho = np.hypot(q[:, 0], q[:, 1])
    disc = (sgn * q[:, 0] >= 0.5 * rg["slot"] - clear) & (rho <= r + clear) & \
        (z >= rg["z_bot"] - clear) & (z <= rg["z_top"] + clear)
    boss = _capsule_dist(P, h["boss"]) <= rg["r_boss"] + clear
    rod = _capsule_dist(P, h["rod"]) <= rg["r_rod"] + clear
    return disc | boss | rod


def half_reach(rg, side):
    """A skeleton of the half (applicator frame) and a reach radius for the quick reject of far nodes."""
    h = rg["halves"][side]
    sk = np.vstack([rg["origin"][None, :], h["boss"], h["rod"]])
    return sk, rg["R_out"] + 6.0


def lead_poses(F, R_rows, h, d, pr):
    """The lead poses of a ring half at approach distance d on the tandem pose (F, R_rows): ring_lead_samples poses up to
    ring_lead_mm further along its approach (none once seated), each (origin, R_rows, dd), dd = how much further along
    the approach (mm; the seat caps it).  The one helper the scene, tf_metrics and tf_gate share."""
    ns = int(pr["ring_lead_samples"])
    d = float(d)
    out = []
    if d > 0.0 and ns > 0:
        for j in range(1, ns + 1):
            dj = max(0.0, d - float(pr["ring_lead_mm"]) * j / ns)
            org, R = half_pose(F, R_rows, h, dj)
            out.append((org, R, d - dj))
    return out


# ============================================================================================ cervix surface geometry
def orient_outward(V, T):
    """The triangles T (closed surface over V) ordered so that their normals point OUT (positive enclosed volume)."""
    V, T = np.asarray(V, float), np.asarray(T, int)
    vol = float(np.einsum("ij,ij->i", V[T[:, 0]], np.cross(V[T[:, 1]], V[T[:, 2]])).sum()) / 6.0
    return T if vol >= 0.0 else T[:, [0, 2, 1]].copy()


def tri_normals(V, T):
    V = np.asarray(V, float)
    return unit(np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]]))


def seg_first_hit(P0, P1, V, T, chunk=256):
    """Each segment P0[i] -> P1[i] against the triangles (V, T): the smallest parameter t in [0, 1] where it crosses
    one (inf where none), that triangle's index (-1) and the crossing's barycentric weights (n, 3).  Moller-Trumbore."""
    P0, P1, V = np.atleast_2d(np.asarray(P0, float)), np.atleast_2d(np.asarray(P1, float)), np.asarray(V, float)
    A = V[T[:, 0]]
    e1, e2 = V[T[:, 1]] - A, V[T[:, 2]] - A
    n = len(P0)
    tb, jb, bb = np.full(n, np.inf), np.full(n, -1, int), np.zeros((n, 3))
    for s in range(0, n, chunk):
        o, d = P0[s:s + chunk], P1[s:s + chunk] - P0[s:s + chunk]
        pv = np.cross(d[:, None, :], e2[None, :, :])
        det = np.einsum("mk,cmk->cm", e1, pv)
        ok = np.abs(det) > 1e-12
        inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
        tv = o[:, None, :] - A[None, :, :]
        u = np.einsum("cmk,cmk->cm", tv, pv) * inv
        qv = np.cross(tv, e1[None, :, :])
        v = np.einsum("ck,cmk->cm", d, qv) * inv
        t = np.einsum("mk,cmk->cm", e2, qv) * inv
        hit = ok & (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0) & (t >= 0.0) & (t <= 1.0)
        tt = np.where(hit, t, np.inf)
        j = np.argmin(tt, axis=1)
        r = np.arange(len(o))
        tb[s:s + chunk] = tt[r, j]
        h = np.isfinite(tt[r, j])
        jb[s:s + chunk] = np.where(h, j, -1)
        bb[s:s + chunk] = np.c_[1.0 - u[r, j] - v[r, j], u[r, j], v[r, j]] * h[:, None]
    return tb, jb, bb


def closest_on_tris(Q, V, T, chunk=128, k_near=None):
    """The closest point of the triangle mesh (V, T) to each point Q (Ericson's regions): point (n, 3), triangle index,
    barycentric weights (n, 3) and distance.  k_near: only the k_near triangles with the nearest centroids are tried per
    point (cheap; exact whenever the closest triangle is among them)."""
    Q, V = np.atleast_2d(np.asarray(Q, float)), np.asarray(V, float)
    T = np.asarray(T, int)
    A0 = V[T[:, 0]]
    AB0, AC0 = V[T[:, 1]] - A0, V[T[:, 2]] - A0
    n = len(Q)
    Pb, jb, bb, db = np.zeros((n, 3)), np.zeros(n, int), np.zeros((n, 3)), np.zeros(n)
    use_k = k_near is not None and int(k_near) < len(T)
    cen = A0 + (AB0 + AC0) / 3.0
    sdiv = lambda a_, b_: a_ / np.where(np.abs(b_) > 1e-18, b_, 1e-18)  # noqa: E731
    for s in range(0, n, chunk):
        p = Q[s:s + chunk][:, None, :]
        if use_k:                                       # (c, k) candidate triangles per point
            cand = np.argpartition(((p - cen[None]) ** 2).sum(-1), int(k_near) - 1, axis=1)[:, :int(k_near)]
            A, ab, ac = A0[cand], AB0[cand], AC0[cand]
        else:
            cand = None
            A, ab, ac = A0[None], AB0[None], AC0[None]
        ap = p - A
        bp, cp = ap - ab, ap - ac
        d1, d2 = (ab * ap).sum(-1), (ac * ap).sum(-1)
        d3, d4 = (ab * bp).sum(-1), (ac * bp).sum(-1)
        d5, d6 = (ab * cp).sum(-1), (ac * cp).sum(-1)
        va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2
        den = va + vb + vc
        den = np.where(np.abs(den) > 1e-18, den, 1e-18)
        v, w = vb / den, vc / den                       # interior
        # regions, lowest priority first, so that Ericson's first-true test wins
        m = (va <= 0.0) & ((d4 - d3) >= 0.0) & ((d5 - d6) >= 0.0)                    # edge BC
        wbc = sdiv(d4 - d3, (d4 - d3) + (d5 - d6))
        v, w = np.where(m, 1.0 - wbc, v), np.where(m, wbc, w)
        m = (vb <= 0.0) & (d2 >= 0.0) & (d6 <= 0.0)                                    # edge AC
        v, w = np.where(m, 0.0, v), np.where(m, sdiv(d2, d2 - d6), w)
        m = (d6 >= 0.0) & (d5 <= d6)                                                   # vertex C
        v, w = np.where(m, 0.0, v), np.where(m, 1.0, w)
        m = (vc <= 0.0) & (d1 >= 0.0) & (d3 <= 0.0)                                    # edge AB
        v, w = np.where(m, sdiv(d1, d1 - d3), v), np.where(m, 0.0, w)
        m = (d3 >= 0.0) & (d4 <= d3)                                                   # vertex B
        v, w = np.where(m, 1.0, v), np.where(m, 0.0, w)
        m = (d1 <= 0.0) & (d2 <= 0.0)                                                  # vertex A
        v, w = np.where(m, 0.0, v), np.where(m, 0.0, w)
        X = A + v[..., None] * ab + w[..., None] * ac
        dd = np.linalg.norm(X - p, axis=-1)
        j = np.argmin(dd, axis=1)
        r = np.arange(len(p))
        Pb[s:s + chunk], db[s:s + chunk] = X[r, j], dd[r, j]
        jb[s:s + chunk] = j if cand is None else cand[r, j]
        bb[s:s + chunk] = np.c_[1.0 - v[r, j] - w[r, j], v[r, j], w[r, j]]
    return Pb, jb, bb, db


def inside_closed(Q, V, T, chunk=256):
    """Points Q inside the closed triangle surface (V, T): ray parity along +z -- the triangles whose xy projection holds
    the point, crossed above it; an odd count is inside.  The rays are nudged by a fixed sub-micron xy offset so that
    none meets an edge or a vertex exactly.  (Cheap: no square roots; the surface must be closed.)"""
    Q, V = np.atleast_2d(np.asarray(Q, float)), np.asarray(V, float)
    T = np.asarray(T, int)
    A, B, C = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    e1, e2 = B - A, C - A
    det = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    ok = np.abs(det) > 1e-12
    A, e1, e2, det = A[ok], e1[ok], e2[ok], det[ok]
    lo, hi = np.minimum(np.minimum(A, A + e1), A + e2), np.maximum(np.maximum(A, A + e1), A + e2)
    out = np.zeros(len(Q), bool)
    qx = Q[:, 0] + 1.2345678e-7
    qy = Q[:, 1] + 2.3456789e-7
    for s in range(0, len(Q), chunk):
        x, y, z = qx[s:s + chunk, None], qy[s:s + chunk, None], Q[s:s + chunk, 2, None]
        box = (x >= lo[None, :, 0]) & (x <= hi[None, :, 0]) & (y >= lo[None, :, 1]) & (y <= hi[None, :, 1]) &             (z <= hi[None, :, 2])
        ci, ti = np.nonzero(box)
        if not len(ci):
            continue
        dx, dy = x[ci, 0] - A[ti, 0], y[ci, 0] - A[ti, 1]
        u = (dx * e2[ti, 1] - dy * e2[ti, 0]) / det[ti]
        v = (e1[ti, 0] * dy - e1[ti, 1] * dx) / det[ti]
        zz = A[ti, 2] + u * e1[ti, 2] + v * e2[ti, 2]
        hit = (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0) & (zz > z[ci, 0])
        cnt = np.bincount(ci[hit], minlength=len(x))
        out[s:s + chunk] = (cnt % 2) == 1
    return out


# ============================================================================================ K2 "live": the vault
def vault_band(t, band_mm):
    """cfg k2_vault "live": the vault band of the rest shape -- every node within band_mm (rest centreline arclength) of
    the top ring, each with its own GENERATOR: the rest polyline at its angle theta (node_theta_rad, interpolated round
    each station ring of its sheet) through the stations from the anchor (the first station below the band, which stays
    at rest) to the top.  Returns the band's nodes, their polylines PL (n, n_st, 3), arclengths from the anchor (cum),
    each node's own arclength a and its generator's length L, the rows of the top ring (in node set vault_top order) and
    the per-sheet angle weights from each band node to the top ring of its sheet."""
    st, sheet, th, X = t["station"], t["sheet"], t["theta"], t["X_rest"]
    dr = t["depth_rest"]
    Kt = int(st.max())
    kdep = {int(k): float(dr[st == k].min()) for k in np.unique(st)}
    band_st = sorted(k for k, d in kdep.items() if d <= float(band_mm))
    kb = min(band_st) - 1
    if kb < int(st.min()):
        raise ValueError("vault_band_mm %.1f covers the whole wall: no anchor station below it" % float(band_mm))
    ks = list(range(kb, Kt + 1))
    nodes = np.nonzero(np.isin(st, band_st))[0]
    PL = np.zeros((len(nodes), len(ks), 3))
    for sh in (0, 2):
        m = sheet[nodes] == sh
        q = nodes[m]
        for jk, k in enumerate(ks):
            idx = np.nonzero((sheet == sh) & (st == k))[0]
            j0, j1, w = periodic_weights(th[q], th[idx])
            PL[m, jk] = (1.0 - w)[:, None] * X[idx[j0]] + w[:, None] * X[idx[j1]]
    seg = np.linalg.norm(np.diff(PL, axis=1), axis=2)
    cum = np.concatenate([np.zeros((len(nodes), 1)), np.cumsum(seg, axis=1)], axis=1)
    kpos = st[nodes] - kb
    a = cum[np.arange(len(nodes)), kpos]
    pos = {int(i): r for r, i in enumerate(nodes)}
    top = t["top"]
    top_rows = np.array([pos[int(i)] for i in top], int)
    wts = {}
    for sh in (0, 2):
        rows = np.nonzero(sheet[nodes] == sh)[0]
        ref = np.nonzero(sheet[top] == sh)[0]                 # indices into vault_top
        j0, j1, w = periodic_weights(th[nodes[rows]], th[top[ref]])
        wts[sh] = dict(rows=rows, j0=ref[j0], j1=ref[j1], w=w)
    return dict(nodes=nodes, PL=PL, cum=cum, a=a, L=cum[:, -1], kb=kb, ks=ks, top_rows=top_rows, wts=wts,
                exact=float(np.abs(PL[np.arange(len(nodes)), kpos] - X[nodes]).max()))


def _band_interp(vb, v_top):
    """Per-top-node values (98, ...) -> every band node (by angle, sheet by sheet; exact on the top ring)."""
    v_top = np.asarray(v_top, float)
    out = np.zeros((len(vb["nodes"]),) + v_top.shape[1:])
    for sh, wt in vb["wts"].items():
        w = wt["w"].reshape((-1,) + (1,) * (v_top.ndim - 1))
        out[wt["rows"]] = (1.0 - w) * v_top[wt["j0"]] + w * v_top[wt["j1"]]
    return out


def _along(PL, cum, a2, ext):
    """Points at arclength a2 along each polyline PL (from its first point), continued beyond its end by the vector ext
    (straight; a2 is capped at the end of ext)."""
    n, m = PL.shape[0], PL.shape[1]
    L = cum[:, -1]
    le = np.linalg.norm(ext, axis=1)
    a2 = np.minimum(np.maximum(a2, 0.0), L + le)
    k = np.clip((cum[:, 1:] < a2[:, None]).sum(1), 0, m - 2)
    r = np.arange(n)
    sl = cum[r, k + 1] - cum[r, k]
    f = np.clip((a2 - cum[r, k]) / np.where(sl > 1e-12, sl, 1.0), 0.0, 1.0)
    P = PL[r, k] + f[:, None] * (PL[r, k + 1] - PL[r, k])
    beyond = a2 > L + 1e-12
    if beyond.any():
        P[beyond] = PL[beyond, -1] + ((a2[beyond] - L[beyond]) / np.maximum(le[beyond], 1e-12))[:, None] * ext[beyond]
    return P


# ============================================================================================ the drive
class WallDrive:
    """The per-step drive (see the module docstring).  tab = node_tables(...), geo = tandem_geometry(app),
    rg = ring_geometry(app) or None, pr = drive_cfg(...).  step() is pure given (row, Xc, state)."""

    def __init__(self, tab, geo, rg, pr, cervix_tri=None):
        self.t, self.geo, self.rg, self.p = tab, geo, rg, pr
        t = tab
        # glue weight per node: 1 on the top ring, falling to 0 glue_blend_mm (start arclength) below it
        L = float(pr["glue_blend_mm"])
        self.g = 1.0 - smoothstep(t["depth"] / max(L, 1e-9)) if L > 0 else (t["depth"] <= 1e-9).astype(float)
        self.g[t["top"]] = 1.0
        self.n = t["n"]
        self.grid = None
        if int(pr.get("ring_smooth_st") or 0) > 0 or int(pr.get("ring_smooth_th") or 0) > 0:
            self.grid = lumen_grid(t)
        # cfg k2_vault "live": the cervix surface (triangles in cervix tet-node indices) and the rest shape's vault band
        self.live = pr.get("k2_vault", "rest") == "live"
        self.cx_tri = None if cervix_tri is None else np.asarray(cervix_tri, int)
        self.vb = None
        if self.live:
            if self.cx_tri is None:
                raise ValueError("wall_drive_params k2_vault 'live' needs the cervix surface (WallDrive(cervix_tri=...))")
            self.vb = vault_band(t, float(pr["vault_band_mm"]))

    # ---- K2 "live": the vault re-derived on the live cervix
    def vault_pair(self, Xc):
        """The junction of every top-ring node on the cervix Xc (the first K2 row): its generator (vault_band) followed up
        from the band's anchor, then vault_reach_mm straight on beyond the rest top, to where it first crosses the
        cervix surface; where it never does, the cervix surface point closest to the rest top (logged as a fallback).
        Returns the pairing: cervix triangle (oriented outward) + barycentric weights, the crossing E0, the extension
        vector beyond the rest top (0 where the junction lies on the rest polyline), rho = new / rest generator length."""
        vb, p = self.vb, self.p
        Xc = np.asarray(Xc, float)
        T = orient_outward(Xc, self.cx_tri)
        it = vb["top_rows"]
        PL, cum = vb["PL"][it], vb["cum"][it]
        L = cum[:, -1]
        u = unit(PL[:, -1] - PL[:, -2])
        path = np.concatenate([PL, (PL[:, -1] + float(p["vault_reach_mm"]) * u)[:, None, :]], axis=1)
        n, m = len(it), path.shape[1]
        seg_len = np.linalg.norm(np.diff(path, axis=1), axis=2)
        cum_p = np.concatenate([np.zeros((n, 1)), np.cumsum(seg_len, axis=1)], axis=1)
        l_new, tri, bary = np.full(n, np.nan), np.full(n, -1, int), np.zeros((n, 3))
        for s in range(m - 1):
            todo = np.nonzero(np.isnan(l_new))[0]
            if not len(todo):
                break
            tt, jj, bb = seg_first_hit(path[todo, s], path[todo, s + 1], Xc, T)
            h = np.isfinite(tt)
            r = todo[h]
            l_new[r] = cum_p[r, s] + tt[h] * seg_len[r, s]
            tri[r], bary[r] = jj[h], bb[h]
        fb = np.isnan(l_new)
        if fb.any():                                    # never meets the cervix within reach: the closest surface point
            Pc, jj, bb, dd = closest_on_tris(PL[fb, -1], Xc, T)
            tri[fb], bary[fb] = jj, bb
            l_new[fb] = L[fb] + dd
        E0 = np.einsum("ij,ijk->ik", bary, Xc[T[tri]])
        ext = np.where((l_new > L)[:, None], E0 - PL[:, -1], 0.0)
        if fb.any():
            ext[fb] = E0[fb] - PL[fb, -1]
        return dict(T=T, tri=tri, bary=bary, E0=E0, ext=ext, rho=l_new / np.maximum(L, 1e-9), l_new=l_new,
                    n_fallback=int(fb.sum()), n_shortened=int(np.sum(l_new < L - 1e-9)),
                    n_extended=int(np.sum(l_new > L + 1e-9)))

    def vault_live(self, Xc, vs):
        """The K2 target with the vault on the live cervix Xc (pairing vs = vault_pair): the rest shape, the band re-spaced
        along each generator to end at its junction, plus the junction's drift since the pairing (falling linearly to
        0 at the anchor); the top ring exactly at its junction (+ vault_offset_mm along the outward normal).  Returns
        (X, junction positions of the top ring, their outward unit normals)."""
        vb, t = self.vb, self.t
        Xc = np.asarray(Xc, float)
        T3 = Xc[vs["T"][vs["tri"]]]
        E = np.einsum("ij,ijk->ik", vs["bary"], T3)
        nT = unit(np.cross(T3[:, 1] - T3[:, 0], T3[:, 2] - T3[:, 0]))
        J = E + float(self.p["vault_offset_mm"]) * nT
        dT = J - vs["E0"]
        rho, ext, dTi = _band_interp(vb, vs["rho"]), _band_interp(vb, vs["ext"]), _band_interp(vb, dT)
        a2 = rho * vb["a"]
        ln = rho * vb["L"]
        Pn = _along(vb["PL"], vb["cum"], a2, ext)
        X = t["X_rest"].copy()
        X[vb["nodes"]] = Pn + (np.minimum(a2, ln + np.linalg.norm(ext, axis=1)) / np.maximum(ln, 1e-9))[:, None] * dTi
        X[t["top"]] = J
        return X, J, nT

    def junction(self, Xc, state):
        """The top ring's current attachment on the cervix Xc: (positions, outward normals, "start" | "live") -- the
        start glue (glue()) until k2_vault "live" has paired, the live junction afterwards."""
        vs = (state or {}).get("vault")
        if self.live and vs is not None:
            _, J, nT = self.vault_live(Xc, vs)
            return J, nT, "live"
        G, _, nG = self.glue(Xc)
        return G, nG[self.t["top"]], "start"

    def keep_out_of_cervix(self, P, Xc, vs, top_sel):
        """k2_vault "live": the top-ring nodes top_sel put ON the cervix surface (closest point + vault_offset_mm along
        its outward normal) and every other node found inside the cervix put back on its surface alike.  Only nodes in
        the cervix's bounding box (+ 2 mm) are tested.  Returns (P, n_pushed_out, max_depth_mm of those)."""
        t, p = self.t, self.p
        Xc = np.asarray(Xc, float)
        T = vs["T"]
        off = float(p["vault_offset_mm"])
        P = np.array(P, float, copy=True)
        top_sel = np.asarray(top_sel, int)
        lo, hi = Xc.min(0) - 2.0, Xc.max(0) + 2.0
        cand = np.nonzero(np.all((P >= lo) & (P <= hi), axis=1))[0]
        cand = cand[~np.isin(cand, t["top"])]
        ins = cand[inside_closed(P[cand], Xc, T)] if len(cand) else cand
        sel = np.concatenate([top_sel, ins]).astype(int)
        depth = 0.0
        if len(sel):
            Pc, jj, _, dd = closest_on_tris(P[sel], Xc, T, k_near=32)
            if len(ins):
                depth = float(dd[len(top_sel):].max())
            P[sel] = Pc + off * tri_normals(Xc, T[jj])
        return P, int(len(ins)), depth

    def smooth_ring_push(self, rp):
        """cfg ring_smooth_st / ring_smooth_th: the ring push-out rp (per inner node, >= 0) max-filtered then mean-filtered
        over the lumen grid (stations x angle cells, angles wrapped, missing cells ignored)."""
        return grid_smooth(rp, self.grid, int(self.p.get("ring_smooth_st") or 0), int(self.p.get("ring_smooth_th") or 0))

    def init_state(self):
        """The drive's state before the first step: no push, no tip yet, the wall at its START shape."""
        return dict(push=np.zeros(self.n), tip_h_max=-np.inf, k2=0.0, P=self.t["X_start"].copy())

    # ---- pieces (public for the tests and the host gates)
    def glue(self, Xc):
        """The top ring's glue positions (cervix surface points + start offsets), every node's glue displacement (the
        top ring's, interpolated round the ring by angle, sheet by sheet) and every node's local portio normal (the
        paired cervix triangle's, oriented along the start offset, interpolated alike; unit)."""
        t = self.t
        Xc = np.asarray(Xc, float)
        T3 = Xc[t["pair_tri"]]
        G = np.einsum("ij,ijk->ik", t["pair_bary"], T3) + t["pair_off"]
        dG = G - t["X_start"][t["top"]]
        nt = np.cross(T3[:, 1] - T3[:, 0], T3[:, 2] - T3[:, 0])
        nt = unit(nt * np.where(np.einsum("ij,ij->i", nt, t["pair_off"]) < 0.0, -1.0, 1.0)[:, None])
        out = np.zeros((self.n, 3))
        nrm = np.zeros((self.n, 3))
        for sh, gw in t["glue_w"].items():
            for src, dst in ((dG, out), (nt, nrm)):
                ref = src[gw["ref"]]
                dst[gw["idx"]] = (1.0 - gw["w"])[:, None] * ref[gw["j0"]] + gw["w"][:, None] * ref[gw["j1"]]
        out[t["top"]] = dG
        nrm[t["top"]] = nt
        return G, out, unit(nrm)

    def omega(self, tip_h_max):
        """Straightening weight per node from the highest tip height reached so far (monotone by construction)."""
        p = self.p
        e = float(tip_h_max) - self.t["hs"] + float(p["straight_lead_mm"])
        return smoothstep(e / max(float(p["straight_ramp_mm"]), 1e-9))

    def heights(self, kappa):
        t = self.t
        return t["hs"] + float(kappa) * (t["hr"] - t["hs"])

    def step(self, row, Xc, state):
        """One step: the driven positions P (n, 3) for the schedule row `row` (keys F, R_rows, tube_axis; optional
        drive_kappa (K1 lift weight), drive_k2 (K2 front progress 0..1), ring {side: (origin, R_rows)} = the halves'
        CURRENT poses and ring_lead {side: (origin, R_rows)} = where they will be ring_lead_mm further on), the cervix
        nodes Xc (for the glue) and the state (init_state / the previous step's).  Returns (P, new_state, diag)."""
        t, p = self.t, self.p
        a, c0 = t["a"], t["c0"]
        F = np.asarray(row["F"], float)
        Rr = np.asarray(row["R_rows"], float)
        at = unit(Rr[2])
        tip = F + self.geo["L_iu"] * at
        tip_h = float((tip - c0) @ a)
        tip_h_max = max(float(state["tip_h_max"]), tip_h)
        kappa = float(row.get("drive_kappa", 0.0) or 0.0)
        h = self.heights(kappa)
        CLw = device_world(self.geo, F, Rr)
        D, dv, rA, ok, s_nose = device_crossing(h, CLw, self.geo["rad"], at, a, c0, p["clear_mm"], p["lead_mm"])
        om = self.omega(tip_h_max)
        G, dG, nG = self.glue(Xc)
        g = self.g
        # ---- floor: carried collapsed section (straightened, glued, lifted).  A glued node (weight g) takes the top
        #      ring's displacement instead of the straightening and the height change (the portio carries it).
        dh = np.outer(h - t["hs"], a)                   # the height change (K1), along a
        dD = D - (t["Cs"] + dh)                         # the device line vs the slab centre at the current height
        dD = dD - np.outer(dD @ a, a)
        wst = om
        Cb = t["Cs"] + (1.0 - g)[:, None] * (dh + wst[:, None] * dD) + g[:, None] * dG
        B = Cb + t["q"]
        # ---- need: push each lumen node along its ray to the far side of every device part it meets.  The ray starts
        #      on the DEVICE LINE wherever the tandem crosses the node's slab (its section is convex, so the lumen is
        #      star-shaped about it: no wedge of over-opening behind the rod), else at the carried slab centre.
        inn = t["inner"]
        O = np.where(ok[:, None], D, Cb)
        V = B - O
        V = V - np.outer(V @ a, a)
        rhoB_all = np.linalg.norm(V, axis=1)
        Eray = np.where((rhoB_all > 1e-6)[:, None], V / np.maximum(rhoB_all, 1e-12)[:, None], t["E"])
        rhoB = rhoB_all[inn]
        tgt = rhoB.copy()
        r2 = ellipse_exit(O[inn], Eray[inn], D[inn], dv[inn], rA[inn], a, p["cos_min"])
        in_tis = (g >= float(p["glue_push_g"])) & (np.einsum("ij,ij->i", D - B, nG) < -float(p["glue_push_mm"]))
        r2 = np.where(ok[inn] & ~in_tis[inn], r2, -np.inf)
        tgt = np.maximum(tgt, r2)
        ring_push = {}
        if self.rg is not None and row.get("ring"):
            # glued nodes (the vault top on the portio) are NOT pushed by the ring: the ring pushes the portio by contact
            # and the glue follows it.  MEASURED why (TF1r1): with them pushed, the lead poses drove the top ring up to
            # 10 mm INTO the cervix before the half arrived (glue residual 11 mm): the vagina-cervix boundary broke.
            free = g[inn] < float(p["glue_push_g"])
            rp_all = np.zeros(len(inn)) if self.grid is not None else None
            for side, pose in row["ring"].items():
                poses = [pose] + list((row.get("ring_lead") or {}).get(side) or [])
                r_h = self._half_exit(side, poses, O[inn], Eray[inn], rhoB)
                r_h = np.where(free, r_h, rhoB)
                ring_push[side] = int(np.sum(r_h > rhoB + 1e-9))
                if rp_all is None:
                    tgt = np.maximum(tgt, r_h)
                else:
                    rp_all = np.maximum(rp_all, r_h - rhoB)
            if rp_all is not None:                      # ring_smooth_*: one smoothed push-out for all live halves
                rp_s = np.where(free, self.smooth_ring_push(rp_all), 0.0)
                tgt = np.maximum(tgt, rhoB + rp_s)
        push_t = np.zeros(self.n)
        push_t[inn] = tgt - rhoB
        prev = np.asarray(state["push"], float)
        rate = float(p["rate_mm"])
        push_i = np.clip(push_t[inn], prev[inn] - rate, prev[inn] + rate)
        push_i = np.maximum(push_i, 0.0)
        push = np.zeros(self.n)
        push[inn] = push_i
        push = push[t["partner"]]                       # the outer sheet moves with its lumen partner ...
        P = B + push[:, None] * Eray[t["partner"]]      # ... along the partner's ray
        # ---- K2: the packing front blends every node onto its K2 target (k2_vault "rest": the rest shape; "live": the rest
        #      shape with the vault on the live cervix), from the vault down
        k2 = float(row.get("drive_k2", 0.0) or 0.0)
        lam = np.zeros(self.n)
        vs = state.get("vault")
        Xk = J = None
        if k2 > 0.0:
            bl = float(p["k2_blend_mm"])
            Lr = float(t["depth_rest"].max())
            lam = smoothstep((k2 * (Lr + bl) - t["depth_rest"]) / max(bl, 1e-9))
            if k2 >= 1.0:
                lam = np.ones(self.n)
            if self.live:
                if vs is None:                          # the first K2 row: pair the top ring on the cervix as it is now
                    vs = self.vault_pair(Xc)
                Xk, J, _ = self.vault_live(Xc, vs)
            else:
                Xk = t["X_rest"]
            P = (1.0 - lam)[:, None] * P + lam[:, None] * Xk
        vl = None
        if self.live and k2 > 0.0:
            # the TARGET kept off the cervix: the top ring on its surface (the blend of the start glue and the live
            # junction, projected: it slides over the portio), any other node inside it put back on it.  On the target,
            # not on the speed-limited position: projecting the position traps a node whose straight path to its target
            # crosses the portio (MEASURED, host dry run of TF2c_c's K2: 26 nodes stuck on the surface, lag 39.6 mm)
            top = t["top"]
            top_sel = top[np.linalg.norm(P[top] - J, axis=1) > 1e-9]
            P, n_out, dmax = self.keep_out_of_cervix(P, Xc, vs, top_sel)
            vl = dict(n_fallback=vs["n_fallback"], n_shortened=vs["n_shortened"], n_extended=vs["n_extended"],
                      rho_min=float(vs["rho"].min()), rho_max=float(vs["rho"].max()),
                      lam_top=float(lam[top].min()), n_top_to_surface=int(len(top_sel)),
                      n_pushed_out_of_cervix=n_out, pushed_out_depth_max_mm=dmax)
            Xk = P
        # ---- speed: no driven node moves more than speed_mm in one step (the lag is caught up later and logged)
        P_t = P
        lag = 0.0
        vmax = p.get("speed_mm")
        Pp = state.get("P")
        if vmax is not None and Pp is not None:
            dP = P - Pp
            nd = np.linalg.norm(dP, axis=1)
            sc = np.minimum(1.0, float(vmax) / np.maximum(nd, 1e-12))
            P = Pp + dP * sc[:, None]
            lag = float(np.linalg.norm(P_t - P, axis=1).max())
        if k2 >= 1.0 and lag <= 1e-9:
            P = Xk.copy()                               # bit-exact K2 target at the end of K2 ("rest": the rest shape)
        if vl is not None:
            vl["junction_residual_max_mm"] = float(np.linalg.norm(P[t["top"]] - J, axis=1).max())
        new = dict(push=push, tip_h_max=tip_h_max, k2=k2, P=P.copy())
        if vs is not None:
            new["vault"] = vs
        excess = push_i - np.maximum(push_t[inn], 0.0)
        diag = dict(n_pushed=int(np.sum(push_i > 1e-6)), push_max_mm=float(push_i.max()) if len(push_i) else 0.0,
                    push_target_max_mm=float(push_t[inn].max()) if len(inn) else 0.0,
                    lag_open_max_mm=float(np.max(np.maximum(push_t[inn] - push_i, 0.0))) if len(inn) else 0.0,
                    over_open_max_mm=float(excess.max()) if len(excess) else 0.0,
                    omega_mean=float(om.mean()), omega_n_full=int(np.sum(om >= 1.0)), tip_h=tip_h,
                    kappa=kappa, k2=k2, lam_mean=float(lam.mean()), ring_push=ring_push,
                    n_device_ok=int(ok.sum()), speed_lag_mm=lag, n_glued_rod_in_tissue=int(in_tis.sum()),
                    speed_max_mm=(float(np.linalg.norm(P - Pp, axis=1).max()) if Pp is not None else None))
        if vl is not None:
            diag["vault_live"] = vl
        return P, new, diag

    def _half_exit(self, side, poses, O, E, rhoB):
        """Far exit radius along each ray (O + rho E) of ring half `side` at any of its poses (the current one and the
        lead ones: (origin, R_rows) or (origin, R_rows, dd) as lead_poses gives them); rhoB where the ray never meets
        it.  A coarse march of ring_ray_step_mm, refined by bisection.  cfg ring_lead_slope s: a pose dd mm further
        along the approach counts its exit minus s * dd (the cone ahead of the half)."""
        rg, p = self.rg, self.p
        clear = float(p["ring_clear_mm"])
        out = rhoB.copy()
        L, st = float(p["ring_ray_mm"]), float(p["ring_ray_step_mm"])
        rs = np.arange(0.0, L + 1e-9, st)
        sk, reach = half_reach(rg, side)
        slope = p.get("ring_lead_slope")
        for pz in poses:
            org, R = np.asarray(pz[0], float), np.asarray(pz[1], float)
            red = float(slope) * float(pz[2]) if (slope is not None and len(pz) > 2) else 0.0
            if red >= L:                                # the cone is below the floor everywhere: no effect
                continue
            skw = org + sk @ R
            dmin = _capsule_dist(O, skw) if len(skw) > 1 else np.linalg.norm(O - skw[0], axis=1)
            cand = np.nonzero(dmin <= reach + L)[0]
            if not len(cand):
                continue
            Q = O[cand][:, None, :] + (rhoB[cand][:, None] + rs[None, :])[..., None] * E[cand][:, None, :]
            Qa = (Q.reshape(-1, 3) - org) @ R.T         # world -> this pose's applicator frame
            ins = half_inside(Qa, rg, side, clear).reshape(len(cand), len(rs))
            has = ins.any(1)
            if not has.any():
                continue
            last = len(rs) - 1 - np.argmax(ins[:, ::-1], axis=1)
            lo = rhoB[cand] + rs[last]
            hi = lo + st
            ci = cand[has]
            lo, hi = lo[has], hi[has]
            for _ in range(6):
                mid = 0.5 * (lo + hi)
                Pm = O[ci] + mid[:, None] * E[ci]
                im = half_inside((Pm - org) @ R.T, rg, side, clear)
                lo = np.where(im, mid, lo)
                hi = np.where(im, hi, mid)
            out[ci] = np.maximum(out[ci], hi - red)
        return out


def lumen_grid(tab):
    """The lumen (inner sheet) nodes on their topological grid: station index x angle index (meta wall.grid_index
    columns 0 and 2), for grid_smooth.  Returns dict(si, ti, nS, nT) indexed like tab["inner"]."""
    gi = np.asarray(tab["grid_index"], int)[tab["inner"]]
    si, ti = gi[:, 0], gi[:, 2]
    return dict(si=si - si.min(), ti=ti - ti.min(), nS=int(si.max() - si.min() + 1), nT=int(ti.max() - ti.min() + 1))


def _shift(A, ds, dt, fill):
    """A (nS, nT) shifted by ds stations (not wrapped: `fill` enters) and dt angle cells (wrapped)."""
    B = np.roll(A, dt, axis=1)
    if ds > 0:
        B = np.vstack([np.full((ds, A.shape[1]), fill), B[:-ds]])
    elif ds < 0:
        B = np.vstack([B[-ds:], np.full((-ds, A.shape[1]), fill)])
    return B


def grid_smooth(v, grid, ws, wt):
    """Per-node values v (lumen nodes) -> max filter over +/- ws stations and +/- wt angle cells (angles wrapped), then
    the mean of that max over the same window (cells without a node ignored).  The result is >= v at every node: every
    cell of the window around node x has x in ITS window, so its max is >= v(x)."""
    v = np.asarray(v, float)
    if grid is None or (ws <= 0 and wt <= 0):
        return v.copy()
    G = np.full((grid["nS"], grid["nT"]), -np.inf)
    G[grid["si"], grid["ti"]] = v
    valid = np.isfinite(G)
    M = np.full_like(G, -np.inf)
    for ds in range(-ws, ws + 1):
        for dt in range(-wt, wt + 1):
            M = np.maximum(M, _shift(G, ds, dt, -np.inf))
    Mv = np.where(valid & np.isfinite(M), M, 0.0)
    cnt = np.where(valid & np.isfinite(M), 1.0, 0.0)
    S = np.zeros_like(G)
    C = np.zeros_like(G)
    for ds in range(-ws, ws + 1):
        for dt in range(-wt, wt + 1):
            S += _shift(Mv, ds, dt, 0.0)
            C += _shift(cnt, ds, dt, 0.0)
    out = S / np.maximum(C, 1.0)
    return np.maximum(out[grid["si"], grid["ti"]], v)


def half_pose(F, R_rows, h, d):
    """World pose (origin, R_rows) of a ring half at approach distance d on the tandem pose (F, R_rows): the half's OBJ
    is stored at its seat in the applicator frame, so world = F + (p_app + disp(d)) @ R_rows = origin + p_app @ R_rows
    with origin = F + disp(d) @ R_rows."""
    R_rows = np.asarray(R_rows, float)
    return np.asarray(F, float) + approach_disp(h, d) @ R_rows, R_rows


def vault_gap(Xw_top, cervix_sd):
    """Vault-to-portio gap of the top ring: cervix signed distances (positive = outside) of the top-ring nodes;
    returns dict(max_gap_mm, mean_gap_mm, min_sd_mm, n_gt_3mm, n_inside_0p5)."""
    s = np.asarray(cervix_sd(np.asarray(Xw_top, float)), float)
    return dict(max_gap_mm=float(s.max()), mean_gap_mm=float(s.mean()), min_sd_mm=float(s.min()),
                n_gt_3mm=int(np.sum(s > 3.0)), n_inside_0p5=int(np.sum(s < -0.5)))
