"""Stage 1 hybrid pelvis SOFA scene (CONTAINER, SOFA v22.12 + SofaPython3, py3.8).  CONTRACT.md sections 4-5.

UNITS mm / kg / s  ->  force mN, stress kPa, stiffness mN/mm, density kg/mm^3.  Frame: preBT world RAS mm.
Nothing patient-derived is hard-coded: every coordinate comes from `<out>/hybrid/meshes/**` (mesh_bodies.py) and
`<out>/hybrid/applicator/{applicator.json, pose.json}` (applicator_venezia.py).

Graph (FreeMotionAnimationLoop + GenericConstraintSolver; gravity 0; dt 0.02 s):
  root
    CollisionPipeline / BruteForceBroadPhase / BVHNarrowPhase / LocalMinDistance / CollisionResponse(Friction)
    /<body>                      one per DEFORMABLE body (cervix, vagina, bladder, rectum, sigmoid)
        EulerImplicitSolver(rayleigh 0.1/0.1) + SparseLDLSolver(Mat3x3d)
        MeshVTKLoader(tets.vtk) + TetrahedronSetTopology* + MechanicalObject + MeshMatrixMass
        TetrahedralCorotationalFEMForceField   (cfg material=neohookean -> TetrahedronHyperelasticityFEMForceField)
        supports (FixedConstraint / RestShapeSpringsForceField / StiffSpringForceField), see SUPPORTS below
        LinearSolverConstraintCorrection
        /surf  Tetra2TriangleTopologicalMapping + MechanicalObject + IdentityMapping
               Triangle/Line/PointCollisionModel (collision `group` = the contact matrix, see GROUPS)
               /vis  OglModel (distinct colour per body) + IdentityMapping
    /corpus                      RIGID, kinematic (pose rule v2); MechanicalObject template=Rigid3d
        /surf   corpus surface.obj  + RigidMapping -> collision + visual
        /iface  rigid-mapped copy of the cervix `interface_corpus` nodes (AttachConstraint target)
    /tandem                      RIGID, kinematic: tube.obj + shaft.obj  + RigidMapping -> collision + visual
    /ovoids                      RIGID, kinematic: ovoid_L.obj + ovoid_R.obj (seating phase; cfg ovoid_mode)
    /targets                     solver-less Vec3d MOs written by the controller each step
        canal_tgt   dilation targets of the cervix `canal` nodes around the tube axis
        apex_tgt    cervix-following targets of the vagina `apex` nodes
    /supports                    solver-less Vec3d MOs: static anchors of the cardinal-ligament springs

PHASES (controller): P pre-settle (rest state under contacts/supports) -> A approach (u: 0 -> u_ios, tip reaches the
internal os) -> T insertion (u: u_ios -> 1, corpus screw motion s(u)) -> D ovoid seating -> H settle (to convergence).

runSofa: env APPSIM_CFG=/out/hybrid/runs/<tag>/cfg.json, then `runSofa -l SofaPython3 /app/hybrid/scene_hybrid.py`.
"""
import json
import os
import sys
import time

sys.dont_write_bytecode = True                      # never leave __pycache__ in the repo
os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
PARENT = os.path.dirname(HERE)
for _p in (HERE, PARENT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import config  # noqa: E402
import geom  # noqa: E402

import Sofa  # noqa: E402,F401
import Sofa.Core  # noqa: E402
import Sofa.Simulation  # noqa: E402

BODIES = ["corpus", "cervix", "vagina", "bladder", "rectum", "sigmoid"]
DEFORMABLE = ["cervix", "vagina", "bladder", "rectum", "sigmoid"]     # corpus is rigid/kinematic (CONTRACT 1)
DEVICE_PARTS = ["tube", "shaft", "ovoid_L", "ovoid_R"]
TANDEM_PARTS = ["tube", "shaft"]
OVOID_PARTS = ["ovoid_L", "ovoid_R"]

# ----------------------------------------------------------------------------- configuration (CONTRACT 4 defaults)
CFG = dict(
    # --- scenario
    flange_shift_mm=None,           # Delta, seating shift (SCENARIO parameter); None -> pose.json default (22.5)
    # --- Stage 2a: the vagina as an expanding hollow WALL (VAGINA_WALL.md, vagina_wall.py)
    vagina_model="solid",           # "solid" = the Stage-1 tet-meshed collapsed vagina body (DEFAULT, unchanged:
                                    #           every Stage-1 result was produced with this and still is)
                                    # "wall"  = the hollow annulus of vagina_wall.py: the device enters the LUMEN
                                    #           and the wall parts and stretches around it
    vagina_wall_dir="vagina_wall",  # which meshes/vagina_wall[_*] variant to load when vagina_model="wall"
    wall_collision="split",         # "split": separate INNER (lumen; contacts the device) and OUTER (contacts
                                    #          bladder/rectum/cervix) collision models on SUBSETS of the wall
                                    #          surface.  The wall is 1-3 mm thick, i.e. thinner than alarm_mm at
                                    #          the tapered ends, so a single model over the closed boundary would
                                    #          report the device against the far side with a reversed normal.
                                    # "single": one model over the whole wall surface (the Stage-1 pattern).
    wall_contact_parts=["shaft", "ovoid_L", "ovoid_R"],
                                    # device parts that collide with the wall's INNER surface.  The intrauterine
                                    # tube is excluded by default: it leaves the lumen through the apex into the
                                    # cervical canal, and during the approach it leans up to 19 mm off the vaginal
                                    # axis (24 deg to it), i.e. outside the wall rather than inside the lumen.
    wall_outer_exclude=["cervix", "rectum"],
                                    # organs the wall's OUTER surface does NOT collide with.  MEASURED at rest
                                    # (vagina_wall.py build, r0 = 5 mm): the volume-conserving ROUND annulus has an
                                    # outer radius ~7.7 mm where the real label section is a flat slit ~8 x 16 mm,
                                    # so it starts 5.9 mm inside the cervix (107 of 1120 nodes) and 4.0 mm inside
                                    # the rectum (114 nodes).  That is deeper than alarm_mm, and proximity
                                    # detection never sees such a vertex again (the trapped-vertex failure the
                                    # Stage-1 ovoids showed), so contact could not push them out -- it would only
                                    # hold them.  Excluded exactly as Stage 1 excludes its own rest overlaps
                                    # (cervix|vagina -0.87, rectum|sigmoid -0.57).  [] enables every organ.
    insertion_axis="auto",          # direction the device TRANSLATES along.  "tube" = Stage 1 (pose.json
                                    # insertion_path) | "shaft" = CONTRACT 3.4 wording (pose.json
                                    # insertion_path_alt, the vagina's own principal axis) | "auto" = "tube" for
                                    # vagina_model="solid" (byte-identical to Stage 1) and "shaft" for "wall".
                                    # MEASURED: along the TUBE axis the shaft sits 16-27 mm off the vaginal axis at
                                    # mid-insertion, so it does NOT travel up the lumen; along the SHAFT axis it
                                    # stays within 0-8.4 mm of the lumen centreline (the 8.4 is the 30 mm shaft
                                    # arc's own 6.2 mm sagitta plus its 2.18 mm radius).
    device_part_files={},           # per-part override of which OBJ a device part loads, e.g.
                                    # {"shaft": "shaft_straight"} -> applicator/shaft_straight.obj.
                                    # MEASURED, and the reason this exists: the default `shaft` is a 30 mm ARC that
                                    # leaves the flange tangent to the TUBE axis and bends 24 deg anteriorly, so its
                                    # far end sits 6.192 mm off its own chord (applicator.json landmarks.shaft_end
                                    # = [0, 6.192, -29.13]).  The vaginal lumen radius is 4.2-5.0 mm, so that rod
                                    # CANNOT be inside the lumen -- geometrically impossible, whatever the solver
                                    # does.  Wall runs V1B..Y5 therefore had the shaft pressed against the wall from
                                    # OUTSIDE (gap 0.1-0.6 mm, in_contact true) and sliding along it, which reads
                                    # identically to "in contact" unless containment is measured with a SIGN.
                                    # `vagina_wall.py shaft` writes the straight rod, which lies on the lumen
                                    # centreline by construction.  applicator_venezia.py owns the device and is not
                                    # modified; the arc stays the default and this is opt-in.
    static_bodies=[],               # Stage-2a simplification: bodies built as NON-DEFORMABLE collision obstacles
                                    # (no ODE solver, no mass, no FEM, no supports, no constraint correction) instead
                                    # of FEM bodies.  They keep their full tet node set and a `dofs` MechanicalObject
                                    # at the rest positions, so every downstream consumer (run_hybrid.write_outputs,
                                    # frame_cache, the per-body displacement log, eval) is unchanged and simply
                                    # reports u = 0 for them.
                                    # MEASURED (runs XO3P5/XO5/XO7/XO7B, the first wall runs whose ovoids reach the
                                    # wall): the run aborts on the BLADDER, not on the vagina -- bladder min volume
                                    # ratio 0.126-0.162 against the wall's own 0.62-0.89 -- while the bladder drifts
                                    # to umax 15 mm under its own 2 mN/mm supports with nothing pushing it.  The
                                    # bladder is 5234 nodes of near-incompressible (nu 0.49) linear corotational
                                    # tets, i.e. the classic locking / ill-conditioning regime.  For Stage 2a the
                                    # vagina is the object of study and these organs are there to occupy space, so
                                    # making them obstacles takes them out of the failure path and cuts the step cost.
                                    # COST, stated in the README: a static organ cannot deform out of the device's
                                    # way, so its own predicted displacement is identically zero and any
                                    # device-to-OAR distance measured against it is a rest-state distance.
    material_by_body={},            # per-body override of `material`, e.g. {"vagina": "neohookean"}
                                    # (per-step frames for the mid-insertion figures are cfg `frame_every`, which is
                                    #  consumed by run_hybrid.write_frame and deliberately NOT a key here)
    # --- materials (ASSUMED, CONTRACT 4).  E in kPa, nu dimensionless, density kg/mm^3
    material="corotational",        # "corotational" -> TetrahedralCorotationalFEMForceField (default)
                                    # "neohookean"   -> TetrahedronHyperelasticityFEMForceField (ParameterSet mu, k)
    hyper_material="NeoHookean",
    E_kPa=dict(cervix=30.0, vagina=15.0, bladder=8.0, rectum=8.0, sigmoid=8.0),
    nu=dict(cervix=0.45, vagina=0.45, bladder=0.49, rectum=0.45, sigmoid=0.45),
    density_kg_per_mm3=1.05e-6,
    lumped_mass=True,               # MeshMatrixMass lumping (diagonal mass: cheaper and steadier)
    # --- time integration (NUMERICAL)
    dt_s=0.02,
    rayleigh_stiffness=0.0,         # CONTRACT 4 says 0.1, but with implicit Euler and a negligible tissue mass the
                                    # step solves (dt*C + dt^2*K) dv = dt*f, i.e. it reaches only dt/(rayleigh + dt)
                                    # = 17 % of the elastic response per step at 0.1/0.02 s.  The cervix bulk then
                                    # chronically lags the rigidly attached corpus interface and the one-element
                                    # transition layer absorbs the whole drag (MEASURED, runs H1/H2: inversion at
                                    # corpus s = 0.28 with only 4.2 mm of displacement).  0 => each step is
                                    # essentially the static solve, which is what a quasi-static scene wants.
    rayleigh_mass=0.1,
    motion_vel_scale=1.0,           # velocity multiplier at the start of each moving step (1 = dynamic)
    settle_vel_scale=0.0,           # ... during the settle phase (0 = fully quasi-static relaxation)
    # --- collision / contact (CONTRACT 4)
    alarm_mm=1.5, contact_mm=0.5, friction_mu=0.2, angle_cone=0.01,
    gcs_max_it=250, gcs_tol=1e-3,
    tandem_lumen_contact=False,     # The intrauterine tube and the shaft travel inside the CERVICAL CANAL and the
                                    # VAGINAL LUMEN.  Neither lumen is meshed: the bodies are solid, voxel-exclusive
                                    # volumes, so a closed surface stands where the lumen is and contact simply
                                    # forbids the tandem from ever entering -- MEASURED (probe 7): the advancing
                                    # tube rams the cervix surface and crushes its elements (min volume ratio 0.94
                                    # at 0.48 mm of displacement, then inversion) while the corpus has not moved.
                                    # Default False: tube|cervix, tube|vagina, shaft|cervix and shaft|vagina are
                                    # excluded from contact, and the tandem acts on the cervix through the canal
                                    # dilation tie instead.  The ovoids, which are outside the tissue, always
                                    # interact by contact.  True = let those four pairs collide.
    ovoid_mode="travel",            # "travel": the ovoids ride with the tandem from u = 0, so they enter from below
                                    #           the introitus and never appear inside tissue (DEFAULT).
                                    # "seat"  : parked below the introitus, then seated over the last ovoid_seat_mm.
                                    #           MEASURED (runs H4/H5): the ovoids then MATERIALISE inside the vagina
                                    #           at the start of the seating phase; the vertices they enclose are
                                    #           further from the ovoid surface than alarmDistance, so proximity
                                    #           detection never sees them again and they stay trapped -- 5-7 mm of
                                    #           permanent interpenetration, identical to 3 decimals whether the
                                    #           seating takes 8 or 20 steps, which is how the teleport (not the step
                                    #           size) was identified as the cause.
                                    # "off"   : ovoid collision disabled (visual only)
    ovoid_seat_mm=30.0,             # only used by ovoid_mode="seat"
    ovoid_vagina_contact=False,     # The ovoids seat in the vaginal FORNICES, i.e. inside the vaginal lumen, which
                                    # the collapsed-lumen vagina body does not represent (there is no configuration
                                    # in which a 39 mm ovoid sits in the fornix without overlapping that mesh).
                                    # Default False, as for the tandem; the vagina follows the cervix through the
                                    # apex springs instead.  The ovoids always contact the cervix (portio seating),
                                    # the bladder and the rectum.
    # --- couplings (CONTRACT 4)
    attach_impl="attach",           # cervix.interface_corpus <-> rigid-mapped corpus copies:
                                    # "attach" = AttachConstraint | "springs" = stiff RestShapeSprings to the copies
    k_attach_mN_per_mm=2000.0,      # only for attach_impl="springs"
    canal_tie=True,                 # cervix.canal nodes dilated onto the tube (sliding along the axis)
    k_canal_mN_per_mm=200.0,
    canal_slack_mm=0.0,             # tie radius = r_tandem + slack
    canal_max_offset_mm=0.5,        # the tie target is at most this far from the node's CURRENT position, so the tie
                                    # force is bounded by k_canal * canal_max_offset (the dilation is still reached,
                                    # over several steps).  Without the bound the target ratchets out to the full
                                    # tube radius and the force reaches k * 2.18 mm, which collapsed the cervix
                                    # elements around the canal (MEASURED, run H1: min volume ratio 0.92 -> 0.06).
    tie_ramp_steps=3,
    apex_attach="follow",           # vagina.apex follows the nearest cervix surface node ("follow") | "off"
    k_apex_mN_per_mm=20.0,
    # --- supports (CONTRACT 4)
    k_cardinal_mN_per_mm=20.0, cardinal_len_mm=25.0,
    cardinal_tension_only=True,     # a ligament is a cable: it resists stretch beyond cardinal_len_mm only
    k_bladder_support_mN_per_mm=2.0,
    k_rectum_support_mN_per_mm=2.0,
    vagina_inferior="fixed",        # "fixed" (CONTRACT) | "spring" (k_vagina_inferior_mN_per_mm)
    k_vagina_inferior_mN_per_mm=50.0,
    vagina_inferior_outer_only=False,
                                    # WALL runs only.  `fixed_inferior` is the introitus ring, and for a hollow wall
                                    # it contains the LUMEN nodes too -- the very nodes the device must push apart as
                                    # it enters.  MEASURED (run V1B, r0 5, n_radial 1): every one of the 8 elements
                                    # below volume ratio 0.5 at the abort touches a `fixed_inferior` node, and all 8
                                    # sit in the two pinned rings (s = -26.1 and -24.1, wall 1.81 and 2.00 mm thick)
                                    # -- while the lumen centre there is only 0.6 mm off the device axis, i.e. the
                                    # shaft IS inside the lumen and still cannot open it.  The device is being driven
                                    # through a ring of nodes held by 1e4 mN/mm springs.  True pins only the OUTER (the
                                    # wall is anchored to the pelvic floor on its outside), leaving the lumen free to
                                    # open.  Default False = unchanged.
    fix_rectum_ends=True, fix_sigmoid_ends=True,
    fixed_impl="spring",            # how a "fixed" node set is imposed:
                                    # "spring"     = RestShapeSpringsForceField at k_fixed_mN_per_mm (stiff but
                                    #                COMPLIANT).  MEASURED: with hard FixedConstraint the rest-state
                                    #                contacts that fall on fixed nodes (vagina|rectum 0.19 mm,
                                    #                corpus|sigmoid 0.05 mm) have zero compliance, so the
                                    #                Gauss-Seidel constraint solve divides by w = 0 and the residual
                                    #                is NaN from step 0 (run HSMOKE).
                                    # "constraint" = FixedConstraint (CONTRACT wording; NaN risk above)
    k_fixed_mN_per_mm=10000.0,      # 1e4 mN/mm: a 1 N load moves such a node 0.1 mm
    # --- schedule (NUMERICAL; CONTRACT 3.4 n_steps 80 is split into approach/insertion, see README)
    n_presettle=4, n_approach=8, n_insert=24, n_seat=12,
    max_settle_steps=30, max_steps=120, wall_limit_s=470.0,
    # --- convergence (CONTRACT 5) and stop rules
    conv_dx_mm=0.02, conv_steps=5,
    conv_constraint_err_per_contact=0.01,
                                    # GenericConstraintSolver residual (`currentError`) PER CONTACT.  currentError is
                                    # an absolute sum over the active set, so it scales with the contact count
                                    # (MEASURED: 0.19 at rest with 85 contacts, 0.78 at the seated pose with 280 --
                                    # 0.0022 and 0.0028 per contact, run H3).  Normalising makes the tolerance
                                    # independent of how much of the anatomy is touching.  Convergence ALSO requires
                                    # `constraint_converged` (the solver met its own 1e-3 tolerance inside its cap).
    min_vol_ratio_abort=0.2,
    # --- bookkeeping
    log_every=1,
)

# Collision groups: two models collide iff their group sets are DISJOINT.  Shared ids therefore encode the
# excluded pairs (CONTRACT 4 + the rest-state overlaps of bodies.json["pairs"]).  Every model has a non-empty
# group, so self-collision is off everywhere.
#   1 corpus|cervix (attached, -0.99 mm rest overlap)   2 cervix|vagina (apex follow, -0.87 mm)
#   3 rectum|sigmoid (-0.57 mm, both ends fixed)        4 bladder (self only)
#   9 corpus|device (the tandem is the corpus's own)   10 vagina|tandem   11 cervix|tandem  (unmeshed lumina)
#  12 vagina|ovoids (the ovoids seat in the vaginal lumen; cfg ovoid_vagina_contact)
GROUPS = dict(corpus=[1, 9], cervix=[1, 2, 11], vagina=[2, 10, 12], bladder=[4], rectum=[3], sigmoid=[3],
              tube=[9, 10, 11], shaft=[9, 10, 11], ovoid_L=[9, 12], ovoid_R=[9, 12])


def groups_for(cfg):
    """Collision groups, extended when the vagina is a hollow WALL with two sides.

    Stage 1 ("solid") returns GROUPS unchanged.  For the wall three further ids are introduced, remembering that
    two models collide iff their group sets are DISJOINT:
      30-34  one PRIVATE id per organ (corpus, cervix, bladder, rectum, sigmoid).  Each belongs to exactly ONE
             organ, so adding them changes no organ-organ pair; the wall then excludes itself from an organ by
             carrying that organ's private id.  A single shared "all organs" id would NOT do: it would make every
             organ pair share an id and silently switch off cervix|bladder, rectum|bladder and the rest.
      21  every device part + the wall's OUTER surface -> the device never contacts the outside of the wall
      22  the two wall sides                      -> the wall does not collide with itself through its thickness,
                                                     and a device part not in `wall_contact_parts` also carries 22
                                                     so it is excluded from the lumen as well.
    The INNER surface carries every private organ id (it faces the device, never an organ); the OUTER surface
    carries 21 plus the private id of each organ in `wall_outer_exclude`.
    """
    g = {k: list(v) for k, v in cfg.get("collision_groups", GROUPS).items()}
    if cfg.get("vagina_model", "solid") != "wall":
        return g
    own = dict(corpus=30, cervix=31, bladder=32, rectum=33, sigmoid=34)
    for b, i in own.items():
        g[b] = g[b] + [i]
    keep = set(cfg.get("wall_contact_parts", []))
    for p in DEVICE_PARTS:
        g[p] = g[p] + [21] + ([] if p in keep else [22])
    g["vagina_inner"] = [22] + sorted(own.values())
    g["vagina_outer"] = [21, 22] + sorted(own[b] for b in cfg.get("wall_outer_exclude", []) if b in own)
    g["vagina"] = list(g["vagina_inner"])                       # wall_collision="single" uses the inner groups
    return g


def load_cfg(overrides=None):
    """CFG updated by a dict or a JSON file path (nested dicts are merged one level deep)."""
    cfg = json.loads(json.dumps(CFG))
    if overrides is None:
        return cfg
    if isinstance(overrides, str):
        overrides = json.loads(overrides) if overrides.strip().startswith("{") else json.load(open(overrides))
    unknown = [k for k in overrides if k not in CFG and not k.startswith("_") and k != "tag"]
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(cfg.get(k), dict):
            cfg[k].update(v)
        else:
            cfg[k] = v
    cfg["_unknown_keys"] = unknown
    return cfg


def paths():
    P = config.paths()
    hyb = P["out"] + "/hybrid"
    return dict(out=P["out"], hybrid=hyb, meshes=hyb + "/meshes", applicator=hyb + "/applicator",
                runs=hyb + "/runs", figs=hyb + "/figs", logs=hyb + "/logs")


# ----------------------------------------------------------------------------- inputs
def read_vtk_points(path):
    """Points of the legacy ASCII VTK unstructured grids written by mesh_bodies.write_vtk_legacy."""
    with open(path) as fh:
        tok = fh.read().split()
    i = tok.index("POINTS")
    n = int(tok[i + 1])
    return np.array(tok[i + 3:i + 3 + 3 * n], float).reshape(n, 3)


def load_inputs(cfg):
    P = paths()
    if cfg.get("vagina_model", "solid") == "wall":
        # run_hybrid.py builds every per-body output path from this single `meshes` directory, so the wall has to
        # be presented under the name `vagina` inside a directory that looks like meshes/ (the other five bodies
        # are copies).  vagina_wall.scene_mesh_root builds and refreshes it.
        import vagina_wall
        P = dict(P, meshes=vagina_wall.scene_mesh_root(P["meshes"], cfg["vagina_wall_dir"]))
    d = dict(P=P, bodies=json.load(open(P["meshes"] + "/bodies.json")),
             app=json.load(open(P["applicator"] + "/applicator.json")),
             pose=json.load(open(P["applicator"] + "/pose.json")), meta={})
    for b in BODIES:
        d["meta"][b] = json.load(open("%s/%s/meta.json" % (P["meshes"], b)))
    return d


def _param(app, key):
    return app["params"][key]["value"]


def corpus_target(inp, cfg):
    """The pose-rule corpus target and device path for the chosen seating shift Delta (pose.json, rule v2)."""
    pose = inp["pose"]
    tbl = pose["corpus"]["by_flange_shift_mm"]
    dz = cfg.get("flange_shift_mm")
    dz = float(pose["default_flange_shift_mm"]) if dz is None else float(dz)
    key = None
    for k in tbl:
        if abs(float(k) - dz) < 1e-9:
            key = k
    if key is None:
        raise ValueError("flange_shift_mm %g not in pose.json corpus.by_flange_shift_mm (available: %s)"
                         % (dz, sorted(float(k) for k in tbl)))
    e = tbl[key]
    # Which axis the device TRANSLATES along (the final pose is the same either way: Delta and both candidate
    # axes only change where the path STARTS).  See CFG["insertion_axis"].
    mode = cfg.get("insertion_axis", "tube")
    if mode == "auto":
        mode = "shaft" if cfg.get("vagina_model", "solid") == "wall" else "tube"
    if mode == "tube":
        p_axis, travel, u_ios = geom.unit(pose["insertion_path"]["axis"]), float(e["travel_mm"]), \
            float(e["u_tip_at_internal_os"])
    elif mode == "shaft":
        alt = pose["insertion_path_alt"]
        p_axis = geom.unit(alt["axis"])
        # Delta slides the flange along the shaft axis and this path translates along that same axis, so the
        # travel at any Delta is the stored one shifted by Delta; u_ios is recomputed as the u at which the tip
        # reaches the internal-os level along the path.
        travel = float(alt["travel_mm"]) + (dz - float(pose["default_flange_shift_mm"]))
        os_pt = np.array(pose["inputs"]["internal_os"]["value"], float)
        u_ios = float(1.0 - float((np.array(e["tip"], float) - os_pt) @ p_axis) / travel)
    else:
        raise ValueError("insertion_axis must be 'auto', 'tube' or 'shaft' (got %r)" % mode)
    return dict(delta_mm=dz, key=key, flange=np.array(e["flange"], float), tip=np.array(e["tip"], float),
                screw=e["screw"], T=np.array(e["T_preBT_to_target"], float), R=np.array(e["R"], float),
                t=np.array(e["t"], float), L_end_target=np.array(e["L_end_target"], float),
                u_ios=u_ios, travel_mm=travel, insertion_axis=mode,
                axis=p_axis, tube_axis=geom.unit(pose["device_final"]["tube_axis"]),
                R_rows=np.array(pose["device_final"]["R_rows"], float),
                is_default=bool(abs(dz - float(pose["default_flange_shift_mm"])) < 1e-9))


def screw_at(screw, s):
    """Rigid transform at fraction s of the screw motion (applicator_venezia.screw_interp; 4x4)."""
    T = np.eye(4)
    k = np.asarray(screw["axis"], float)
    if screw.get("pure_translation"):
        T[:3, 3] = s * float(screw["pitch_mm"]) * k
        return T
    p0 = np.asarray(screw["point"], float)
    Rs = rodrigues(k, s * float(screw["angle_deg"]))
    T[:3, :3] = Rs
    T[:3, 3] = p0 - Rs @ p0 + s * float(screw["pitch_mm"]) * k
    return T


def rodrigues(k, deg):
    k = geom.unit(k)
    a = np.radians(float(deg))
    K = geom.skew(k)
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * (K @ K)


def quat_from_R(R):
    """[qx, qy, qz, qw] of a proper rotation matrix (SOFA Rigid3d convention)."""
    R = np.asarray(R, float)
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = 0.5 / np.sqrt(tr + 1.0)
        q = np.array([(R[2, 1] - R[1, 2]) * s, (R[0, 2] - R[2, 0]) * s, (R[1, 0] - R[0, 1]) * s, 0.25 / s])
    else:
        i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * np.sqrt(max(1e-30, 1.0 + R[i, i] - R[j, j] - R[k, k]))
        q = np.zeros(4)
        q[3] = (R[k, j] - R[j, k]) / s
        q[i] = 0.25 * s
        q[j] = (R[j, i] + R[i, j]) / s
        q[k] = (R[k, i] + R[i, k]) / s
    return q / np.linalg.norm(q)


def rigid_pose(R, t):
    q = quat_from_R(R)
    return [[float(t[0]), float(t[1]), float(t[2]), float(q[0]), float(q[1]), float(q[2]), float(q[3])]]


# ----------------------------------------------------------------------------- schedule
def build_schedule(cfg, tgt):
    """Per-step device / corpus kinematics.  u in [0, 1] is the pose-rule path parameter (pose.json insertion_path):
    the device flange is F(u) = F_final - (1 - u) * travel * axis, and the corpus follows the screw motion
    s(u) = smoothstep((u - u_ios) / (1 - u_ios)).  Phases: P (pre-settle), A (approach), T (insertion),
    D (ovoid seating), H (settle)."""
    F1, a, D, u_ios = tgt["flange"], tgt["axis"], tgt["travel_mm"], tgt["u_ios"]

    def F_of(u):
        return F1 - (1.0 - u) * D * a

    def s_of(u):
        return geom.smoothstep((u - u_ios) / max(1e-9, 1.0 - u_ios)) if u > u_ios else 0.0

    sched = []
    for k in range(int(cfg["n_presettle"])):
        sched.append(dict(phase="P", u=0.0, s=0.0, ov_lag=None))
    nA = max(1, int(cfg["n_approach"]))
    for k in range(1, nA + 1):
        sched.append(dict(phase="A", u=u_ios * k / nA, s=0.0, ov_lag=None))
    nT = max(1, int(cfg["n_insert"]))
    for k in range(1, nT + 1):
        u = u_ios + (1.0 - u_ios) * k / nT
        sched.append(dict(phase="T", u=u, s=s_of(u), ov_lag=None))
    nD = max(0, int(cfg["n_seat"]))
    seat = float(cfg["ovoid_seat_mm"])
    for k in range(1, nD + 1):
        sched.append(dict(phase="D", u=1.0, s=1.0, ov_lag=seat * (1.0 - k / float(nD))))
    for row in sched:                                  # ovoid lag during P/A/T depends on the mode
        if row["ov_lag"] is None:
            if cfg["ovoid_mode"] == "travel":
                row["ov_lag"] = 0.0
            else:                                      # "seat"/"off": parked at the u = 0 flange, below the introitus
                row["ov_lag"] = float((F_of(row["u"]) - F_of(0.0)) @ a)
    for row in sched:
        row["F"] = F_of(row["u"])
        row["T_corpus"] = screw_at(tgt["screw"], row["s"])
    return sched


# ----------------------------------------------------------------------------- scene
def add_deformable(root, b, inp, cfg, X0, static=False):
    """One body.  `static=True` (cfg `static_bodies`) builds it as a NON-DEFORMABLE collision obstacle: the tet
    topology, the full-node `dofs` MechanicalObject and the surface collision models are kept exactly as for a
    deformable body -- so ctrl.X(b), run_hybrid.write_outputs and frame_cache see identical shapes and simply read
    u = 0 -- but there is no ODE solver, no mass, no FEM and (see build_scene) no constraint correction, so nothing
    integrates the body and it stays at its rest pose."""
    P = inp["P"]
    wall = bool(b == "vagina" and cfg.get("vagina_model", "solid") == "wall")
    n = root.addChild(b)
    if not static:
        n.addObject("EulerImplicitSolver", name="ode", rayleighStiffness=float(cfg["rayleigh_stiffness"]),
                    rayleighMass=float(cfg["rayleigh_mass"]))
        n.addObject("SparseLDLSolver", name="ls", template="CompressedRowSparseMatrixMat3x3d")
    n.addObject("MeshVTKLoader", name="loader", filename="%s/%s/tets.vtk" % (P["meshes"], b))
    n.addObject("TetrahedronSetTopologyContainer", name="topo", src="@loader")
    n.addObject("TetrahedronSetTopologyModifier", name="tmod")
    n.addObject("TetrahedronSetGeometryAlgorithms", name="tgeo", template="Vec3d")
    n.addObject("MechanicalObject", name="dofs", template="Vec3d", src="@loader")
    E = float(cfg["E_kPa"][b])
    nu = float(cfg["nu"][b])
    if not static:
        n.addObject("MeshMatrixMass", name="mass", topology="@topo", massDensity=float(cfg["density_kg_per_mm3"]),
                    lumping=bool(cfg["lumped_mass"]))
        if cfg.get("material_by_body", {}).get(b, cfg["material"]) == "neohookean":
            mu = E / (2.0 * (1.0 + nu))
            kk = E / (3.0 * (1.0 - 2.0 * nu))
            n.addObject("TetrahedronHyperelasticityFEMForceField", name="fem", topology="@topo",
                        materialName=cfg["hyper_material"], ParameterSet="%.6g %.6g" % (mu, kk))
        else:
            n.addObject("TetrahedralCorotationalFEMForceField", name="fem", topology="@topo", method="large",
                        youngModulus=E, poissonRatio=nu)
    s = n.addChild("surf")
    s.addObject("TriangleSetTopologyContainer", name="stopo")
    s.addObject("TriangleSetTopologyModifier", name="smod")
    s.addObject("TriangleSetGeometryAlgorithms", name="sgeo", template="Vec3d")
    s.addObject("Tetra2TriangleTopologicalMapping", name="t2t", input="@../topo", output="@stopo")
    s.addObject("MechanicalObject", name="sdofs", template="Vec3d")
    s.addObject("IdentityMapping", name="idm", input="@../dofs", output="@sdofs")
    grp = groups_for(cfg)
    if wall and cfg["wall_collision"] == "split":
        # One model per SIDE of the wall, on a subset of its surface, so the lumen and the outside can carry
        # different contact groups (see groups_for).  `surf` above keeps the visual model only.
        meta = inp["meta"][b]
        for side in ("inner", "outer"):
            idx = np.asarray(meta["node_sets"][side + "_surface"], int)
            tri = np.asarray(meta["node_set_extra"][side + "_triangles"], int)
            rem = -np.ones(len(X0), np.int64)
            rem[idx] = np.arange(len(idx))
            ch = n.addChild(side)
            ch.addObject("MechanicalObject", name="sdofs", template="Vec3d", position=X0[idx].tolist())
            ch.addObject("MeshTopology", name="stopo", position=X0[idx].tolist(), triangles=rem[tri].tolist())
            ch.addObject("SubsetMapping", name="sub", input="@../dofs", output="@sdofs",
                         indices=[int(i) for i in idx])
            gg = list(grp["vagina_" + side])
            ch.addObject("TriangleCollisionModel", name="tri", group=gg)
            ch.addObject("LineCollisionModel", name="lin", group=gg)
            ch.addObject("PointCollisionModel", name="pnt", group=gg)
    else:
        g = list(grp[b])
        # a static body is an OBSTACLE: moving=False, simulated=False makes the contact one-sided, so the deformable
        # side alone carries the constraint and no constraint correction is needed on this node
        kw = dict(moving=False, simulated=False) if static else {}
        s.addObject("TriangleCollisionModel", name="tri", group=g, **kw)
        s.addObject("LineCollisionModel", name="lin", group=g, **kw)
        s.addObject("PointCollisionModel", name="pnt", group=g, **kw)
    v = s.addChild("vis")
    col = list(inp["bodies"]["bodies"][b]["color"]) + [1.0]
    v.addObject("OglModel", name="ogl", color=col)
    v.addObject("IdentityMapping", name="vm", input="@../sdofs", output="@ogl")
    return n


def add_rigid_parts(root, name, files, groups, colors, pose0):
    """A kinematic Rigid3d body carrying triangle surfaces (collision + visual) through RigidMapping.
    The surfaces' own coordinates are the LOCAL frame (device: applicator frame; corpus: preBT world at rest)."""
    nd = root.addChild(name)
    nd.addObject("MechanicalObject", name="rig", template="Rigid3d", position=pose0)
    for k, (nm, f) in enumerate(files):
        ch = nd.addChild(nm)
        ch.addObject("MeshObjLoader", name="ld", filename=f)
        ch.addObject("MeshTopology", name="mt", src="@ld")
        ch.addObject("MechanicalObject", name="mo", src="@ld")
        ch.addObject("RigidMapping", name="rm", input="@../rig", output="@mo")
        g = list(groups[nm])
        if g:
            ch.addObject("TriangleCollisionModel", name="tri", group=g, moving=True, simulated=False)
            ch.addObject("LineCollisionModel", name="lin", group=g, moving=True, simulated=False)
            ch.addObject("PointCollisionModel", name="pnt", group=g, moving=True, simulated=False)
        vis = ch.addChild("vis")
        vis.addObject("OglModel", name="ogl", src="@../ld", color=list(colors[nm]))
        vis.addObject("IdentityMapping", name="vm", input="@../mo", output="@ogl")
    return nd


def build_scene(root, cfg=None, inp=None):
    cfg = cfg if cfg is not None else load_cfg(os.environ.get("APPSIM_CFG"))
    inp = inp or load_inputs(cfg)
    P = inp["P"]
    tgt = corpus_target(inp, cfg)
    grp = groups_for(cfg)
    X0 = {b: read_vtk_points("%s/%s/tets.vtk" % (P["meshes"], b)) for b in BODIES}

    root.gravity = [0.0, 0.0, 0.0]
    root.dt = float(cfg["dt_s"])
    root.addObject("RequiredPlugin", name="pComp", pluginName="Sofa.Component")
    root.addObject("RequiredPlugin", name="pGL", pluginName="Sofa.GL.Component")
    root.addObject("VisualStyle", name="style",
                   displayFlags="showVisual showBehaviorModels hideCollisionModels hideWireframe hideMappings")
    root.addObject("FreeMotionAnimationLoop", name="fal")
    root.addObject("GenericConstraintSolver", name="gcs", maxIterations=int(cfg["gcs_max_it"]),
                   tolerance=float(cfg["gcs_tol"]), computeConstraintForces=True)
    root.addObject("CollisionPipeline", name="pipe")
    root.addObject("BruteForceBroadPhase", name="bf")
    root.addObject("BVHNarrowPhase", name="bvh")
    root.addObject("LocalMinDistance", name="lmd", alarmDistance=float(cfg["alarm_mm"]),
                   contactDistance=float(cfg["contact_mm"]), angleCone=float(cfg["angle_cone"]))
    root.addObject("CollisionResponse", name="resp", response="FrictionContactConstraint",
                   responseParams="mu=%g" % float(cfg["friction_mu"]))

    # ---- solver-less target / anchor states written by the controller
    tg = root.addChild("targets")
    sup = root.addChild("supports")

    # ---- deformable bodies (cfg `static_bodies` are built as non-deformable collision obstacles instead)
    stat = [b for b in DEFORMABLE if b in cfg.get("static_bodies", [])]
    defo = [b for b in DEFORMABLE if b not in stat]
    nodes = {}
    for b in DEFORMABLE:
        nodes[b] = add_deformable(root, b, inp, cfg, X0[b], static=(b in stat))

    # ---- rigid corpus (pose rule) + its rigid-mapped copy of the cervix interface nodes
    cdev = dict(corpus=list(inp["bodies"]["bodies"]["corpus"]["color"]) + [1.0])
    corp = add_rigid_parts(root, "corpus", [("surf", "%s/corpus/surface.obj" % P["meshes"])],
                           dict(surf=grp["corpus"]), dict(surf=cdev["corpus"]), rigid_pose(np.eye(3), np.zeros(3)))
    iface = np.asarray(inp["meta"]["cervix"]["node_sets"]["interface_corpus"], int)
    ifn = corp.addChild("iface")                       # local coords == preBT world (the corpus rigid starts at identity)
    ifn.addObject("MechanicalObject", name="mo", template="Vec3d", position=X0["cervix"][iface].tolist())
    ifn.addObject("RigidMapping", name="rm", input="@../rig", output="@mo")

    # ---- rigid device: tandem (tube + shaft) and ovoids (seated separately, cfg ovoid_mode)
    dcol = dict(tube=[0.15, 0.15, 0.20, 1.0], shaft=[0.15, 0.15, 0.20, 1.0],
                ovoid_L=[0.35, 0.35, 0.42, 1.0], ovoid_R=[0.35, 0.35, 0.42, 1.0])
    dgrp = {k: list(grp[k]) for k in DEVICE_PARTS}
    if cfg.get("vagina_model", "solid") != "wall":       # (the wall has a real lumen: `wall_contact_parts` governs
                                                        #  which device parts touch it, see groups_for)
        if cfg["tandem_lumen_contact"]:                 # let the tandem collide with the cervix and the vagina
            for p in TANDEM_PARTS:
                dgrp[p] = [g for g in dgrp[p] if g not in (set(grp["vagina"]) | set(grp["cervix"]))]
        if cfg["ovoid_vagina_contact"]:                 # let the ovoids collide with the vagina
            for p in OVOID_PARTS:
                dgrp[p] = [g for g in dgrp[p] if g not in set(grp["vagina"])]
    if cfg["ovoid_mode"] == "off":
        dgrp["ovoid_L"] = dgrp["ovoid_R"] = []
    sched = build_schedule(cfg, tgt)
    F0 = sched[0]["F"]
    Rdev = tgt["R_rows"].T                              # columns = applicator x, y, z in preBT world
    pf = cfg.get("device_part_files", {})               # cfg override of a part's OBJ (e.g. the straight shaft)

    def _pobj(p):
        return "%s/%s.obj" % (P["applicator"], pf.get(p, p))

    tandem = add_rigid_parts(root, "tandem", [(p, _pobj(p)) for p in TANDEM_PARTS],
                             dgrp, dcol, rigid_pose(Rdev, F0))
    ov_pose = F0 - sched[0]["ov_lag"] * tgt["axis"]
    ovoids = add_rigid_parts(root, "ovoids", [(p, _pobj(p)) for p in OVOID_PARTS],
                             dgrp, dcol, rigid_pose(Rdev, ov_pose))

    # ---- couplings and supports
    ctx = dict(root=root, cfg=cfg, inp=inp, tgt=tgt, sched=sched, X0=X0, nodes=nodes, corpus=corp,
               tandem=tandem, ovoids=ovoids, iface=iface, Rdev=Rdev, targets=tg, supports=sup, extra={},
               deformable=defo, static=stat)
    _add_couplings(ctx)
    _add_supports(ctx)
    _wall_diag_setup(ctx)                               # read-only per-step lumen diagnostics (wall runs only)
    # Constraint correction LAST in each body node, so the compliance it builds from the linear solver sees every
    # force field above it.  Without it the constraint solver has no compliance at all: W = 0, the Gauss-Seidel
    # residual and every multiplier come back NaN and contacts produce no force (MEASURED, probe 5).
    for b in defo:
        nodes[b].addObject("LinearSolverConstraintCorrection", name="cc", linearSolver="@ls")
    ctx["tets"], ctx["vol0"] = {}, {}                   # filled by post_init (the topology exists only after init)
    return ctx


def _add_couplings(ctx):
    cfg, inp, X0 = ctx["cfg"], ctx["inp"], ctx["X0"]
    cvx = ctx["nodes"]["cervix"]
    iface = ctx["iface"]
    # (a) cervix.interface_corpus  <->  rigid-mapped corpus copies (CONTRACT 4).  The copies are initialised AT the
    #     cervix rest positions, so the constraint is a no-op at rest and the corpus then carries those nodes.
    if cfg["attach_impl"] == "attach":
        cvx.addObject("AttachConstraint", name="attach_corpus", object1="@/corpus/iface/mo", object2="@/cervix/dofs",
                      indices1=list(range(len(iface))), indices2=[int(i) for i in iface], twoWay=False)
    else:
        cvx.addObject("RestShapeSpringsForceField", name="attach_corpus", points=[int(i) for i in iface],
                      stiffness=[float(cfg["k_attach_mN_per_mm"])] * len(iface),
                      external_rest_shape="@/corpus/iface/mo", external_points=list(range(len(iface))))
    # (b) cervix.canal: dilation ties onto the tube (lateral only; free sliding along the axis), written per step
    canal = np.asarray(inp["meta"]["cervix"]["node_sets"]["canal"], int)
    ctx["canal"] = canal
    if cfg["canal_tie"] and len(canal):
        ctx["canal_tgt"] = ctx["targets"].addObject("MechanicalObject", name="canal_tgt", template="Vec3d",
                                                    position=X0["cervix"][canal].tolist())
        ctx["canal_ff"] = cvx.addObject("RestShapeSpringsForceField", name="canal_tie",
                                        points=[int(i) for i in canal], stiffness=[0.0] * len(canal),
                                        external_rest_shape="@/targets/canal_tgt",
                                        external_points=list(range(len(canal))))
    # (c) vagina.apex follows the nearest cervix surface node (one-way: no reaction on the cervix, so the two
    #     solvers stay decoupled and the thin vagina cannot destabilise the cervix)
    apex = np.asarray(inp["meta"]["vagina"]["node_sets"]["apex"], int)
    ctx["apex"] = apex
    if cfg["apex_attach"] == "follow" and len(apex):
        csurf = np.asarray(inp["meta"]["cervix"]["node_sets"]["surface_nodes"], int)
        d = np.linalg.norm(X0["vagina"][apex][:, None, :] - X0["cervix"][csurf][None, :, :], axis=2)
        pair = csurf[d.argmin(1)]
        ctx["apex_pair"] = pair
        ctx["apex_off"] = X0["vagina"][apex] - X0["cervix"][pair]        # zero force at rest
        ctx["apex_tgt"] = ctx["targets"].addObject("MechanicalObject", name="apex_tgt", template="Vec3d",
                                                   position=X0["vagina"][apex].tolist())
        ctx["nodes"]["vagina"].addObject("RestShapeSpringsForceField", name="apex_follow",
                                         points=[int(i) for i in apex],
                                         stiffness=[float(cfg["k_apex_mN_per_mm"])] * len(apex),
                                         external_rest_shape="@/targets/apex_tgt",
                                         external_points=list(range(len(apex))))


def _add_supports(ctx):
    cfg, inp, X0 = ctx["cfg"], ctx["inp"], ctx["X0"]
    ns = {b: inp["meta"][b]["node_sets"] for b in BODIES}
    # Cardinal ligaments: UNIAXIAL springs from cervix.lateral_os_level to static anchors cardinal_len_mm laterally
    # (CONTRACT 4).  Implemented as a RestShapeSpringsForceField whose target the controller rewrites each step to
    # x + (|A - x| - L) * unit(A - x): the force is then k (|A - x| - L) along the ligament, i.e. a cable of natural
    # length L, and nothing acts back on the anchors.  (A StiffSpringForceField to a massless, solver-less anchor
    # object blew the cervix up by 35 mm in the first rest-state step -- MEASURED, run HSMOKE.)
    lat = np.asarray(ns["cervix"]["lateral_os_level"], int)
    if len(lat) and float(cfg["k_cardinal_mN_per_mm"]) > 0:
        os_x = float(inp["meta"]["cervix"]["node_set_extra"]["internal_os_mm"][0])
        sgn = np.where(X0["cervix"][lat, 0] >= os_x, 1.0, -1.0)
        L = float(cfg["cardinal_len_mm"])
        ctx["lig_idx"] = lat
        ctx["lig_anchor"] = X0["cervix"][lat] + L * np.c_[sgn, np.zeros(len(lat)), np.zeros(len(lat))]
        ctx["supports"].addObject("MechanicalObject", name="lig_anchor", template="Vec3d",
                                  position=ctx["lig_anchor"].tolist())        # static, for the viewer only
        ctx["lig_tgt"] = ctx["targets"].addObject("MechanicalObject", name="lig_tgt", template="Vec3d",
                                                  position=X0["cervix"][lat].tolist())
        ctx["lig_ff"] = ctx["nodes"]["cervix"].addObject(
            "RestShapeSpringsForceField", name="cardinal", points=[int(i) for i in lat],
            stiffness=[float(cfg["k_cardinal_mN_per_mm"])] * len(lat),
            external_rest_shape="@/targets/lig_tgt", external_points=list(range(len(lat))))
        ctx["extra"]["n_cardinal"] = int(len(lat))
    def _pin(node, name, idx, k=None):
        """A 'fixed' node set: hard FixedConstraint, or a stiff (but compliant) spring to the rest position."""
        idx = [int(i) for i in np.asarray(idx, int)]
        if not idx:
            return
        if k is None and cfg["fixed_impl"] == "constraint":
            node.addObject("FixedConstraint", name=name, indices=idx)
        else:
            kk = float(cfg["k_fixed_mN_per_mm"] if k is None else k)
            node.addObject("RestShapeSpringsForceField", name=name, points=idx, stiffness=[kk] * len(idx))

    # vagina inferior (a wall may pin only its OUTER layer there, so the lumen can still open -- see the cfg note)
    vfix = np.asarray(ns["vagina"]["fixed_inferior"], int)
    if cfg.get("vagina_inferior_outer_only") and cfg.get("vagina_model", "solid") == "wall":
        vfix = np.intersect1d(vfix, np.asarray(ns["vagina"]["outer_surface"], int))
        ctx["extra"]["vagina_inferior_outer_only"] = True
    _pin(ctx["nodes"]["vagina"], "fix_inf", vfix,
         None if cfg["vagina_inferior"] == "fixed" else float(cfg["k_vagina_inferior_mN_per_mm"]))
    # bladder / rectum soft supports to rest
    stat = set(ctx.get("static", []))               # a static body has no solver: a force field on it would be inert
    bs = np.asarray(ns["bladder"]["anterior_support"], int)
    if float(cfg["k_bladder_support_mN_per_mm"]) > 0 and "bladder" not in stat:
        ctx["nodes"]["bladder"].addObject("RestShapeSpringsForceField", name="support", points=[int(i) for i in bs],
                                          stiffness=[float(cfg["k_bladder_support_mN_per_mm"])] * len(bs))
    rs = np.asarray(ns["rectum"]["posterior_support"], int)
    if float(cfg["k_rectum_support_mN_per_mm"]) > 0 and "rectum" not in stat:
        ctx["nodes"]["rectum"].addObject("RestShapeSpringsForceField", name="support", points=[int(i) for i in rs],
                                         stiffness=[float(cfg["k_rectum_support_mN_per_mm"])] * len(rs))
    # cut ends fixed
    if cfg["fix_rectum_ends"] and "rectum" not in stat:
        _pin(ctx["nodes"]["rectum"], "fix_ends", ns["rectum"]["fixed_ends"])
    if cfg["fix_sigmoid_ends"] and "sigmoid" not in stat:
        _pin(ctx["nodes"]["sigmoid"], "fix_ends", ns["sigmoid"]["fixed_ends"])
    ctx["extra"].update(n_vagina_fixed=int(len(vfix)), n_bladder_support=int(len(bs)), n_rectum_support=int(len(rs)),
                        n_rectum_ends=int(len(ns["rectum"]["fixed_ends"])),
                        n_sigmoid_ends=int(len(ns["sigmoid"]["fixed_ends"])), fixed_impl=cfg["fixed_impl"])


def _ring_perim(R):
    return float(np.linalg.norm(np.diff(np.r_[R, R[:1]], axis=0), axis=1).sum())


def _wall_diag_setup(ctx):
    """Per-step lumen diagnostics for a WALL run (VAGINA_WALL.md).  Nothing here touches the mechanics.

    The wall's apex follows the cervix, which rides the KINEMATIC corpus, while the device translates along a fixed
    straight line.  Whether the shaft OPENS the lumen or is pressed THROUGH the wall sideways is decided by the
    offset between the lumen centreline and that line, so it is measured every step, per axial station, and written
    into log.jsonl.  MEASURED (run V1B): the lumen slides up to 9.3 mm off the device axis, which is why the wall
    inverts at the contact patch instead of opening."""
    cfg, inp = ctx["cfg"], ctx["inp"]
    if cfg.get("vagina_model", "solid") != "wall":
        return
    w = inp["meta"]["vagina"].get("wall")
    if not w:
        return
    g = np.asarray(w["grid_index"], int)
    rings = [np.nonzero((g[:, 0] == k) & (g[:, 1] == 0))[0] for k in range(int(w["n_axial"]))]
    X0 = ctx["X0"]["vagina"]
    # Device surfaces in the APPLICATOR frame, for the per-station device-to-lumen gap below.  Read once.
    # This is THE measurement that separates a mechanics failure from a geometry one: if the ovoids never come
    # within contact distance of the lumen wall by u = 1, then nothing the solver does could ever open it.
    pf = cfg.get("device_part_files", {})            # measure the rod the scene ACTUALLY loaded, not the default
    parts = {}
    for p in cfg.get("wall_contact_parts", []):
        try:
            Vp, _ = geom.read_obj("%s/%s.obj" % (ctx["inp"]["P"]["applicator"], pf.get(p, p)))
            parts[p] = np.asarray(Vp, float)
        except Exception:
            pass
    n_rd = int(w["n_radial"])
    rings_out = [np.nonzero((g[:, 0] == k) & (g[:, 1] == n_rd))[0] for k in range(int(w["n_axial"]))]
    C0 = np.array([X0[r].mean(0) for r in rings])
    ctx["wall"] = dict(rings=rings, rings_out=rings_out, grid=g, s=np.asarray(w["s"], float),
                       r0=float(w["lumen_r0_mm"]), n_ax=int(w["n_axial"]),
                       perim0=np.array([_ring_perim(X0[r]) for r in rings]),
                       seg0=np.linalg.norm(np.diff(C0, axis=0), axis=1), dev_parts=parts)


def wall_metrics(ctx, X, F, a, Fo=None, R=None):
    """Per axial station: lumen centre offset from the device axis LINE through the current flange F along a,
    the mean lumen radius, the circumferential stretch, the station's axial coordinate along that line, and the
    minimum distance from each contacting device part's surface to that station's lumen ring.

    The last one answers the question the lumen radius alone cannot: whether the ovoids ever geometrically REACH
    the wall.  A gap at or below `contact_mm` means contact is live and the wall is being loaded; a gap that stays
    large all the way to u = 1 would mean the device never touches the lumen, i.e. the geometry (not the solver)
    is what prevents the lumen from opening."""
    W = ctx["wall"]
    C = np.array([X[r].mean(0) for r in W["rings"]])
    q = C - np.asarray(F, float)
    t = q @ np.asarray(a, float)
    off = np.linalg.norm(q - np.outer(t, a), axis=1)
    rad = np.array([float(np.linalg.norm(X[r] - C[k], axis=1).mean()) for k, r in enumerate(W["rings"])])
    per = np.array([_ring_perim(X[r]) for r in W["rings"]])
    # local tangent and outer radius, for the SIGNED containment below
    Tg = np.gradient(C, axis=0)
    Tg /= np.maximum(np.linalg.norm(Tg, axis=1, keepdims=True), 1e-12)
    r_out = np.array([float(np.linalg.norm(X[q] - C[k], axis=1).mean())
                      for k, q in enumerate(W.get("rings_out") or [])]) if W.get("rings_out") else None
    # per-station AXIAL stretch, so the lumen radius can be quoted free of incompressible Poisson necking:
    # an axial stretch lam thins an incompressible tube by 1/sqrt(lam), so r * sqrt(lam) is the radius the lumen
    # would have at unchanged length.  Without this, necking under the cervix's pull reads as "failed to dilate".
    lam = np.ones(len(C))
    if W.get("seg0") is not None and len(C) > 1:
        ls = np.linalg.norm(np.diff(C, axis=0), axis=1) / np.maximum(W["seg0"], 1e-9)
        lam[0], lam[-1] = ls[0], ls[-1]
        if len(C) > 2:
            lam[1:-1] = 0.5 * (ls[:-1] + ls[1:])
    gap, r_dev = {}, {}
    if R is not None:
        half = float(np.median(np.abs(np.diff(W["s"])))) if len(W["s"]) > 1 else 2.0
        for p, Vp in (W.get("dev_parts") or {}).items():
            # ovoid parts ride at the ovoid origin (flange minus the seating lag), tube/shaft at the flange
            org = np.asarray(Fo, float) if (p in OVOID_PARTS and Fo is not None) else np.asarray(F, float)
            Pw = org + Vp @ np.asarray(R, float)
            gap[p] = np.array([float(np.linalg.norm(Pw[None, :, :] - X[r][:, None, :], axis=2).min())
                               for r in W["rings"]])
            # radius of the nearest device vertex about THIS station's own lumen axis: < r_in means the part is
            # inside the lumen, > r_out means it is outside the wall entirely.  A plain gap cannot tell these apart.
            rd = np.full(len(C), np.nan)
            for k in range(len(C)):
                q = Pw - C[k]
                ax = q @ Tg[k]
                m = np.abs(ax) <= half
                if m.any():
                    rd[k] = float(np.linalg.norm(q[m] - np.outer(ax[m], Tg[k]), axis=1).min())
            r_dev[p] = rd
    return off, rad, per / W["perim0"], t, gap, dict(r_out=r_out, r_dev=r_dev, axial_stretch=lam)


def scene_summary(ctx):
    cfg, inp, tgt = ctx["cfg"], ctx["inp"], ctx["tgt"]
    s = dict(bodies={}, delta_mm=tgt["delta_mm"], flange_final=tgt["flange"].round(4).tolist(),
             tube_axis=tgt["tube_axis"].round(6).tolist(), travel_mm=tgt["travel_mm"], u_tip_at_internal_os=tgt["u_ios"],
             corpus_rotation_deg=round(float(geom.rot_angle_deg(tgt["R"])), 3),
             corpus_centroid_shift_mm=round(float(np.linalg.norm(
                 np.array(inp["pose"]["corpus"]["corpus_centroid_target"]) -
                 np.array(inp["pose"]["corpus"]["corpus_centroid_pre"]))), 3) if tgt["is_default"] else None,
             n_sched=len(ctx["sched"]), phases={p: sum(1 for r in ctx["sched"] if r["phase"] == p) for p in "PATD"},
             material=cfg["material"], groups=groups_for(cfg), insertion_axis=tgt["insertion_axis"],
             vagina_model=cfg.get("vagina_model", "solid"),
             wall=(dict(dir=cfg["vagina_wall_dir"], collision=cfg["wall_collision"],
                        contact_parts=list(cfg["wall_contact_parts"]),
                        lumen_r0_mm=inp["meta"]["vagina"].get("wall", {}).get("lumen_r0_mm"),
                        n_axial=inp["meta"]["vagina"].get("wall", {}).get("n_axial"),
                        n_theta=inp["meta"]["vagina"].get("wall", {}).get("n_theta"),
                        material=cfg.get("material_by_body", {}).get("vagina", cfg["material"]),
                        E_kPa=cfg["E_kPa"]["vagina"])
                   if cfg.get("vagina_model", "solid") == "wall" else None),
             static_bodies=list(ctx.get("static", [])), deformable_bodies=list(ctx.get("deformable", DEFORMABLE)),
             n_canal_tie=int(len(ctx.get("canal", []))), n_apex=int(len(ctx.get("apex", []))),
             n_attach=int(len(ctx["iface"])), attach_impl=cfg["attach_impl"], ovoid_mode=cfg["ovoid_mode"])
    s.update(ctx["extra"])
    for b in DEFORMABLE:
        s["bodies"][b] = dict(nodes=int(len(ctx["X0"][b])), E_kPa=cfg["E_kPa"][b], nu=cfg["nu"][b])
    s["bodies"]["corpus"] = dict(nodes=int(len(ctx["X0"]["corpus"])), role="rigid kinematic")
    return s


# ----------------------------------------------------------------------------- controller
class HybridController(Sofa.Core.Controller):
    """Drives the kinematic corpus / device, writes the per-step tie targets, logs, and applies the stop rules."""

    def __init__(self, *args, **kw):
        Sofa.Core.Controller.__init__(self, *args, **kw)
        self.ctx = kw["ctx"]
        self.out_dir = kw.get("out_dir")
        c = self.ctx
        self.cfg = c["cfg"]
        self.sched = c["sched"]
        self.tgt = c["tgt"]
        self.L_iu = float(_param(c["inp"]["app"], "L_iu_mm"))
        self.r_tube = float(_param(c["inp"]["app"], "r_tandem_mm"))
        self.k = 0
        self.settle = 0
        self.done = False
        self.status = "running"
        self.rows = []
        self.deformable = list(c.get("deformable", DEFORMABLE))   # cfg `static_bodies` are obstacles, not solved
        self.X_prev = {b: c["X0"][b].copy() for b in self.deformable}
        self.eng_step = np.full(len(c.get("canal", [])), -1, int)
        self.dx_hist = []
        self.ok_hist = []
        self.lig_ext_max = 0.0
        self.n_canal_active = 0
        self.drift = None
        self.rho = None
        self.t0 = time.perf_counter()
        self._log = open(os.path.join(self.out_dir, "log.jsonl"), "w") if self.out_dir else None

    # ---------------------------------------------------------------- helpers
    def X(self, b):
        return np.array(self.ctx["nodes"][b].dofs.position.value, dtype=float, copy=True)

    def row_now(self):
        if self.k < len(self.sched):
            return self.sched[self.k]
        r = dict(self.sched[-1])
        r["phase"] = "H"
        return r

    def _set_rigid(self, node, R, t):
        node.rig.position.value = rigid_pose(R, t)

    # ---------------------------------------------------------------- step begin
    def onAnimateBeginEvent(self, ev):
        try:
            self._begin()
        except Exception:
            import traceback
            traceback.print_exc()
            raise

    def onAnimateEndEvent(self, ev):
        try:
            self._end()
        except Exception:
            import traceback
            traceback.print_exc()
            raise

    def _begin(self):
        c, cfg = self.ctx, self.cfg
        self.t_step = time.perf_counter()
        r = self.row_now()
        self.cur = r
        a = self.tgt["axis"]
        # (1) kinematic bodies
        Tc = np.asarray(r["T_corpus"], float)
        self._set_rigid(c["corpus"], Tc[:3, :3], Tc[:3, 3])
        self._set_rigid(c["tandem"], c["Rdev"], r["F"])
        self._set_rigid(c["ovoids"], c["Rdev"], r["F"] - float(r["ov_lag"]) * a)
        # (2) velocity scaling (quasi-static relaxation during the settle phase)
        sc = float(cfg["settle_vel_scale"] if r["phase"] == "H" else cfg["motion_vel_scale"])
        if sc != 1.0:
            for b in self.deformable:
                mo = c["nodes"][b].dofs
                mo.velocity.value = (np.array(mo.velocity.value, dtype=float) * sc).tolist()
        # (3) canal dilation ties: target = closest point on the tube surface at the node's own axial level
        self.n_canal_active = 0
        if c.get("canal_ff") is not None and len(c["canal"]):
            at = self.tgt["tube_axis"]                          # the TUBE's own direction, not the path direction
            X = self.X("cervix")[c["canal"]]                    # (they coincide unless insertion_axis="shaft")
            F, rad = np.asarray(r["F"], float), self.r_tube + float(cfg["canal_slack_mm"])
            q = X - F
            s = q @ at
            lat = q - np.outer(s, at)
            d = np.linalg.norm(lat, axis=1)
            inside = (s >= 0.0) & (s <= self.L_iu)              # the tube currently occupies this axial level
            new = inside & (self.eng_step < 0)
            self.eng_step[new] = self.k
            act = inside & (d < rad)                            # dilate only: never pull tissue inwards
            u = np.where(d[:, None] > 1e-9, lat / np.maximum(d, 1e-9)[:, None], _perp(at)[None, :])
            # target = the node pushed radially out to the tube surface, but at most canal_max_offset_mm away from
            # where it is now (bounded tie force; the full dilation is reached over several steps)
            off = np.minimum(rad - d, float(cfg["canal_max_offset_mm"]))
            tgt = np.where(act[:, None], X + np.maximum(off, 0.0)[:, None] * u, X)
            ramp = np.clip((self.k - self.eng_step + 1) / float(cfg["tie_ramp_steps"]), 0.0, 1.0)
            kk = np.where(act, float(cfg["k_canal_mN_per_mm"]) * ramp, 0.0)
            c["canal_tgt"].position.value = tgt.tolist()
            c["canal_ff"].stiffness.value = kk.tolist()
            self.n_canal_active = int(act.sum())
        # (3b) cardinal ligaments: cable of natural length cardinal_len_mm to the static lateral anchors
        if c.get("lig_ff") is not None:
            Xl = self.X("cervix")[c["lig_idx"]]
            dv = c["lig_anchor"] - Xl
            nn = np.linalg.norm(dv, axis=1)
            ext = nn - float(cfg["cardinal_len_mm"])                # > 0: stretched
            c["lig_tgt"].position.value = (Xl + (ext / np.maximum(nn, 1e-9))[:, None] * dv).tolist()
            if cfg["cardinal_tension_only"]:
                c["lig_ff"].stiffness.value = np.where(ext > 0.0, float(cfg["k_cardinal_mN_per_mm"]), 0.0).tolist()
            self.lig_ext_max = float(ext.max())
        # (4) vagina apex follows the cervix
        if c.get("apex_tgt") is not None:
            Xc = self.X("cervix")[c["apex_pair"]] + c["apex_off"]
            c["apex_tgt"].position.value = Xc.tolist()

    # ---------------------------------------------------------------- step end
    def _end(self):
        c, cfg = self.ctx, self.cfg
        r = self.cur
        if not c["tets"]:                               # live-viewer path: cache the topology on the first step
            post_init(c)
        wall = 1000.0 * (time.perf_counter() - self.t_step)
        disp, dx, finite, minvol = {}, 0.0, True, 1.0
        Xv, vrv = None, None
        for b in DEFORMABLE:
            # A cfg `static_bodies` obstacle is never integrated, so it contributes nothing to dx or min_vol_ratio --
            # but it MUST keep its key here: run_hybrid.py (which this module may not edit) prints
            # row["disp"]["bladder"]["umax"] and friends for all five bodies every step, and eval reads the same
            # schema.  Dropping the key raised KeyError: 'bladder' at step 0 (MEASURED, run YSMOKE).
            if b not in self.deformable:
                disp[b] = dict(umax=0.0, umean=0.0, dx=0.0, min_vol_ratio=1.0, static=True)
                continue
            X = self.X(b)
            finite = finite and bool(np.all(np.isfinite(X)))
            u = np.linalg.norm(X - c["X0"][b], axis=1)
            d = float(np.linalg.norm(X - self.X_prev[b], axis=1).max()) if finite else float("nan")
            disp[b] = dict(umax=round(float(u.max()), 4), umean=round(float(u.mean()), 4), dx=round(d, 5))
            dx = max(dx, d if np.isfinite(d) else 1e9)
            if finite and c["tets"].get(b) is not None:
                vr = _tet_vol(X, c["tets"][b]) / c["vol0"][b]
                disp[b]["min_vol_ratio"] = round(float(vr.min()), 4)
                minvol = min(minvol, float(vr.min()))
                if b == "vagina":
                    vrv = vr
            if b == "vagina":
                Xv = X
            self.X_prev[b] = X
        # ---- wall run: where the lumen is relative to the device axis, and where the worst element is
        wallm = None
        if c.get("wall") is not None and finite and Xv is not None:
            a_path = np.asarray(self.tgt["axis"], float)
            Fo = np.asarray(r["F"], float) - float(r["ov_lag"]) * a_path
            off, rad, st, tt, gap, cont = wall_metrics(c, Xv, r["F"], a_path, Fo, c["Rdev"])
            j = int(np.argmin(np.abs(tt)))              # the station at the flange level = where shaft/ovoids are
            wallm = dict(lumen_axis_off_mm=dict(max=round(float(off.max()), 3), mean=round(float(off.mean()), 3),
                                                at_flange=round(float(off[j]), 3), station_at_flange=j),
                         lumen_r_mm=dict(mean=round(float(rad.mean()), 3), max=round(float(rad.max()), 3),
                                         min=round(float(rad.min()), 3)),
                         circ_stretch=dict(max=round(float(st.max()), 4), mean=round(float(st.mean()), 4)),
                         per_station=dict(off_mm=[round(float(v), 2) for v in off],
                                          r_mm=[round(float(v), 2) for v in rad],
                                          stretch=[round(float(v), 3) for v in st]))
            if gap:
                wallm["device_gap_mm"] = {p: dict(min=round(float(v.min()), 3), station_min=int(v.argmin()),
                                                  per_station=[round(float(x), 2) for x in v])
                                          for p, v in gap.items()}
                ov = [v for p, v in gap.items() if p in OVOID_PARTS]
                if ov:
                    mg = np.minimum.reduce(ov)
                    j2 = int(mg.argmin())
                    wallm["ovoid_gap_mm"] = dict(min=round(float(mg.min()), 3), station_min=j2,
                                                 s_mm=round(float(c["wall"]["s"][j2]), 2),
                                                 in_contact=bool(mg.min() <= float(cfg["contact_mm"])),
                                                 per_station=[round(float(x), 2) for x in mg])
            if cont.get("r_dev") and cont.get("r_out") is not None:
                rin, rout, lam = rad, cont["r_out"], cont["axial_stretch"]
                cl = {}
                for p, rd in cont["r_dev"].items():
                    ok = np.isfinite(rd)
                    cl[p] = dict(n_stations_seen=int(ok.sum()),
                                 n_inside_lumen=int(np.sum(ok & (rd < rin))),
                                 n_embedded_in_wall=int(np.sum(ok & (rd >= rin) & (rd <= rout))),
                                 n_outside_wall=int(np.sum(ok & (rd > rout))),
                                 r_dev_mm=[None if not np.isfinite(v) else round(float(v), 2) for v in rd])
                wallm["containment"] = dict(
                    parts=cl, r_in_mm=[round(float(v), 2) for v in rin],
                    r_out_mm=[round(float(v), 2) for v in rout],
                    axial_stretch=[round(float(v), 3) for v in lam],
                    lumen_r_norm_mm=[round(float(v), 3) for v in rin * np.sqrt(np.maximum(lam, 1e-9))],
                    note="r_dev = radius of the nearest device vertex about that station's own lumen axis: "
                         "inside the lumen if r_dev < r_in, embedded in the wall if r_in <= r_dev <= r_out, "
                         "OUTSIDE the wall if r_dev > r_out (a plain gap cannot tell the first from the last). "
                         "lumen_r_norm = r_in * sqrt(axial_stretch) removes incompressible Poisson necking.")
            if vrv is not None:
                i = int(np.argmin(vrv))
                gi = c["wall"]["grid"][c["tets"]["vagina"][i]]
                k = int(round(float(gi[:, 0].mean())))
                wallm["min_vol_tet"] = dict(ratio=round(float(vrv.min()), 4), tet=i, station=k,
                                            s_mm=round(float(c["wall"]["s"][k]), 2),
                                            layer=round(float(gi[:, 1].mean()), 2),
                                            n_tets_lt_0p5=int((vrv < 0.5).sum()))
        gcs = c["root"].gcs
        nrows = int(gcs.currentNumConstraints.value)
        err = float(gcs.currentError.value)
        it = int(gcs.currentIterations.value)
        lam_sum = lam_max = 0.0
        try:                       # Lagrange multipliers of the contact constraints (see README: units caveat)
            lam = np.asarray(gcs.constraintForces.value, dtype=float)
            if lam.size:
                lam_sum = float(np.abs(lam).sum())
                lam_max = float(np.abs(lam).max())
        except Exception:
            pass
        row = dict(step=self.k, phase=r["phase"], u=round(float(r["u"]), 5), corpus_s=round(float(r["s"]), 5),
                   ov_lag_mm=round(float(r["ov_lag"]), 3), wall_ms=round(wall, 1), disp=disp, wall=wallm,
                   dx_max_mm=round(dx, 5), n_contacts=nrows // 3, n_constraint_rows=nrows,
                   constraint_err=round(err, 6), constraint_err_per_contact=round(err / max(1, nrows // 3), 6),
                   constraint_it=it,
                   constraint_converged=bool(it < int(cfg["gcs_max_it"]) and np.isfinite(err)),
                   lambda_abs_sum=round(lam_sum, 4), lambda_max=round(lam_max, 4),
                   n_canal_ties=int(self.n_canal_active), cardinal_max_stretch_mm=round(self.lig_ext_max, 4),
                   min_vol_ratio=round(minvol, 4), finite=bool(finite))
        # ---- convergence (CONTRACT 5) during the settle phase
        if r["phase"] == "H":
            self.settle += 1
            ok = bool(dx < float(cfg["conv_dx_mm"])
                      and row["constraint_err_per_contact"] < float(cfg["conv_constraint_err_per_contact"])
                      and row["constraint_converged"])
            self.dx_hist.append(dx)
            self.ok_hist.append(ok)
            n = int(cfg["conv_steps"])
            if len(self.dx_hist) >= n + 1:
                d = np.array(self.dx_hist[-(n + 1):])
                self.rho = float(np.median(d[1:] / np.maximum(d[:-1], 1e-12)))
                self.drift = float(dx * self.rho / (1 - self.rho)) if self.rho < 1 else float("inf")
            row.update(settle_step=self.settle, settle_ok=ok, dx_ratio=self.rho, drift_est_mm=self.drift)
            if len(self.ok_hist) >= n and all(self.ok_hist[-n:]):
                self.done, self.status = True, "converged"
            elif self.settle >= int(cfg["max_settle_steps"]):
                self.done, self.status = True, "settle_not_converged"
        # ---- stop rules
        if not finite:
            self.done, self.status = True, "abort_nan"
        elif minvol < float(cfg["min_vol_ratio_abort"]):
            self.done, self.status = True, "abort_inverted_tets"
        elif time.perf_counter() - self.t0 > float(cfg["wall_limit_s"]):
            self.done, self.status = True, "abort_wall_time"
        self.rows.append(row)
        if self._log:
            self._log.write(json.dumps(row) + "\n")
            self._log.flush()
        self.k += 1

    def close(self):
        if self._log:
            self._log.close()
            self._log = None


def _perp(a):
    a = geom.unit(a)
    e = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    return geom.unit(np.cross(a, e))


def _tet_vol(X, T):
    A = X[T]
    return np.einsum("ij,ij->i", np.cross(A[:, 1] - A[:, 0], A[:, 2] - A[:, 0]), A[:, 3] - A[:, 0]) / 6.0


def post_init(ctx):
    """Cache the tet connectivity and rest volumes (available only after Simulation.init)."""
    ctx["tets"], ctx["vol0"] = {}, {}
    for b in ctx.get("deformable", DEFORMABLE):
        T = np.array(ctx["nodes"][b].topo.tetrahedra.value, dtype=int, copy=True)
        ctx["tets"][b] = T
        ctx["vol0"][b] = _tet_vol(ctx["X0"][b], T)
    return ctx


def createScene(rootNode):
    """runSofa entry point (reads the run config from env APPSIM_CFG; defaults if unset)."""
    cfg = load_cfg(os.environ.get("APPSIM_CFG"))
    ctx = build_scene(rootNode, cfg)
    rootNode.addObject(HybridController(name="hybrid", ctx=ctx, out_dir=None))
    print("[scene_hybrid] " + json.dumps(scene_summary(ctx), default=str), flush=True)
    return rootNode
