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
ROD_PARTS = ["rod_L", "rod_R", "packing"]   # Stage 3: the two cap rods and the packing cylinder, ride with the ovoids body
OVOID_BODY_PARTS = OVOID_PARTS + ROD_PARTS

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
                                    # THAT REASONING HOLDS ONLY FOR A STRAIGHT-TUBE WALL (vagina_wall.py fornix=
                                    # False).  There the vault has no recess around the portio, so the tube has no
                                    # route out and, excluded from contact, simply passes THROUGH the wall -- the
                                    # "shaft breaking out of the vagina" seen in insertion_Z7S_lumen.mp4 (it is the
                                    # tube above the 24 deg junction, not the shaft: MEASURED, the shaft is inside
                                    # the lumen at every station it reaches).  With a FORNIX build the vault is an
                                    # annulus around the cervix and the tube enters the portio where it should, so
                                    # add "tube" here: contact then makes it DISPLACE the vault wall it touches
                                    # instead of ghosting through it.  Kept out of the default so every Stage-2a
                                    # run reproduces byte-identically; set it in the run cfg.
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
    # --- Stage 3 (2026-09-21): the device as the updated BT applicator label shows it, and a path through the canal
    applicator_dir="applicator",    # hybrid/<dir> holding applicator.json / pose.json / <part>.obj.  "applicator_v3" =
                                    # 28.9 deg junction (MEASURED on the label; the rule then matches the BT device to
                                    # 1.5 deg / 2.3 mm flange / 1.2 mm tip, uterus Dice 0.88), straight tandem rod, two
                                    # cap rods, ring perpendicular to the TUBE (the user's device knowledge).
    insertion_path="rule",          # "rule" (unchanged): the device translates along the path axis with its FINAL
                                    #   orientation from u = 0 and the corpus follows its screw schedule.
                                    # "canal": geom.canal_path -- S1 the tube goes UP THE VAGINA along the vaginal
                                    #   axis with the corpus at rest; S2 a tip-pinned swing onto the final tube axis
                                    #   while the corpus is drawn from rest onto the pose rule's target for the
                                    #   current tube pose (screw fraction w, no jump); end state = device_final.
                                    #   WHY: MEASURED, the preBT os is 16 mm off the vaginal axis line and the lower
                                    #   canal 24 deg from it, so no straight tube lies in the vagina and along the
                                    #   canal at once; and with "rule" the canal was 20-24 mm from the tube until
                                    #   u = 0.8 (G18 canal_d_mm).  The tandem goes up the vagina; the cervix is drawn
                                    #   onto it.  n_approach steps cover S1, n_insert steps S2.
    ovoid_park_mm=None,             # canal mode: the caps (and rods) wait at F_final - park * a_shaft until seating
                                    #   (None = the rule path's travel_mm, i.e. the u = 0 flange as in rule mode)
    device_rods=False,              # load rod_L / rod_R (cap rods, applicator_v3) as parts of the ovoids body
    device_packing=False,           # load packing.obj (applicator_v3 --rods): a kinematic cylinder about the rods,
                                    #   riding with the ring, that the vaginal wall rests on from inside.  It touches
                                    #   the lumen sheet only (list it in wall_contact_parts).  MEASURED: at BT the
                                    #   vagina at the ring's level is the device + packing; in the model the wall was
                                    #   crushed between the cervix bulk and the springs holding the ring (G25).
    wall_inner_contact_organs=[],   # organs the wall's LUMEN sheet collides with, e.g. ["cervix"]: the vault stays
                                    #   physically wrapped around the portio instead of following it by springs only
                                    #   (the user: "maintain the physical contact of the vagina and HR-CTV").  These
                                    #   pairs are OFF during the balloon phase (the driven wall passes through the
                                    #   cervix) and switched on at the hand-off, like the outer sheet's.
    # --- Stage 2b: relax the OARs OUTWARD before the wall is introduced (the "balloon" phase, B)
    n_balloon=0,                    # steps of phase "B" ahead of the pre-settle.  WHY: the distended reference
                                    # wall is built in the collapsed preBT anatomy, so at rest its OUTER sheet
                                    # starts INSIDE the neighbours (MEASURED, tet26: 186 nodes up to 8.5 mm into
                                    # the rectum, 83 up to 5.2 mm into the bladder) -- deeper than alarm_mm, i.e.
                                    # the trapped-vertex regime where contact holds tissue in rather than pushing
                                    # it out.  During B a KINEMATIC copy of the wall's outer sheet (the balloon)
                                    # is driven from a collapsed slit (balloon_start_*) to the wall's rest shape
                                    # and pushes bladder / rectum / sigmoid outward by contact; the wall itself
                                    # is inert meanwhile (its outer sheet ignores the OARs, its rest shape is
                                    # untouched), and at the first non-B step the balloon is switched off and the
                                    # wall's outer contact with the OARs switched on: the tissue is then exactly
                                    # at contact distance from the wall it will be loaded by.  0 = off (byte-
                                    # identical schedules).  Needs deformable OARs (static_bodies without them)
                                    # and wall_outer_exclude WITHOUT rectum, or the relaxation is thrown away.
    balloon_start_a_mm=6.0,         # mm  LR semi-axis of the balloon's start section about the lumen centreline
    balloon_start_b_mm=2.5,         # mm  AP semi-axis (the preBT slit is ~16 x 8 mm mid-vagina; the start must lie
                                    #     inside the label so it begins clear of the OARs)
    balloon_ease="smoothstep",      # "smoothstep" | "linear" ramp of the balloon from start to rest over n_balloon
    balloon_drive_wall=False,       # True: during B the WALL's own nodes are driven kinematically along with the
                                    # balloon, from the same collapsed section (every node's lateral radius scaled
                                    # by the balloon's start/rest ratio at its station and angle) to the rest shape,
                                    # velocities zeroed each step; at w = 1 the wall is exactly at rest with zero
                                    # velocity, so the mechanics from P on are identical to balloon_drive_wall=False.
                                    # PRESENTATION only: the wall's REST state is the distended reference, so without
                                    # this the frames show a vagina open from frame 0 and organs pushed by an
                                    # invisible surface; with it they show the vagina opening from the preBT slit
                                    # and displacing the organs.  The wall's vol ratio is not checked during B.
    balloon_mode="release",         # what happens to the balloon AFTER phase B:
                                    # "release": it is switched off and the wall's own outer sheet takes the OAR
                                    #   contact over (two-way coupling).  MEASURED (G12): the OARs then RECOIL --
                                    #   bladder 5.3 -> 2.4 mm, rectum 8.7 -> 3.3 mm within four steps -- and crush
                                    #   the 10 kPa wall (vol ratio 1.00 -> 0.50).  Nothing in that model holds the
                                    #   distended vagina open against the OARs' elastic push-back; clinically that
                                    #   is the PACKING and the applicator, and the real wall is folded (rugae), not
                                    #   a stretched continuum, so re-collapsing it costs no tissue stress either.
                                    # "follow": the balloon STAYS the OARs' contact surface for the whole run and
                                    #   tracks the wall's outer sheet (positions copied from the wall each step,
                                    #   one step behind): the OARs feel the wall's shape, the wall does not feel
                                    #   the OARs = the packing holds it open.  One-way coupling, stated as such.
                                    #   The wall is then loaded by the cervix (apex springs), the introitus springs
                                    #   and the device on its lumen only.
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
                                    # "rods"  : (canal mode only) the VAGINAL PART (caps + rods + packing) is its own
                                    #           body: it goes up the ROD line (the wall's rest axis) with its FINAL
                                    #           orientation, reaching ovoid_lead_mm below the os level at the end of
                                    #           S1 and closing on the flange with the swing weight w.  MEASURED: riding
                                    #           the tube rigidly (G24, G26) tilts the rods/packing by up to 29 deg
                                    #           inside the vagina; the 18.5 mm packing then needs a 24 mm lumen at
                                    #           the introitus (G26: wall inverted at station 0, abort step 109).
    ovoid_lead_mm=10.0,             # "rods": the flange plane's lead below the os's projected height at the end of S1
    ovoid_close="T",                # "rods": "T" = the lead closes with the swing weight w (the caps reach the os as
                                    #   the tandem seats); "D" = the body keeps the lead below the os through the
                                    #   whole swing (lag = d_os (1 - w) + lead) and seats over the D steps -- the
                                    #   ovoids are pushed up against the cervix only once the tandem is in place.
    ovoid_seat_mm=30.0,             # only used by ovoid_mode="travel" (see build_schedule: for "seat" the ramp
                                    # continues from the lag the caps actually have, a full travel behind the flange)
    tandem_rotation="off",          # "off" = DEFAULT and the CORRECT setting.  Do not use "canal": REFUTED, run G5.
                                    # The idea was that the 24 deg intrauterine tube sweeps sideways through the
                                    # vaginal wall under a fixed orientation, so the device should be pushed up the
                                    # vagina and then angulated onto the canal.  Both halves turned out wrong:
                                    #  (1) THE DEFECT IS NOT REAL at the seated pose.  The "tube outside the lumen"
                                    #      count is dominated by EARLY steps when the tandem has not been inserted
                                    #      yet.  MEASURED on G2, the crossing stations retreat monotonically as the
                                    #      device advances -- 1-25 at u = 0, 16-27 at u = 0.28, 23-27 at u = 0.56,
                                    #      station 27 alone at u = 0.70, and NONE from u = 0.85 on.  That is what
                                    #      insertion looks like, not piercing.
                                    #  (2) The applicator's 24 deg bend is already built into its GEOMETRY (shaft
                                    #      down -z, tube up +z), so with the device frame's z on the tube axis the
                                    #      shaft already lies along the vaginal axis.  Rotating the rigid body to
                                    #      put the TUBE on the vaginal axis therefore drags the SHAFT 24 deg out of
                                    #      alignment and levers its tail against the sprung introitus ring
                                    #      (k 5 mN/mm, the mesh boundary): G5 crushes the vagina at station 0,
                                    #      min vol ratio -0.740 by step 24 (u = 0.275), where G2 sits at 0.994.
                                    #      Proximity is not the cause -- G2's shaft comes CLOSER there (0.23 mm vs
                                    #      0.32) and is fine; the difference is bearing sideways instead of sliding.
    tandem_rotation_onset=-1.0,     # (only meaningful for the refuted "canal" mode) u at which the angulation
                                    # starts; < 0 means u_ios (0.406 here), where the tube tip reaches the internal
                                    # os.  The slerp completes at u = 1, so the final pose is unchanged.
    tandem_oar_contact=True,        # True = DEFAULT and the behaviour of every run so far: the tube and shaft
                                    # collide with bladder / rectum / sigmoid, because GROUPS makes them disjoint.
                                    # For the TUBE that is an artefact, not a model: the back-translated start pose
                                    # puts 47 % of its vertices up to 17.3 mm inside the bladder at u = 0.  Frozen
                                    # OARs absorb this (0 % by u = 1.0); deformable ones are dragged onto the rod
                                    # (47 -> 71 -> 83 %) until the bladder inverts, which is what kills G3 and G4 and
                                    # therefore blocks vaginal expansion, since expansion needs the OARs to move.
                                    # False -> the parts in `tandem_oar_exclude_parts` also carry the OARs' own ids.
    tandem_oar_exclude_parts=["tube"],   # which tandem parts stop colliding with the OARs when the flag is off.
                                    # The SHAFT is deliberately not in this list: it is the vaginal segment, stays
                                    # low, and its contact with the OARs is physical.  Single variable.
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
    canal_tie=True,
    canal_tie_below_mm=0.0,         # "centre" only: canal nodes up to this far BELOW the flange (s < 0, the os
                                    #   region, where there is no tube but the vaginal rod) are tied laterally to
                                    #   the ROD line through the flange instead of being left free.  MEASURED
                                    #   (G18, canal_d_mm): with the canal drawn onto the tube (median 2.4 mm at
                                    #   seating) the vault STILL sat 13 mm off the rod line, because the vault
                                    #   wraps the portio below the flange (h = -12..-5 mm), which the tie never
                                    #   touches and which swings anteriorly with the corpus's 13 deg anteversion.
                                    #   Clinically the external os sits on the applicator axis at the flange,
                                    #   between the caps.  0 = unchanged.
    canal_engage_mm=3.0,            # "centre" only: a canal node ENGAGES (and stays engaged) once it lies within
                                    #   tube radius + this of the tube axis, i.e. once the tube has actually reached
                                    #   it.  MEASURED (G16): engaging by axial span alone, as "dilate" may, pulled
                                    #   12 canal nodes toward the PARKED tube at u = 0 (20+ mm away, anterior),
                                    #   dragged the cervix 7.2 mm during the balloon phase and inverted the rectum.
    canal_tie_mode="dilate",        # "dilate" (unchanged): a canal node inside the tube's axial span is pushed
                                    #   radially OUT to the tube surface if it lies within the tube radius, and
                                    #   never pulled in -- the tie only dilates.  "centre": the same nodes are
                                    #   driven ONTO the tube surface in both directions (bounded per step by
                                    #   canal_max_offset_mm), so the canal -- and with it the portio and the vault
                                    #   that follows it -- stays centred on the tube.  MEASURED (G14, seated
                                    #   pose): with "dilate" the canal nodes near the flange sit 2.5-6.2 mm off
                                    #   the tube line and the portio ends 6-7 mm anterior of it, which drags the
                                    #   vault 9-12 mm off the rod line, and the ring crosses the posterior vault
                                    #   wall.  The tube passes through the external os by definition; "centre"
                                    #   is that fact as a boundary condition.                 # cervix.canal nodes dilated onto the tube (sliding along the axis)
    k_canal_mN_per_mm=200.0,
    canal_slack_mm=0.0,             # tie radius = r_tandem + slack
    canal_max_offset_mm=0.5,        # the tie target is at most this far from the node's CURRENT position, so the tie
                                    # force is bounded by k_canal * canal_max_offset (the dilation is still reached,
                                    # over several steps).  Without the bound the target ratchets out to the full
                                    # tube radius and the force reaches k * 2.18 mm, which collapsed the cervix
                                    # elements around the canal (MEASURED, run H1: min volume ratio 0.92 -> 0.06).
    tie_ramp_steps=3,
    apex_attach="follow",           # "recentre": as "canal", plus the targets shift by w(s) * e, where e is the
                                    #   REST vector from the vault ring's centre to the paired canal nodes' centre
                                    #   and w = the corpus screw parameter s (0 through P/A, smoothstep to 1 over
                                    #   T): zero force at rest, and at the seated pose the vault ring is centred ON
                                    #   the canal.  MEASURED (G21, springs relaxed 0.62 mm): with "canal" the vault
                                    #   is at its targets and those are 10.7 mm off the axis, because at REST the
                                    #   canal sits 13.2 mm off the vaginal axis (x -9.8, y -8.2) inside the vault,
                                    #   so the fixed canal->apex offsets are 18.8 mm vectors; when the tie draws
                                    #   the canal onto the axis the vault is carried rigidly by the same -13 mm and
                                    #   lands 13 mm off on the OPPOSITE side (x +5.8, y +8.2).  The fornices
                                    #   re-centre on the cervix as the tandem straightens it; this is that.
                                    # vagina.apex follows the nearest cervix SURFACE node ("follow") | "canal":
                                    #   the nearest cervix CANAL node | "off".  MEASURED (G19, seated, springs
                                    #   relaxed to 0.16 mm): with "follow" the vault sits exactly at its targets,
                                    #   and the targets ride the cervix's outer surface, which in this tumour-
                                    #   bearing cervix is 15-24 mm from the canal (up to 43 mm) and swings 13 mm
                                    #   anteriorly with the anteversion while the tied canal stays on the axis --
                                    #   so the vault leaves the applicator axis before the caps arrive and its
                                    #   posterior wall ends up ~1 mm from the axis.  The fornices belong around the
                                    #   canal, which the caps define; "canal" keeps the vault centred on it.
                                    # "lift": the apex nodes' AXIAL coordinate (along the wall axis) is prescribed
                                    #   each step = their rest coordinate + the mean axial displacement of the paired
                                    #   cervix nodes (apex_lift_pair): the fornices go up with the os and are FREE
                                    #   laterally, so the ring / packing centre the vault by contact and no spring
                                    #   can squeeze the wall against the cervix bulk (G25) or drag it through the
                                    #   caps (G24).  Written directly into the vagina dofs before the step, as the
                                    #   balloon phase drives wall nodes: SOFA v22.12's ProjectToPlaneConstraint
                                    #   implements neither applyConstraint (assembled matrix) nor
                                    #   projectJacobianMatrix (contact constraints) and errors every step (G29 first
                                    #   try).  MEASURED (G28, apex off): nothing lifts the open-topped wall, the caps
                                    #   leave it through its top (ovoid gap 11 mm above station 27 at u 0.87) and the
                                    #   seated ring sits 26 mm above the vault.
    k_apex_mN_per_mm=20.0,
    apex_lift_pair="canal",         # "lift": pair the apex nodes with the nearest cervix "canal" | "surface_nodes"
    apex_lift_fix=True,             # "lift": FixedConstraint on the written nodes, so the write HOLDS through the
                                    #   solve (dx = 0, zero compliance for contacts) and the vault ring is pinned
                                    #   laterally on the device axis -- the ring + packing at BT.  MEASURED: without
                                    #   it the write is a soft prescription for a STRETCHING shape of this wall: G29
                                    #   (apex nodes only) kept 4.6 of 9.2 mm, G30 (whole wall, linear) 0.35 of
                                    #   10.4 mm, while the balloon's slit-to-tube drive holds because bending is
                                    #   what the 1.2 mm wall barely resists.  Use with apex_lift_profile "top".
    apex_lift_profile="top",        # "lift": "linear" = EVERY wall node's axial coordinate is prescribed, rest
                                    #   coordinate x (1 + lift / H) about the introitus level (H = the apex height):
                                    #   the wall unfolds uniformly from the fixed introitus to the lifted vault, the
                                    #   write is consistent across the wall and nothing snaps back.  "top" = only
                                    #   the apex nodes are written.  MEASURED (G29, "top"): the stretched elements
                                    #   below pull the written nodes back within the implicit step -- the wall
                                    #   retained 4.6 mm of a 9.2 mm lift at w 0.59.
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
    k_rectum_ends_mN_per_mm=None,   # None = the cut ends are "fixed" (fixed_impl: 1e4 mN/mm springs or a hard
    k_sigmoid_ends_mN_per_mm=None,  # constraint); a number = those end slabs are held by springs of THIS
                                    # stiffness instead.  MEASURED (run G11, the first OAR pre-relaxation): the
                                    # rectum's LOWER cut end sits at the level of the mid-lower vagina, ~4 mm from
                                    # the wall's rest outer sheet, and the balloon pushing that region 4 mm sheared
                                    # the 4 mm pinned end slab to inversion (worst tets at s = -49..-51 mm along the
                                    # rectum axis with 1-2 of 4 nodes pinned; vol ratio -0.08) while the actual
                                    # contact patch 25 mm higher, pushed 6-8 mm, was healthy at 0.64.  The anorectal
                                    # junction is anchored to the pelvic floor compliantly, not welded in space.
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
              tube=[9, 10, 11], shaft=[9, 10, 11], ovoid_L=[9, 12], ovoid_R=[9, 12], rod_L=[9, 12], rod_R=[9, 12],
              packing=[3, 4, 9, 11, 12])         # the packing meets the vaginal LUMEN only: not the organs, not the cervix


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
    # The intrauterine tube is excluded from vagina (10), cervix (11) and corpus (9) because it travels inside
    # unmeshed lumens -- but it is DISJOINT from bladder [4] and rectum/sigmoid [3], so it collides with all three.
    # That is the wrong way round, and it is not harmless.  MEASURED (ray-parity, control-validated): the
    # back-translated start pose puts 47 % of the tube's vertices up to 17.3 mm INSIDE the bladder at u = 0, because
    # tube_axis leans anteriorly and "anterior and low" 83 mm back along the path is the bladder, not the canal.
    # With the OARs frozen this is transient and self-correcting (0 % by u = 1.0), which is why G2 survives; with
    # them DEFORMABLE the bladder is dragged onto the rod instead -- 47 % -> 71 % -> 83 % -- until it inverts, which
    # is why G3 and G4 both die on the bladder.  Since expansion needs deformable OARs to make radial room, this
    # single pair blocks the whole route.  Sharing the OARs' own ids removes exactly those three pairs, nothing else
    # (3 is shared by rectum and sigmoid, 4 is the bladder's alone, so no organ-organ pair is touched).
    if not cfg.get("tandem_oar_contact", True):
        for p in cfg.get("tandem_oar_exclude_parts", ["tube"]):
            if p in g:
                g[p] = sorted(set(g[p]) | {3, 4})
    if cfg.get("vagina_model", "solid") != "wall":
        return g
    own = dict(corpus=30, cervix=31, bladder=32, rectum=33, sigmoid=34)
    for b, i in own.items():
        g[b] = g[b] + [i]
    keep = set(cfg.get("wall_contact_parts", []))
    for p in DEVICE_PARTS + ROD_PARTS:
        g[p] = g[p] + [21] + ([] if p in keep else [22])
    touch = set(cfg.get("wall_inner_contact_organs", []))
    g["vagina_inner"] = [22] + sorted(v for b, v in own.items() if b not in touch)
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
    P = dict(P, applicator=P["hybrid"] + "/" + cfg.get("applicator_dir", "applicator"))
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
                is_default=bool(abs(dz - float(pose["default_flange_shift_mm"])) < 1e-9),
                canal=pose.get("insertion_path_canal"))


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
def build_schedule_canal(cfg, tgt):
    """Stage 3 schedule: geom.canal_path for A (S1, corpus at rest) and T (S2, corpus drawn on), caps parked at a
    fixed world point until D, then seated along the shaft (vaginal) axis.  Rows carry F, R_rows, tube_axis,
    T_corpus and ov_pos (world) instead of a scalar lag."""
    cp = tgt.get("canal")
    if not cp:
        raise ValueError("insertion_path='canal' needs pose.json insertion_path_canal (regenerate the applicator)")
    a_v = geom.unit(np.asarray(cp["a_v"], float)); base = np.asarray(cp["base"], float)
    F_fin, a_fin = np.asarray(tgt["flange"], float), geom.unit(tgt["tube_axis"])     # Delta-adjusted final pose
    x_fin = np.asarray(tgt["R_rows"], float)[0]
    L, h_intro, below = float(cp["L_iu_mm"]), float(cp["h_intro_mm"]), float(cp["below_mm"])
    u1 = geom.canal_path(0.0, base, a_v, F_fin, a_fin, L, h_intro, x_fin, below)["u1"]
    park = cfg.get("ovoid_park_mm")
    park = float(tgt["travel_mm"]) if park is None else float(park)
    P_park = F_fin - park * a_v
    # ovoid_mode "travel" in canal mode: the caps (and rods) RIDE THE FLANGE from u = 0 -- the ring is on the tandem
    # while the cervix is pushed up, and it is the ring that keeps the vault centred.  MEASURED (G23, caps parked
    # until D): during T the vault followed the cervix 13-14 mm anteriorly BEFORE the caps arrived, and the caps
    # then rose into a vault that had left their axis (posterior halves 8-10 mm beyond the posterior wall, section
    # verdict "crossing"); at BT the ovoid label is 100 % inside the vagina label and the vault's section is
    # centred on the ring to 1-3 mm.  "seat": the caps wait at P_park and seat during D (G23).
    ride = cfg.get("ovoid_mode", "seat") == "travel"
    rods = cfg.get("ovoid_mode", "seat") == "rods"
    R_ov = np.asarray(tgt["R_rows"], float)              # "rods": the body's fixed frame = the final applicator frame
    d_os = float((F_fin - base) @ a_v)                   # the os's projected rise along the rod line during S2 (25 mm)
    lag1 = d_os + float(cfg.get("ovoid_lead_mm", 10.0))  # "rods": lag below F_fin at the end of S1
    sched = []
    nB = int(cfg.get("n_balloon", 0) or 0)
    for k in range(1, nB + 1):
        sched.append(dict(phase="B", u=0.0, bal=k / float(nB)))
    for k in range(int(cfg["n_presettle"])):
        sched.append(dict(phase="P", u=0.0))
    nA, nT = max(1, int(cfg["n_approach"])), max(1, int(cfg["n_insert"]))
    for k in range(1, nA + 1):
        sched.append(dict(phase="A", u=u1 * k / nA))
    for k in range(1, nT + 1):
        sched.append(dict(phase="T", u=u1 + (1.0 - u1) * k / nT))
    # The corpus rides the FINAL screw (rest -> pose-rule target) with the S2 weight w.  MEASURED (S3SMOKE): tying
    # it to the rule's target for the CURRENT flange dragged it along the flange's ~85 mm S2 travel at up to
    # 5.7 mm / 2.2 deg per step and inverted the cervix (vol ratio -1.6 in one 4.5 mm step); the final screw moves
    # it 23.5 mm / 15 deg over the S2 steps, as smoothly as rule mode did.  The tube's intermediate line and the
    # canal cannot coincide mid-swing in any case -- the canal ties and the cervix|lumen contact take that up.
    for row in sched:
        q = geom.canal_path(row["u"], base, a_v, F_fin, a_fin, L, h_intro, x_fin, below)
        row.update(s=float(q["w"]), F=q["F"], tube_axis=q["a"], R_rows=np.array([q["x"], np.cross(q["a"], q["x"]), q["a"]]),
                   T_corpus=screw_at(tgt["screw"], float(q["w"])), stage=q["stage"])
        if rods:
            if q["stage"] == "S1":
                lag = park + (lag1 - park) * min(1.0, row["u"] / max(1e-9, u1))
            elif cfg.get("ovoid_close", "T") == "D":
                lag = d_os * (1.0 - float(q["w"])) + (lag1 - d_os)
            else:
                lag = lag1 * (1.0 - float(q["w"]))
            row.update(ov_pos=F_fin - lag * a_v, ov_lag=float(lag), ov_R_rows=R_ov.copy())
        else:
            row.update(ov_pos=(np.asarray(q["F"], float).copy() if ride else P_park.copy()), ov_lag=(0.0 if ride else float(park)))
    nD = max(0, int(cfg["n_seat"]))
    last = sched[-1]
    lag_D0 = float(last["ov_lag"]) if rods else park            # "rods"/"D": the lead still open at the end of T
    for k in range(1, nD + 1):
        lag = 0.0 if ride else lag_D0 * (1.0 - k / float(nD))
        sched.append(dict(phase="D", u=1.0, s=1.0, F=last["F"], tube_axis=last["tube_axis"], R_rows=last["R_rows"],
                          T_corpus=last["T_corpus"], stage="D", ov_pos=F_fin - lag * a_v, ov_lag=float(lag)))
        if rods:
            sched[-1]["ov_R_rows"] = R_ov.copy()
    return sched


def build_schedule(cfg, tgt):
    """Per-step device / corpus kinematics.  u in [0, 1] is the pose-rule path parameter (pose.json insertion_path):
    the device flange is F(u) = F_final - (1 - u) * travel * axis, and the corpus follows the screw motion
    s(u) = smoothstep((u - u_ios) / (1 - u_ios)).  Phases: P (pre-settle), A (approach), T (insertion),
    D (ovoid seating), H (settle)."""
    if cfg.get("insertion_path", "rule") == "canal":
        return build_schedule_canal(cfg, tgt)
    F1, a, D, u_ios = tgt["flange"], tgt["axis"], tgt["travel_mm"], tgt["u_ios"]

    def F_of(u):
        return F1 - (1.0 - u) * D * a

    def s_of(u):
        return geom.smoothstep((u - u_ios) / max(1e-9, 1.0 - u_ios)) if u > u_ios else 0.0

    sched = []
    nB = int(cfg.get("n_balloon", 0) or 0)
    for k in range(1, nB + 1):                         # B: device parked at u = 0, corpus at rest, balloon k/nB
        sched.append(dict(phase="B", u=0.0, s=0.0, ov_lag=None, bal=k / float(nB)))
    for k in range(int(cfg["n_presettle"])):
        sched.append(dict(phase="P", u=0.0, s=0.0, ov_lag=None))
    nA = max(1, int(cfg["n_approach"]))
    for k in range(1, nA + 1):
        sched.append(dict(phase="A", u=u_ios * k / nA, s=0.0, ov_lag=None))
    nT = max(1, int(cfg["n_insert"]))
    for k in range(1, nT + 1):
        u = u_ios + (1.0 - u_ios) * k / nT
        sched.append(dict(phase="T", u=u, s=s_of(u), ov_lag=None))
    for row in sched:                                  # ovoid lag during P/A/T depends on the mode -- resolve it
        if row["ov_lag"] is None:                      # BEFORE the seating ramp, which must continue from it
            if cfg["ovoid_mode"] == "travel":
                row["ov_lag"] = 0.0
            else:                                      # "seat"/"off": parked at the u = 0 flange, below the introitus
                row["ov_lag"] = float((F_of(row["u"]) - F_of(0.0)) @ a)
    # The seating ramp starts from the lag the caps ACTUALLY have at the end of T, not from `ovoid_seat_mm`.
    # Parked at F(0) the caps sit a full travel (D mm) behind the flange, so the old ramp -- which started at
    # ovoid_seat_mm = 30 -- made the lag jump D -> 30 in ONE step and teleported them ~53 mm into the middle of the
    # vagina.  That is the H4/H5 trapped-vertex failure, and it explains why the interpenetration there was
    # "identical to 3 decimals whether the seating takes 8 or 20 steps": it was never the step count, it was the
    # first step.  `ovoid_seat_mm` now only applies to "travel", where the caps ride with the flange and a genuine
    # trailing offset is what is wanted.
    nD = max(0, int(cfg["n_seat"]))
    lag0 = float(sched[-1]["ov_lag"]) if sched else 0.0
    if cfg["ovoid_mode"] == "travel":
        lag0 = float(cfg["ovoid_seat_mm"])
    for k in range(1, nD + 1):
        sched.append(dict(phase="D", u=1.0, s=1.0, ov_lag=lag0 * (1.0 - k / float(nD))))
    for row in sched:
        row["F"] = F_of(row["u"])
        row["T_corpus"] = screw_at(tgt["screw"], row["s"])

    # ---- per-step device ORIENTATION (Stage 2b).  The intrauterine tube sits `angle_deg` = 24 deg off the vaginal
    # axis, so a FIXED orientation sweeps it sideways through the vaginal wall for the whole travel: MEASURED, the
    # tube is inside the lumen at only 26 % of (step, station) pairs and outside by up to 24.9 mm.  Clinically the
    # tandem is pushed up the VAGINA and then angulated onto the cervical canal as the tip enters the os, so the
    # device axis slerps from the shaft (vaginal) axis onto the tube axis starting at `onset`.  MEASURED on the rest
    # lumen: onset >= 0.4 gives 100 % containment, and the natural onset is u_ios (0.406), the step at which the tip
    # reaches the internal os.  The slerp completes at u = 1, so the FINAL pose is bit-identical
    # (tip = F1 + L_iu * a_tube) and the BT-validated placement is untouched.
    #   "off"   (default) -> a(u) = a_tube for every row, i.e. exactly the previous constant orientation
    #   "canal"           -> a(u) = slerp(shaft axis, tube axis, smoothstep((u - onset) / (1 - onset)))
    rot = cfg.get("tandem_rotation", "off")
    a_tube = geom.unit(tgt["tube_axis"])
    a_entry = geom.unit(tgt["axis"]) if rot == "canal" else a_tube
    onset = float(cfg.get("tandem_rotation_onset", -1.0))
    if onset < 0.0:
        onset = float(u_ios)
    x_ref = np.asarray(tgt["R_rows"], float)[0]
    a_prev = x_prev = None
    for row in sched:
        t = 0.0 if row["u"] <= onset else (row["u"] - onset) / max(1e-9, 1.0 - onset)
        a_u = geom.unit(geom.slerp(a_entry, a_tube, geom.smoothstep(t)))
        # parallel transport of the applicator x axis, as geom.pose_path does, so the device does not spin about its
        # own axis while it angulates
        x_u = geom.ortho(x_ref, a_u) if a_prev is None else geom.ortho(geom.rot_between(a_prev, a_u) @ x_prev, a_u)
        row["R_rows"] = np.array([x_u, np.cross(a_u, x_u), a_u])
        row["tube_axis"] = a_u
        a_prev, x_prev = a_u, x_u
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
    ov_parts = OVOID_PARTS + (["rod_L", "rod_R"] if cfg.get("device_rods") else []) + (["packing"] if cfg.get("device_packing") else [])
    dgrp = {k: list(grp[k]) for k in DEVICE_PARTS + ROD_PARTS}
    if cfg.get("vagina_model", "solid") != "wall":       # (the wall has a real lumen: `wall_contact_parts` governs
                                                        #  which device parts touch it, see groups_for)
        if cfg["tandem_lumen_contact"]:                 # let the tandem collide with the cervix and the vagina
            for p in TANDEM_PARTS:
                dgrp[p] = [g for g in dgrp[p] if g not in (set(grp["vagina"]) | set(grp["cervix"]))]
        if cfg["ovoid_vagina_contact"]:                 # let the ovoids collide with the vagina
            for p in OVOID_BODY_PARTS:
                dgrp[p] = [g for g in dgrp[p] if g not in set(grp["vagina"])]
    if cfg["ovoid_mode"] == "off":
        for p in OVOID_BODY_PARTS:
            dgrp[p] = []
    sched = build_schedule(cfg, tgt)
    F0 = sched[0]["F"]
    Rdev = tgt["R_rows"].T                              # columns = applicator x, y, z in preBT world
    pf = cfg.get("device_part_files", {})               # cfg override of a part's OBJ (e.g. the straight shaft)

    def _pobj(p):
        return "%s/%s.obj" % (P["applicator"], pf.get(p, p))

    R0 = np.asarray(sched[0]["R_rows"], float).T        # step-0 orientation; == Rdev when tandem_rotation="off"
    tandem = add_rigid_parts(root, "tandem", [(p, _pobj(p)) for p in TANDEM_PARTS],
                             dgrp, dcol, rigid_pose(R0, F0))
    ov_pose = np.asarray(sched[0]["ov_pos"], float) if "ov_pos" in sched[0] else F0 - sched[0]["ov_lag"] * tgt["axis"]
    R_ov0 = ovoid_R_rows(sched[0]).T if "R_rows" in sched[0] else R0      # "rods": the body's own (final) frame
    dcol.update(rod_L=dcol["shaft"], rod_R=dcol["shaft"], packing=[0.9, 0.9, 0.8, 0.3])
    ovoids = add_rigid_parts(root, "ovoids", [(p, _pobj(p)) for p in ov_parts],
                             dgrp, dcol, rigid_pose(R_ov0, ov_pose))

    # ---- couplings and supports
    ctx = dict(root=root, cfg=cfg, inp=inp, tgt=tgt, sched=sched, X0=X0, nodes=nodes, corpus=corp,
               tandem=tandem, ovoids=ovoids, iface=iface, Rdev=Rdev, targets=tg, supports=sup, extra={},
               deformable=defo, static=stat)
    _add_couplings(ctx)
    _add_supports(ctx)
    _wall_diag_setup(ctx)                               # read-only per-step lumen diagnostics (wall runs only)
    _add_balloon(ctx)                                   # Stage 2b OAR pre-relaxation (cfg n_balloon > 0, wall runs)
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
    if cfg["apex_attach"] in ("follow", "canal", "recentre") and len(apex):
        csurf = np.asarray(inp["meta"]["cervix"]["node_sets"]["canal" if cfg["apex_attach"] in ("canal", "recentre")
                                                             else "surface_nodes"], int)
        d = np.linalg.norm(X0["vagina"][apex][:, None, :] - X0["cervix"][csurf][None, :, :], axis=2)
        pair = csurf[d.argmin(1)]
        ctx["apex_pair"] = pair
        ctx["apex_off"] = X0["vagina"][apex] - X0["cervix"][pair]        # zero force at rest
        ctx["apex_e"] = (X0["cervix"][pair].mean(0) - X0["vagina"][apex].mean(0)
                         if cfg["apex_attach"] == "recentre" else np.zeros(3))
        ctx["extra"]["apex_recentre_mm"] = np.round(ctx["apex_e"], 3).tolist()
        ctx["apex_tgt"] = ctx["targets"].addObject("MechanicalObject", name="apex_tgt", template="Vec3d",
                                                   position=X0["vagina"][apex].tolist())
        ctx["nodes"]["vagina"].addObject("RestShapeSpringsForceField", name="apex_follow",
                                         points=[int(i) for i in apex],
                                         stiffness=[float(cfg["k_apex_mN_per_mm"])] * len(apex),
                                         external_rest_shape="@/targets/apex_tgt",
                                         external_points=list(range(len(apex))))
    elif cfg["apex_attach"] == "lift" and len(apex):
        cset = np.asarray(inp["meta"]["cervix"]["node_sets"][cfg.get("apex_lift_pair", "canal")], int)
        d = np.linalg.norm(X0["vagina"][apex][:, None, :] - X0["cervix"][cset][None, :, :], axis=2)
        ctx["apex_pair"] = cset[d.argmin(1)]
        a_w = geom.unit(np.asarray(inp["meta"]["vagina"]["axis"]["axis"], float))
        gi = np.asarray(inp["meta"]["vagina"]["wall"]["grid_index"], int)
        g = gi[apex, 0]                                                               # station of each apex node
        h_all = X0["vagina"] @ a_w
        h_intro = float(h_all[gi[:, 0] == 0].mean())                                  # the introitus level
        if cfg.get("apex_lift_profile", "linear") == "linear":
            idx = np.arange(len(X0["vagina"]))
            H_top = float(h_all[apex].mean() - h_intro)
            ctx["apex_lift"] = dict(idx=idx, h0=h_all[idx] - h_intro, a=a_w, h_intro=h_intro, H=H_top, profile="linear")
        else:
            ctx["apex_lift"] = dict(idx=apex, h0=h_all[apex] - h_intro, a=a_w, h_intro=h_intro, H=None, profile="top")
        ctx["apex_axis"] = a_w
        if cfg.get("apex_lift_fix", True):
            ctx["nodes"]["vagina"].addObject("FixedConstraint", name="apex_fix", fixAll=False,
                                             indices=[int(i) for i in ctx["apex_lift"]["idx"]])
        ctx["extra"]["apex_lift_fix"] = bool(cfg.get("apex_lift_fix", True))
        ctx["extra"]["apex_lift_stations"] = [dict(station=int(st), n=int((g == st).sum()))
                                              for st in sorted(set(int(v) for v in g))]
        ctx["extra"]["apex_lift_profile"] = dict(profile=ctx["apex_lift"]["profile"], n_nodes=int(len(ctx["apex_lift"]["idx"])),
                                                 H_mm=(round(ctx["apex_lift"]["H"], 3) if ctx["apex_lift"]["H"] else None))


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
        _pin(ctx["nodes"]["rectum"], "fix_ends", ns["rectum"]["fixed_ends"], cfg.get("k_rectum_ends_mN_per_mm"))
    if cfg["fix_sigmoid_ends"] and "sigmoid" not in stat:
        _pin(ctx["nodes"]["sigmoid"], "fix_ends", ns["sigmoid"]["fixed_ends"], cfg.get("k_sigmoid_ends_mN_per_mm"))
    ctx["extra"].update(n_vagina_fixed=int(len(vfix)), n_bladder_support=int(len(bs)), n_rectum_support=int(len(rs)),
                        k_rectum_ends=cfg.get("k_rectum_ends_mN_per_mm"), k_sigmoid_ends=cfg.get("k_sigmoid_ends_mN_per_mm"),
                        n_rectum_ends=int(len(ns["rectum"]["fixed_ends"])),
                        n_sigmoid_ends=int(len(ns["sigmoid"]["fixed_ends"])), fixed_impl=cfg["fixed_impl"])


def _ring_perim(R):
    return float(np.linalg.norm(np.diff(np.r_[R, R[:1]], axis=0), axis=1).sum())


def ovoid_R_rows(row):
    """Applicator-frame rows of the OVOIDS body for a schedule row: its own frame when the vaginal part travels
    separately from the tube (ovoid_mode "rods": ov_R_rows), else the tandem's."""
    return np.asarray(row.get("ov_R_rows", row["R_rows"]), float)


def ovoid_origin(row, a_path):
    """World origin of the ovoids body for a schedule row: an explicit park/seat point (canal mode) or the flange
    minus the scalar lag along the path axis (rule mode)."""
    if row.get("ov_pos") is not None:
        return np.asarray(row["ov_pos"], float)
    return np.asarray(row["F"], float) - float(row["ov_lag"]) * np.asarray(a_path, float)


def _set_group(model, groups):
    """REPLACE a collision model's `group` at runtime and return what it reads back as.

    MEASURED (probe_groups.py, SOFA v22.12 / SofaPython3): assigning a python LIST to this Data<std::set<int>>
    APPENDS ({1} -> [2] -> {1, 2} -> [3, 4] -> {1, 2, 3, 4}); only the space-separated STRING form replaces
    ("5 6" -> {5, 6}, "" -> {}).  The broad phase honours the change on the next step (contacts 2 -> 0 -> 2)."""
    model.group.value = " ".join(str(int(g)) for g in groups)
    return model.findData("group").getValueString()


def _add_balloon(ctx):
    """The OAR pre-relaxation balloon (cfg n_balloon): a kinematic copy of the wall's OUTER sheet.

    Nodes = the wall's outer_surface, triangles = outer_triangles, positions driven by the controller between
    X_start (the same nodes squeezed onto a `balloon_start_a/b_mm` ellipse about the lumen centreline, never
    beyond their rest radius) and X_end (their rest positions).  Collision group: shares an id with the device
    (9), the wall's two sheets (21, 22) and the cervix (11) / corpus (9), so it touches bladder, rectum and
    sigmoid ONLY.  Nothing here is integrated (moving=True, simulated=False, as the device parts)."""
    cfg, inp, X0 = ctx["cfg"], ctx["inp"], ctx["X0"]
    nB = int(cfg.get("n_balloon", 0) or 0)
    if nB <= 0 or cfg.get("vagina_model", "solid") != "wall":
        return
    meta = inp["meta"]["vagina"]
    w = meta["wall"]
    outer = np.asarray(meta["node_sets"]["outer_surface"], int)
    tri = np.asarray(meta["node_set_extra"]["outer_triangles"], int)
    g = np.asarray(w["grid_index"], int)
    C = np.asarray(w["centreline"], float)
    Tg = np.asarray(w["tangent"], float)
    U = np.asarray(w["frame_u"], float)
    V = np.asarray(w["frame_v"], float)
    phi = np.radians(np.asarray((w.get("lumen_profile") or {}).get("lr_angle_in_frame_deg") or np.zeros(len(C)), float))
    Xo = X0["vagina"][outer]
    k = g[outer, 0]
    q = Xo - C[k]
    ax = np.einsum("ij,ij->i", q, Tg[k])
    lat = q - ax[:, None] * Tg[k]
    rho = np.linalg.norm(lat, axis=1)
    e = lat / np.maximum(rho, 1e-9)[:, None]
    th = np.arctan2(np.einsum("ij,ij->i", lat, V[k]), np.einsum("ij,ij->i", lat, U[k]))
    d = th - phi[k]
    A, B = float(cfg["balloon_start_a_mm"]), float(cfg["balloon_start_b_mm"])
    r0 = (A * B) / np.sqrt((B * np.cos(d)) ** 2 + (A * np.sin(d)) ** 2)
    r0 = np.minimum(r0, rho)                            # never start OUTSIDE the rest outer sheet
    X_start = C[k] + ax[:, None] * Tg[k] + r0[:, None] * e
    rem = -np.ones(len(X0["vagina"]), np.int64)
    rem[outer] = np.arange(len(outer))
    grp = groups_for(cfg)
    g_on = sorted({9, 10, 11, 12, 21, 22})              # disjoint from bladder [4,32], rectum [3,33], sigmoid [3,34]
    g_off = sorted(set(g_on) | {1, 2, 3, 4})            # shares an id with everything: no contact at all
    nd = ctx["root"].addChild("balloon")
    nd.addObject("MechanicalObject", name="mo", template="Vec3d", position=X_start.tolist())
    nd.addObject("MeshTopology", name="mt", position=X_start.tolist(), triangles=rem[tri].tolist())
    models = [nd.addObject("TriangleCollisionModel", name="tri", group=g_on, moving=True, simulated=False),
              nd.addObject("LineCollisionModel", name="lin", group=g_on, moving=True, simulated=False),
              nd.addObject("PointCollisionModel", name="pnt", group=g_on, moving=True, simulated=False)]
    vis = nd.addChild("vis")
    vis.addObject("OglModel", name="ogl", color=[0.95, 0.75, 0.35, 0.35])
    vis.addObject("IdentityMapping", name="vm", input="@../mo", output="@ogl")
    # the wall's own outer models: they ignore the OARs while the balloon works, and take over afterwards
    vn = ctx["nodes"]["vagina"]
    side = vn.outer if cfg["wall_collision"] == "split" else vn.surf
    wall_models = [side.tri, side.lin, side.pnt]
    wall_on = list(grp["vagina_outer"] if cfg["wall_collision"] == "split" else grp["vagina"])
    wall_off = sorted(set(wall_on) | {3, 4})
    own = dict(corpus=30, cervix=31, bladder=32, rectum=33, sigmoid=34)
    touch = [b for b in cfg.get("wall_inner_contact_organs", []) if b in own]
    if touch and cfg["wall_collision"] == "split":     # the driven wall passes through these organs during B
        inner_on = list(grp["vagina_inner"])
        wall_models += [vn.inner.tri, vn.inner.lin, vn.inner.pnt]
        wall_on_list = [wall_on] * 3 + [inner_on] * 3
        wall_off_list = [wall_off] * 3 + [sorted(set(inner_on) | {own[b] for b in touch})] * 3
    else:
        wall_on_list, wall_off_list = [wall_on] * 3, [wall_off] * 3
    W_start = None
    if cfg.get("balloon_drive_wall", False):
        # every wall node: lateral radius scaled by (balloon start radius / rest outer radius) at its station and
        # angle, the outer rest radius interpolated round that station's outer ring; axial coordinate kept
        Xw = X0["vagina"]
        kw = g[:, 0]
        qw = Xw - C[kw]
        axw = np.einsum("ij,ij->i", qw, Tg[kw])
        latw = qw - axw[:, None] * Tg[kw]
        rhow = np.linalg.norm(latw, axis=1)
        ew = latw / np.maximum(rhow, 1e-9)[:, None]
        thw = np.arctan2(np.einsum("ij,ij->i", latw, V[kw]), np.einsum("ij,ij->i", latw, U[kw]))
        scale = np.ones(len(Xw))
        for kk in range(len(C)):
            ring = np.nonzero(k == kk)[0]                 # outer nodes of this station (indices into `outer`)
            if len(ring) < 3:
                continue
            o = np.argsort(th[ring])
            t_r, rho_r, r0_r = th[ring][o], rho[ring][o], r0[ring][o]
            t_p = np.r_[t_r[-1] - 2 * np.pi, t_r, t_r[0] + 2 * np.pi]      # periodic
            sel = np.nonzero(kw == kk)[0]
            rho_o = np.interp(thw[sel], t_p, np.r_[rho_r[-1], rho_r, rho_r[0]])
            r0_o = np.interp(thw[sel], t_p, np.r_[r0_r[-1], r0_r, r0_r[0]])
            scale[sel] = np.clip(r0_o / np.maximum(rho_o, 1e-9), 0.0, 1.0)
        W_start = C[kw] + axw[:, None] * Tg[kw] + (rhow * scale)[:, None] * ew
    ctx["balloon"] = dict(node=nd, models=models, X_start=X_start, X_end=Xo.copy(), idx=outer, g_on=g_on,
                          g_off=g_off, wall_models=wall_models, wall_on=wall_on, wall_off=wall_off,
                          wall_on_list=wall_on_list, wall_off_list=wall_off_list,
                          active=False, released=False, w=0.0, n=nB, W_start=W_start, W_cur=None,
                          drive=bool(W_start is not None))
    ctx["extra"]["balloon"] = dict(n_steps=nB, nodes=int(len(outer)), triangles=int(len(tri)),
                                   start_semi_axes_mm=[A, B],
                                   start_radius_mm=dict(min=round(float(r0.min()), 3), max=round(float(r0.max()), 3)),
                                   travel_mm=dict(max=round(float(np.linalg.norm(Xo - X_start, axis=1).max()), 3),
                                                  mean=round(float(np.linalg.norm(Xo - X_start, axis=1).mean()), 3)),
                                   groups=dict(balloon_on=g_on, balloon_off=g_off, wall_outer_on=wall_on,
                                               wall_outer_off=wall_off))


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


def wall_metrics(ctx, X, F, a, Fo=None, R=None, R_ov=None):
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
            ovb = p in OVOID_BODY_PARTS and Fo is not None
            org = np.asarray(Fo, float) if ovb else np.asarray(F, float)
            Pw = org + Vp @ np.asarray((R_ov if (ovb and R_ov is not None) else R), float)
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
             n_sched=len(ctx["sched"]), phases={p: sum(1 for r in ctx["sched"] if r["phase"] == p) for p in "BPATD"},
             insertion_path=cfg.get("insertion_path", "rule"), applicator_dir=cfg.get("applicator_dir", "applicator"),
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
        Rr = np.asarray(r["R_rows"], float)             # PER-STEP orientation (constant when rotation is "off")
        self._set_rigid(c["tandem"], Rr.T, r["F"])      # rigid_pose wants columns = applicator axes, hence .T
        self._set_rigid(c["ovoids"], ovoid_R_rows(r).T, ovoid_origin(r, a))   # its own frame in "rods" mode
        if c.get("balloon") is not None:
            self._balloon(r)
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
            centre = cfg.get("canal_tie_mode", "dilate") == "centre"
            below = float(cfg.get("canal_tie_below_mm", 0.0) or 0.0)
            if centre and below > 0.0:
                # the os region below the flange: lateral offset from the ROD line (the caps' axis), not the tube
                ar = np.asarray(self.tgt["axis"], float)
                sr = q @ ar
                latr = q - np.outer(sr, ar)
                dr = np.linalg.norm(latr, axis=1)
                low = (sr < 0.0) & (sr >= -below)
                inside = inside | low
                lat = np.where(low[:, None], latr, lat)
                d = np.where(low, dr, d)
                s = np.where(low, sr, s)
            if centre:                                          # engage only once the tube has REACHED the node
                new = inside & (d < rad + float(cfg.get("canal_engage_mm", 3.0))) & (self.eng_step < 0)
            else:
                new = inside & (self.eng_step < 0)
            self.eng_step[new] = self.k
            act = (inside & (self.eng_step >= 0)) if centre else (inside & (d < rad))   # dilate: never pull inwards
            u = np.where(d[:, None] > 1e-9, lat / np.maximum(d, 1e-9)[:, None], _perp(at)[None, :])
            # (below the flange `u` is radial about the rod line; above it about the tube line -- both lateral)
            # target = the node moved radially to the tube surface, but at most canal_max_offset_mm away from
            # where it is now (bounded tie force; the full move is reached over several steps).  "dilate" moves
            # outward only; "centre" also pulls a node that sits beyond the tube radius back onto it.
            cap = float(cfg["canal_max_offset_mm"])
            off = np.clip(rad - d, -cap, cap) if centre else np.maximum(np.minimum(rad - d, cap), 0.0)
            tgt = np.where(act[:, None], X + off[:, None] * u, X)
            ramp = np.clip((self.k - self.eng_step + 1) / float(cfg["tie_ramp_steps"]), 0.0, 1.0)
            kk = np.where(act, float(cfg["k_canal_mN_per_mm"]) * ramp, 0.0)
            c["canal_tgt"].position.value = tgt.tolist()
            c["canal_ff"].stiffness.value = kk.tolist()
            self.n_canal_active = int(act.sum())
            # canal-to-tube distance of the nodes inside the tube's span: the number that says whether the tube is
            # IN the canal or beside it (MEASURED G17: the centring tie engaged 0 nodes until u = 0.81 because
            # every canal node was > tube radius + 3 mm from the tube axis until then)
            self.canal_d = (dict(n_in_span=int(inside.sum()), min=round(float(d[inside].min()), 3),
                                 median=round(float(np.median(d[inside])), 3), max=round(float(d[inside].max()), 3))
                            if inside.any() else dict(n_in_span=0))
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
            if cfg["apex_attach"] == "recentre":                 # vault ring -> centred on the canal as s -> 1
                Xc = Xc + float(r["s"]) * c["apex_e"][None, :]
            c["apex_tgt"].position.value = Xc.tolist()
        if c.get("apex_lift"):                                    # "lift": the apex rises with the paired cervix nodes
            L = c["apex_lift"]
            pr, a_w = c["apex_pair"], L["a"]
            lift = float(((self.X("cervix")[pr] - c["X0"]["cervix"][pr]) @ a_w).mean())
            mo = c["nodes"]["vagina"].dofs
            X = np.array(mo.position.value, dtype=float, copy=True)
            Xi = X[L["idx"]]
            if L["profile"] == "linear":                          # uniform unfolding: introitus 0 -> apex `lift`
                h_t = L["h_intro"] + L["h0"] * (1.0 + lift / max(1e-9, L["H"]))
            else:
                h_t = L["h_intro"] + L["h0"] + lift
            Xi += (h_t - Xi @ a_w)[:, None] * a_w[None, :]         # prescribe the AXIAL coordinate only
            X[L["idx"]] = Xi
            mo.position.value = X.tolist()
            self.apex_lift = lift

    def _balloon(self, r):
        """Phase B: drive the balloon from its start section to the wall's rest outer sheet; on the first step
        after B switch it off and let the wall's own outer sheet take the OAR contact over."""
        b, cfg = self.ctx["balloon"], self.cfg
        if r["phase"] == "B":
            if not b["active"]:
                b["active"] = True
                b["wall_group_seen"] = [_set_group(m, g) for m, g in zip(b["wall_models"], b["wall_off_list"])]
            t = float(r["bal"])
            w = float(geom.smoothstep(t)) if cfg.get("balloon_ease", "smoothstep") == "smoothstep" else t
            b["w"] = w
            b["node"].mo.position.value = (b["X_start"] + w * (b["X_end"] - b["X_start"])).tolist()
            if b["drive"]:
                X0w = self.ctx["X0"]["vagina"]
                b["W_cur"] = b["W_start"] + w * (X0w - b["W_start"])
                if w >= 1.0:
                    b["W_cur"] = X0w.copy()             # bit-exact rest at the end of B
                mo = self.ctx["nodes"]["vagina"].dofs
                mo.position.value = b["W_cur"].tolist()
                mo.velocity.value = np.zeros_like(b["W_cur"]).tolist()
        elif b["active"] and cfg.get("balloon_mode", "release") == "follow":
            # the packing: the balloon keeps carrying the OAR contact and copies the wall's outer sheet (end of
            # the previous step); the wall's own outer models stay excluded from the OARs for the whole run --
            # but the LUMEN sheet's organ pairs (wall_inner_contact_organs, e.g. the cervix) come on now
            b["w"] = 1.0
            b["released"] = False
            b["node"].mo.position.value = self.X("vagina")[b["idx"]].tolist()
            if not b.get("inner_on_done") and len(b["wall_models"]) > 3:
                b["inner_on_done"] = True
                b["wall_group_seen"] = [_set_group(m, g) for m, g in zip(b["wall_models"][3:], b["wall_on_list"][3:])]
        elif b["active"] and not b["released"]:
            b["released"] = True
            b["w"] = 1.0
            b["node"].mo.position.value = b["X_end"].tolist()
            b["balloon_group_seen"] = [_set_group(m, b["g_off"]) for m in b["models"]]
            b["wall_group_seen"] = [_set_group(m, g) for m, g in zip(b["wall_models"], b["wall_on_list"])]

    # ---------------------------------------------------------------- step end
    def _end(self):
        c, cfg = self.ctx, self.cfg
        r = self.cur
        if not c["tets"]:                               # live-viewer path: cache the topology on the first step
            post_init(c)
        driven = bool(c.get("balloon") and c["balloon"]["drive"] and r["phase"] == "B" and c["balloon"]["W_cur"] is not None)
        if driven:                                      # re-impose after the solve, so the frame shows the driven shape
            mo = c["nodes"]["vagina"].dofs
            mo.position.value = c["balloon"]["W_cur"].tolist()
            mo.velocity.value = np.zeros_like(c["balloon"]["W_cur"]).tolist()
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
                if b == "vagina" and driven:            # a kinematically collapsed wall is not a mechanical state
                    vr = np.ones_like(vr)
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
            Fo = ovoid_origin(r, a_path)
            # R_rows (ROWS = applicator x, y, z), NOT Rdev.  wall_metrics places device vertices as
            # `Pw = org + Vp @ R`, the same row convention as animate_hybrid.device_world and run_hybrid.write_frame,
            # but it was called with Rdev = R_rows.T.  MEASURED on Z7S: the two differ by 17.1 deg and put shaft
            # vertices 4.52 mm (mean) to 8.94 mm (max) from their true positions, so every device_gap_mm,
            # ovoid_gap_mm (and its in_contact flag) and containment.r_dev_mm logged before this fix was taken
            # against a mis-rotated device.  Ground truth for the convention: device_final.json's recorded
            # ovoid_centres_mm reproduce exactly with R_rows and are 2.897 mm out with R_rows.T.
            # The SOFA posing was never affected: rigid_pose wants COLUMNS = applicator axes, which Rdev correctly is.
            off, rad, st, tt, gap, cont = wall_metrics(c, Xv, r["F"], a_path, Fo, np.asarray(r["R_rows"], float),
                                                       ovoid_R_rows(r))
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
                ov = [v for p, v in gap.items() if p in OVOID_PARTS]        # the caps only, not the rods
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
                   n_canal_ties=int(self.n_canal_active), canal_d_mm=getattr(self, "canal_d", None),
                   apex_lift_mm=(round(float(self.apex_lift), 3) if hasattr(self, "apex_lift") else None),
                   cardinal_max_stretch_mm=round(self.lig_ext_max, 4),
                   min_vol_ratio=round(minvol, 4), finite=bool(finite))
        if c.get("balloon") is not None:
            b = c["balloon"]
            row["balloon"] = dict(w=round(float(b["w"]), 4), active=bool(b["active"]), released=bool(b["released"]),
                                  mode=cfg.get("balloon_mode", "release"))
            if r["phase"] == "B" or (b["released"] and self.k < len(self.sched) and self.sched[self.k].get("bal") is None
                                     and self.k > 0 and self.sched[self.k - 1]["phase"] == "B"):
                row["balloon"]["groups"] = dict(wall_outer=b.get("wall_group_seen"), balloon=b.get("balloon_group_seen"))
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
