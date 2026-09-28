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
      cfg corpus_model "fem" (S9): /corpus is a deformable body as above, FIRST in the graph, without collision models;
        /iface is a BarycentricMapping of its tets (same path); RestShapeSprings "pose" hold corpus_pose_set to the
        kinematic target (controller-written /targets/corpus_pose_tgt); "canal_tie" drives its canal nodes onto the
        tube; /corpus_follow (solver-less) is the surface the OARs collide with (corpus_oar_contact "follow")
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
                                    # "tandem_first" (G32 fix plan S5 / TF0): pose.json insertion_path_tandem_first,
                                    #   regenerated by geom.tandem_first_path (build_schedule_tandem_first): phases
                                    #   V (tip up the vaginal slit of the physician's path to the os O_true, corpus at
                                    #   rest), C (tip through the labelled lower canal while the corpus ROTATES about
                                    #   O_true by w_r), L (tandem + corpus lifted together onto device_final and the
                                    #   pose-rule corpus target).  The record's rows were checked on the host by
                                    #   hybrid/tf_gate.py (containment in the wall's lumen, canal in the tube, per-step
                                    #   limits); their counts replace n_approach / n_insert / n_seat.  Needs
                                    #   ovoid_mode "none" (TF0 is the tandem body alone).
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
    E_kPa=dict(cervix=30.0, vagina=15.0, bladder=8.0, rectum=8.0, sigmoid=8.0,
               corpus=40.0),        # corpus: read only when corpus_model="fem" (see there for the value's source)
    nu=dict(cervix=0.45, vagina=0.45, bladder=0.49, rectum=0.45, sigmoid=0.45, corpus=0.45),
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
                                    # "none"  : NO ovoid body at all (not built, not posed, not logged): the tandem
                                    #           body alone, e.g. applicator_v4 --tandem-only (TF0), whose directory
                                    #           holds no ovoid / rod / packing OBJ.  Required by a tandem-only
                                    #           applicator (applicator.json params.tandem_only) and by
                                    #           insertion_path "tandem_first"; device_rods / device_packing must be off.
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
                                    # "bilateral" (corpus_model "fem" only): a root-level BilateralInteractionConstraint
                                    #   between the corpus-mapped copies and the cervix nodes -- TWO-WAY: the corpus
                                    #   then also carries the cervix's load (cardinal cables, canal ties, contacts).
                                    #   "attach" / "springs" stay one-way with an elastic corpus as with a rigid one.
    attach_target="mapped",         # what "attach" / "springs" hold the cervix interface nodes to:
                                    # "mapped" (DEFAULT, every run so far): the scene's mapped copy /corpus/iface/mo
                                    #   (RigidMapping of the kinematic corpus; BarycentricMapping of an elastic one).
                                    #   MEASURED (2026-09-27, frames with exact nodal u; scratchpad eu_implement/
                                    #   lagcheck*.py): the constraint reads that copy TWO STEPS STALE -- in TF0c and
                                    #   EUISO_R0 the cervix interface sits on the corpus pose of row k-2 to 0.000 mm
                                    #   through L and 1.0-1.5 mm off the current pose (EUISO_B, elastic: the same, the
                                    #   residual vs the k-2 state 0.00-0.05 mm).  The settle removes it (the target stops),
                                    #   so final states are unaffected; during C / L the corpus-cervix junction is torn by
                                    #   ~2 rows of corpus motion.
                                    # "predicted": a solver-less /targets/corpus_iface_tgt written at the start of each
                                    #   step with where the interface will be at its END: rigid = X0 @ T_corpus(row), exact;
                                    #   elastic = the corpus nodes now + this step's change of the pose target, mapped by
                                    #   the same barycentric weights SOFA's mapper uses (bary_weights).  Not with
                                    #   attach_impl "bilateral" (a constraint on the corpus itself, no copy involved).
                                    # MEASURED in the isolated scene (EUISO_*, TF0c's rows, cervix + corpus alone; residual
                                    #   = cervix interface vs the corpus points it belongs to, median over L / cervix min
                                    #   volume ratio in C): rigid attach mapped ~0.98 mm (2 steps) / 0.398, predicted 0.49
                                    #   (the AttachConstraint still applies the target ONE step late; writing free_position
                                    #   as well changed nothing, bit for bit) / 0.398; attach_impl "springs" mapped ~0.31 /
                                    #   0.561, springs predicted 0.30 / 0.527 (what is left is the springs' compliance under
                                    #   load).  Elastic: attach mapped 0.90 / 0.548, predicted 0.43 / 0.486, springs mapped
                                    #   0.57 / 0.597, springs predicted 0.22 / 0.565, "bilateral" 0.006 / 0.620 (3.6x the
                                    #   step cost, and two-way).  Defaults unchanged ("attach" / "mapped"); TF1v and TF1n
                                    #   pin "springs" + "predicted", the combination canal_tie_follow requires.
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
                                    #   is that fact as a boundary condition.                 # cervix.canal nodes dilated onto the tube
                                    #   (the correction is lateral, but the spring is not: see canal_tie_follow)
    k_canal_mN_per_mm=200.0,
    canal_slack_mm=0.0,             # tie radius = r_tandem + slack
    canal_max_offset_mm=0.5,        # the tie target is at most this far from the node's CURRENT position, so the tie
                                    # force is bounded by k_canal * canal_max_offset (the dilation is still reached,
                                    # over several steps).  Without the bound the target ratchets out to the full
                                    # tube radius and the force reaches k * 2.18 mm, which collapsed the cervix
                                    # elements around the canal (MEASURED, run H1: min volume ratio 0.92 -> 0.06).
    canal_tie_follow=False,         # eu2 (review HIGH-2).  False (DEFAULT, every run up to TF1u): the target is the node's
                                    #   START-OF-STEP position + the capped lateral correction.  The RestShapeSpring is
                                    #   isotropic, so that target also holds the node against its own carried motion in
                                    #   the step, axial included: the tie is "lateral, sliding freely along the tube" only
                                    #   while the tissue is at rest.  "corpus": the node is first carried by this step's
                                    #   rigid corpus increment (T_corpus row k-1 -> k at the node, rigid_step_increment;
                                    #   exactly 0 while the corpus is at rest, so B/P/V and the settle rows, which repeat
                                    #   the last row, are unchanged), then the capped lateral correction is computed there
                                    #   (canal_tie_step carry=): the cap bounds the misfit only, and a canal carried with
                                    #   the uterus feels no force.  "tube": the tandem's rigid motion row k-1 -> k with its
                                    #   slide along the ROW's tube axis removed (tube_lateral_increment; also under
                                    #   canal_tie_axis "final") + the corpus increment's part along that axis: the tube's
                                    #   sideways motion is imposed uncapped (force per node up to k (|carry| + cap)).
                                    #   REQUIRES (a build error otherwise, eu3 / review HIGH-1): canal_tie true, and
                                    #   attach_impl "springs" with attach_target "predicted".  The carry puts the tie nodes
                                    #   on row k's corpus pose, while a lagged attach holds the cervix interface on row k-2
                                    #   ("mapped", the default: read two steps stale) or k-1 ("attach" + "predicted": the
                                    #   AttachConstraint applies its target one step late); on TF0c's rows that offset
                                    #   reaches 1.47 mm in C and 0.98 mm through L, and the carry differs from the lagged
                                    #   tissue motion by > 0.3 mm in 12 of the 39 moving C rows (0.85 mm at the C -> L
                                    #   boundary): the corpus-cervix junction is sheared.  "tube" is also refused with
                                    #   below-flange ties (canal_tie_mode "centre" + canal_tie_below_mm > 0): those nodes
                                    #   sit on the rod line, which the tube's lateral increment does not describe.
                                    #   MEASURED on TF0c's exact frames (scratch eu2_fix/m1_predict.py; 3-step motion of the
                                    #   engaged canal_path tie nodes, |actual - predicted| median mm, off / "corpus" /
                                    #   "tube"): V 0.33 / 0.33 (identical: T_corpus is bit-constant through B, P, V) / 0.57;
                                    #   C 0.236 / 0.232 / 0.314 (canal_s 0-12, near the internal os: 0.250 / 0.178 / 0.250);
                                    #   L 1.09 / 0.34 / 0.34.  The 149 untied neighbours 1.5-5 mm away agree.  So "corpus"
                                    #   matches the tissue best (L: the old target holds back the 0.43 mm/step axial lift),
                                    #   and TF0c's tissue did NOT follow the tube's lateral motion in V / C (partly the old
                                    #   cap itself).  But the C-ramp cap binding is RELATIVE tube-vs-uterus motion: largest
                                    #   one-step change of a tie node's lateral offset to the tube line in C 0.75 mm (off,
                                    #   steps 236-237) / 0.69 (corpus, 228-237) / 0.08 (tube); only "tube" removes it.
                                    #   MEASURED in SOFA, isolated scene (cervix + corpus, TF0c's rows, on the mapped attach:
                                    #   cfgs now refused), vs its control: rigid corpus, "corpus" (EU2ISO_R0F vs EUISO_R0):
                                    #   bit-identical frames through step 120 (C, until T_corpus first moves); C
                                    #   lower_canal_in_tube_frac min 0.952 (0.810), no C frame < 0.9 (2); cervix min volume
                                    #   C 0.392 (0.398), L 0.425 (0.433), H 0.660 (0.544); tie force C net <= 2.34 N, <= 0.39
                                    #   N per node.  "tube" (EU2ISO_FT vs EU2ISO_F, elastic corpus): V cervix min volume
                                    #   0.671 (0.770), V net tie force 0.62 N (0.32).
                                    #   MEASURED in the full scene (tf_metrics; "lower canal" = the C-ramp frames failing
                                    #   lower_canal_in_tube, of 30):
                                    #     TF1v and TF1n (= TF1v with corpus tie stiffness 0: a rigid-equivalent corpus),
                                    #     "corpus" + springs / predicted: 0/30, worst 2.25 / 2.67 mm (TF0c and TF1u, follow
                                    #     off: 5/30, worst 3.86 / 3.92 mm); cervix min volume in C 0.519 / 0.524 (TF0c 0.400).
                                    #     TF1n_noB (springs / predicted, follow off): 3/30, worst 3.61 mm -- the follow, not
                                    #     the attach, cures the lower-canal slip.
                                    #     TF1n_noC ("corpus" on the default mapped attach): ABORTED at step 241 (L,
                                    #     abort_inverted_tets: junction tets 1437 -> 0.30, 2101 -> 0.14; TF1n 0.57-0.62
                                    #     there); cervix tie net force 2.79 / 2.91 N at the C rate jumps (steps 218 / 225;
                                    #     TF1n 2.03 / 2.13), attach residual 1.25-1.48 mm (0.13-0.15).
                                    #     TF1n_attP ("corpus" + attach_impl "attach" / "predicted", one step late): 0/30
                                    #     (worst 2.93 mm), but TF0c's C squeeze is back (tet 9449 0.397) and the settle
                                    #     chatters in part.
                                    #   OPEN with the allowed combination (TF1v, TF1n, TF1p): a period-2 settle chatter of
                                    #   the LEFT portio (the 40 largest-flip cervix nodes sit 12-17 mm to the patient's
                                    #   left, ~0.65 mm swing) against the vaginal fornix (cervix 1.3 mm/step in the settle;
                                    #   TF0c / TF1u 0.25-0.28), not the carry (exactly 0 there).  It seeds, but does not
                                    #   cause, the separate late-settle instability of every run (G32 / TF0c / TF1u
                                    #   included): a period-2 flip of the lower-mid vaginal wall growing 1.09-1.22x per
                                    #   step, because apex_lift_fix holds the mid-wall at stretch 1.36-1.43 and the
                                    #   corotational step has no geometric stiffness for that tension (host eigenvalue
                                    #   -1.20 to -1.28; threshold ~1.22).  Proposed, NOT implemented: a settle-phase
                                    #   Rayleigh stiffness on the wall.  Until then score final organ numbers with the
                                    #   README's M121@k* convention (a 1-2-1 average of frames k-6, k-3, k at k*, the last
                                    #   frame before OAR vertices get behind the flipping wall), not the final frame.
    log_canal_tie_force=False,      # eu2: log row["canal_tie_force"] (net / max-node / summed spring force of the cervix
                                    #   ties at the END of the step, N, and the largest carry) also with canal_tie_follow
                                    #   off, so the old and new targets can be compared; always logged when follow is on.
    tie_ramp_steps=3,
    # --- S1 of the G32 fix plan (logs/audit_G32/fix_plan_G32_audit.md): which line the tie aims at, which nodes it
    #     drives and when they engage.  All three defaults are the G16-G32 behaviour, bit for bit (hybrid/test_ties.py
    #     checks the tie block against the pre-S1 code), so every earlier run reproduces.
    canal_tie_axis="final",         # S1a.  "final" (DEFAULT, the pre-S1 code): the tie and its canal_d_mm diagnostic
                                    #   are measured about the FINAL tube axis (tgt tube_axis) through the CURRENT
                                    #   flange, and below the flange about tgt axis (the path axis).  That is a BUG
                                    #   whenever the row's axis turns (insertion_path "canal", tandem_rotation "canal",
                                    #   tandem-first): the tie drives the canal onto a line the tube is not on.
                                    #   MEASURED on G32's own schedule (host replay, scratchpad impl_s12_scene/
                                    #   replay_g32.py): the row axis is 28.9 deg off the final one through B, P, A and
                                    #   the start of T (steps 0-96), i.e. the tie line is 24.2 / 29.0 mm off at 50 / 60
                                    #   mm above the flange; G32 got away with it only because radius engagement fired
                                    #   late (step 125, u 0.764), when the angle was down to 10.2 deg (7-9 mm at 40-50
                                    #   mm).  "row": the CURRENT row's tube axis (and the row's rod axis below the
                                    #   flange, see canal_rod_axis); canal_d_mm is then the distance to the line the
                                    #   tube is actually on.  Identical to "final" for any schedule whose axis does not
                                    #   turn (rule mode with tandem_rotation "off").
    canal_rod_axis="shaft",         # S1a, "row" only: the line the below-flange ties (canal_tie_below_mm) aim at.
                                    #   "shaft" = the tandem's own vaginal rod in the row's frame, -landmarks.
                                    #   shaft_end_dir @ R_rows (applicator_v3: the straight rod; at G32's final row it
                                    #   is 0.0016 deg from tgt axis); "tube" = the tube axis continued below the flange
                                    #   (applicator v4, whose tube runs straight through the ring bore to z_app ~ -22).
    canal_tie_set="canal",          # S1b.  The cervix node set the tie drives.  "canal" (DEFAULT): the 38 nodes along
                                    #   canal.npz, 23 of them on its segment extrapolated along a0 below L_end, 12 of
                                    #   those outside the labelled tract (fix plan R4).  "canal_path": tandem_path.py's
                                    #   set, the cervix nodes within 3 mm of the physician's tandem path from s_F up to
                                    #   the internal os + 3 mm, with their path arclengths in "canal_path_s" (S3).
    canal_engage="radius",          # S1b, "centre" only.  "radius" (DEFAULT): a node engages once it is within tube
                                    #   radius + canal_engage_mm of the tie line.  "depth": node i engages at the first
                                    #   step whose schedule row carries tip_s >= canal_s[i] (tip_s = the tip's arclength
                                    #   along the tandem path, same origin as canal_s: O_true; negative = still in the
                                    #   vagina), so nodes with canal_s < 0 engage during late V.  Needs a set with
                                    #   "<set>_s" and rows carrying tip_s (the tandem-first schedule, S5); rows of phase
                                    #   B/P may omit it (they engage nothing), any other row without it is a build
                                    #   error.  MEASURED why (G16): radius engagement grabs a tube that is still BESIDE
                                    #   the canal and inverted the rectum; depth engagement ties a node only once the
                                    #   tip has actually passed it.  Do not use it with a schedule whose tip is not on
                                    #   the path (fix plan S1, "Do not").
    apex_attach="follow",          # "recentre": as "canal", plus the targets shift by w(s) * e, where e is the
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
    stop_after_phase=None,          # end the run (status "stopped_after_<phase>") once the LAST row of this phase is solved:
                                    #   "B" for the balloon-only contact probes (S2 P0-P4), "T" for P5 (through the end of
                                    #   insertion).  None (DEFAULT) = the whole schedule, then the settle.
    log_min_vol_where=False,        # S6 diagnostics: log, per body, WHERE its worst element is (disp[body].min_vol_where:
                                    #   tet index, its 4 nodes, centroid, counts below 0.6 / 0.8).  MEASURED why (TF0): the
                                    #   cervix minimum volume ratio fell to 0.267 for single steps while the uterus rotated
                                    #   in phase C, and the log gave only the value.  Logging only; mechanics untouched.
    frame_u_bodies=[],              # S6 diagnostics, read by run_hybrid.write_frame: with frames on, also save
                                    #   frames/step_<k>_<body>_u.npy (float32 displacement of EVERY tet node) for these
                                    #   bodies, so the canal -- interior nodes the tie drives -- can be carried exactly
                                    #   per frame.  MEASURED why (TF0 final frame): the harmonic extension of the surface
                                    #   that the frames allow misplaces the tie nodes by 0.86 / 1.24 mm (median / max)
                                    #   against final/cervix_u.npy, as large as the 1 mm canal-in-tube tolerance.
                                    #   [] (DEFAULT) = surfaces only, as before.
    # --- S2 of the G32 fix plan: contact-integrity probes.  Every default leaves the scene exactly as G32 built it.
    #     MEASURED before designing them (G32 provenance: this file's hash is the one G32 ran): every organ surface, the
    #     balloon and both wall sheets ALREADY carry Triangle + Line + Point collision models (add_deformable,
    #     _add_balloon), so the plan's P2 ("add Point and Line models") and P5c ("Point/Line models on both wall
    #     sheets") describe the G32 scene itself.  What the probes vary instead is what decides whether a kinematic sheet
    #     keeps its contacts: LocalMinDistance runs on the positions AFTER the controller has moved the sheet, and a
    #     TriangleCollisionModel is one-sided (bothSide 0, SOFA v22.12 container probe), so a vertex that the sheet
    #     overtakes by more than its standing gap (contact_mm 0.5) in one step is behind the triangle when detection
    #     runs and gets no contact at all.  G32's balloon moves its outer nodes up to 25.7 mm over 60 smoothstep steps:
    #     MEASURED (scratchpad impl_s12_scene/check_probe_geom.py) up to 0.643 mm/step, above 0.5 on steps 16-43 (76
    #     nodes at the peak), 0.322 at n_balloon 120; the oar_sheet monitor below reads 15 / 15 bladder / rectum
    #     vertices behind the sheet on G32's frame 30, 52 / 96 on frame 60 and 59 / 89 on frame 180 (the audit's vtk
    #     counts: 50 / 96-102 and 59 / 89).
    collision_proximity_mm={},      # P2 / P5c: per-model `proximity` (mm) keyed "balloon", "vagina_inner", "vagina_outer" or
                                    #   an organ name.  LocalMinDistance adds it to alarm AND contact distance for that
                                    #   model's pairs only, so e.g. {"balloon": 1.0} holds the organs 1.5 mm off the sheet
                                    #   (more than one step of travel) without raising alarm_mm globally (G27).
    balloon_subdivide=0,            # P3: 1-to-4 midpoint subdivisions of the balloon's triangles (0 = the wall's outer
                                    #   triangles as they are).  The subdivided vertices are written each step as the
                                    #   means of their parent nodes, so the sheet's SHAPE is unchanged; only the density
                                    #   of collision primitives rises (4487 -> 17948 triangles at 1).
    oar_sheet_monitor=False,        # log, every step, the signed distance of the OAR surface nodes (oar_sheet_bodies) to
                                    #   the sheet the OARs are loaded by (the balloon; the wall's outer sheet once a
                                    #   "release" balloon is off): count below 0 and below -0.5 mm, and the minimum.  Nearest
                                    #   sheet node + its outward (area-weighted) normal; nodes beyond the sheet's two end
                                    #   rings are skipped.  A cheap in-run monitor only; the S2 gate is the host signed
                                    #   distance on the saved frames (vtkImplicitPolyDataDistance, S0).
    oar_sheet_bodies=["bladder", "rectum", "sigmoid"],
    oar_sheet_penalty_mN_per_mm=0.0,    # P4: > 0 turns on a python penalty on the same nodes (implies the monitor): a node
                                    #   closer than oar_sheet_margin_mm to the sheet, or behind it, gets a spring to the
                                    #   point oar_sheet_margin_mm in front of the sheet along the sheet normal, at most
                                    #   oar_sheet_max_offset_mm from where it is (bounded force, as the canal tie), so a
                                    #   vertex the contact has lost is pulled back out.  0 = off (DEFAULT).
    oar_sheet_margin_mm=0.5,
    oar_sheet_max_offset_mm=1.0,
    cervix_collision_refine=None,   # P5b: dict(levels=1, radius_mm=8.0): the cervix's collision models sit on a REFINED
                                    #   copy of its surface instead of the tet boundary: the surface triangles within
                                    #   radius_mm of the rest vaginal wall are split 1-to-4 `levels` times and the copy is
                                    #   carried by a BarycentricMapping of the cervix tets.  No re-meshing: the FEM mesh,
                                    #   its 792 surface nodes and every node set are unchanged.  None = off (DEFAULT).
    # --- S9 of the G32 fix plan (2026-09-27): an ELASTIC corpus.  The user: "the uterus is not a rigid body; it stretches
    #     a bit as the tandem is inserted".  Every default below keeps the corpus RIGID and kinematic, exactly as G32 and
    #     TF0c ran (host replay of both scenes against b3745b1: identical graph, schedule, summary and controller Data).
    #     MEASURED (scratchpad eu_evidence / eu_design and the two eu reviews, patient-derived, local): the BT uterus is
    #     explained by rigid motion to Dice 0.883 (TF0c pose; best rigid 0.892); what is left is organ-scale and small --
    #     mean surface residual ~2 mm.  Along the tandem BT stretches only 1.9-2.7 % by the volume-consistent label affine
    #     (sub-voxel; the design's "~5 %" was an in-sample Dice fit that does not survive in the device frame, see
    #     corpus_pose_stretch_lam).  Across it the BT uterus is REALLY narrower left-right, by 3.2-5.9 mm (8-15 %) 40-55 mm
    #     above the flange, with ~9 % front-back thickening: no tie mechanism reproduces that (models 0.1-1.1 mm), and
    #     neither does organ contact (TF1c, corpus_oar_contact "two_way": 0.076 N settled, all sigmoid at the fundus, no
    #     load at 40-55 mm; no labelled organ lies within 2 mm of the BT uterus there).  The BT uterus does NOT bend where
    #     the labelled canal's top would have to go (0.2-0.4 mm at h 51-57 mm), and forcing the whole canal onto the tube
    #     in a solid mesh inverts tets and drops Dice to ~0.81.  So the gross motion stays the pose rule's (springs to the
    #     kinematic target), and the elastic part is what the tandem, the cervix and the OARs add on top of it.
    #     OUTCOME (full scene, TF1v against TF1n = the same graph with corpus tie stiffness 0, i.e. a rigid-equivalent
    #     corpus; noise floor TF1p = stiffness 4): the elastic corpus pulls the labelled canal inside it 2-3.6 mm closer
    #     to the tandem (upper residual 17.3 -> 14.5 mm) but does not bring the uterus SHAPE closer to BT (Dice 0.883
    #     either way); scored with M121@k*, the only organ changes above the floor (<= 0.0003 Dice) are small and away from
    #     BT (sigmoid -0.0012, cervix -0.0006).  The BT scan also shows the uterine cavity (T2-bright slit) still reaching
    #     10-15 mm to the patient's left of the real tandem 28-40 mm up, i.e. the canal label's left-curving top is cavity
    #     the tandem does not fill; pulling it onto the tube (ties above s ~25) is therefore not supported by BT
    #     (corpus_canal_s_max).
    corpus_model="rigid",           # "rigid" (DEFAULT): the kinematic Rigid3d corpus, placed each step at X0 @ T_corpus(row)
                                    #   (every run up to TF0c).
                                    # "fem": a TetrahedralCorotational FEM body on meshes/corpus/tets.vtk (1779 nodes,
                                    #   9001 tets, true min dihedral 6.3 deg), solved BEFORE the cervix; its gross pose is
                                    #   held by soft springs from corpus_pose_set to the SAME kinematic target (target
                                    #   positions rewritten each step), /corpus/iface/mo becomes a BarycentricMapping of
                                    #   the corpus tets (same path, so attach_impl "attach" / "springs" work unchanged; 64
                                    #   of the 127 cervix interface nodes lie up to 2.1 mm outside the corpus tets and are
                                    #   extrapolated), corpus canal ties (below) drive the corpus's own canal nodes onto
                                    #   the tube, and the OARs meet it through corpus_oar_contact.  E / nu: E_kPa.corpus,
                                    #   nu.corpus (material / material_by_body as for the other bodies).
                                    #   E_kPa.corpus = 40: in vivo (MRE, Jiang 2014) the corpus is ~1.3x the cervix (|G*|
                                    #   2.58 vs 2.00 kPa), and the model's cervix is 30 kPa; the literature range of E is
                                    #   ~6-40 kPa (MRE, transabdominal / transvaginal SWE) -- S9's ">= 50" is above it.
                                    #   E is NOT shape-neutral, and 40 is chosen for stability, not from tissue data.
                                    #   MEASURED (probe P7, isolated scene, pose k 20, corpus ties without
                                    #   corpus_tie_follow): upper canal residual 12.65 / 13.28 / 13.75 / 14.39 / 15.13 mm at
                                    #   E 20 / 25 / 30 / 40 / 60; E 15 aborts in L (the 36 corpus tets whose four nodes are
                                    #   all tie nodes collapse), E 20 reaches a min tet ratio of 0.27, and at E <= 30 more
                                    #   than 1 % of the tets end above 30 % strain (T4).  With corpus_tie_follow E 25 still
                                    #   fails T4 (EU3_E25: p95 24.1 %, 3.0 % of tets above 30 %).  E 40 is the softest value
                                    #   that passes.
    corpus_pose_set="serosa",       # fem: the nodes held to the pose target.  "serosa" = surface_nodes minus
                                    #   interface_cervix (463); "surface" = all 592 surface nodes; "all" = 1779 (a stiff
                                    #   control: the corpus ~rigid); "shell" = the serosa at k_fixed_mN_per_mm (a rigid
                                    #   shell around a deformable core, by springs rather than FixedConstraint).
    k_corpus_pose_mN_per_mm=5.0,    # fem: pose spring per node (463 x 5 = 2.3 N/mm on the serosa; the net tie force of
                                    #   2.5-3.5 N then drifts the corpus ~1.1-1.9 mm rigidly, host estimate).  This default
                                    #   FAILS the S9 uterus Dice gate (>= 0.87): isolated EUISO_B 0.859, probe P7 k 5 at E
                                    #   15-60: 0.859-0.862 (pose drift 2 mm mean).  k 20 passes (0.878-0.882; rigid 0.883)
                                    #   and TF1u pins 20 in its cfg; the default stays 5 because the EU* cfgs rely on it.
    k_corpus_bottom_mN_per_mm=None, # fem: extra springs on the 129 interface_cervix nodes to the same target (None = none):
                                    #   the anchor any stretch along the tandem needs (probe 5 / 50).
    corpus_pose_exclude_canal_mm=0.0,   # fem: drop pose-set nodes within this distance of the REST labelled canal
                                    #   (inputs/tandem_path.npz, s >= 0) so the pose springs do not fight the canal ties
                                    #   locally (serosal nodes within 8 / 10 / 12 mm: 33 / 57 / 107).  0 = keep all.
    corpus_pose_stretch_lam=1.0,    # fem, CALIBRATED / SCENARIO: the pose target becomes a volume-preserving stretch
                                    #   along the row's tube axis applied after T_row, anchored at the corpus bottom (h0 =
                                    #   0.5 percentile of the posed nodes' h about their centroid): h' = h0 + lam_row (h -
                                    #   h0), lateral / sqrt(lam_row), lam_row = 1 + (lam - 1) w (w = the row's w_r in
                                    #   tandem-first, else its s).  1.0 = the rigid target, bit for bit.  MEASURED in-sample
                                    #   optimum of the PELVIS-frame Dice 1.05 (0.883 -> 0.893 on the TF0c pose; 1.10 back
                                    #   to 0.883, 1.15 0.863): fitted to THIS patient's BT label, so tagged CALIBRATED in the
                                    #   summary.  It is NOT evidence of stretch: aligned on the BT tandem (device frame)
                                    #   the 1.05 run scores LOWER than no stretch (EU3_D3 0.882 vs EU2ISO_F 0.887); the
                                    #   pelvis-frame gain is within the device-placement error along the tandem (~1.1 mm),
                                    #   and the label affine gives BT only 1.9-2.7 % axial stretch.  A scenario knob only.
    corpus_oar_contact="follow",    # fem: how the OARs meet the corpus.  "follow" (DEFAULT) = a solver-less copy of the
                                    #   corpus surface (surface.obj faces, identical to the rigid collision surface at rest,
                                    #   same groups) rewritten each step from the corpus nodes, predicted forward by this
                                    #   step's change of the pose target: exactly the rigid corpus's contact when the
                                    #   corpus follows its target, one-way (the OARs feel the corpus, not vice versa).
                                    #   "two_way" = collision models on the FEM corpus itself; "off" = none.  MEASURED in
                                    #   the full scene (TF1c = TF1v + "two_way", floors TF1cp / TF1cn): settled contact on
                                    #   the corpus 0.076 N (sigmoid at the fundus), no squeeze at 40-55 mm, uterus Dice
                                    #   -0.00002 (floor 0.0001), +8 % runtime; it keeps the bladder out of the corpus
                                    #   through C (one-way: 72-73 bladder vertices up to 8.7 mm inside at step 237) but not
                                    #   through L; flange drift in L 0.04 -> 0.38 mm.  Not adopted as the default.
    corpus_canal_tie=True,          # fem: tie the corpus's own canal nodes onto the tube (as canal_tie_mode "centre" does
                                    #   for the cervix: lateral, bounded per step, depth engagement, the row's tube axis).
                                    #   Needs schedule rows with tip_s (insertion_path "tandem_first").
    corpus_canal_tie_set="canal_path",  # fem: the corpus node set (meshes/corpus/meta.json, written by
                                    #   `python -P hybrid/tandem_path.py corpus_nodesets`): corpus nodes within 3 mm of the
                                    #   labelled canal (s >= 0), 39 nodes, canal_s 9.6-46.6 in "canal_path_s".
    k_corpus_canal_mN_per_mm=400.0, # fem: tie stiffness (as TF0c's cervix ties).  The force per node is bounded by k x
                                    #   max_offset ONLY with corpus_tie_follow true (see there and below).
    corpus_canal_max_offset_mm=0.5, # fem: per-step bound on the target's offset from the node: with corpus_tie_follow true
                                    #   the spring force per node is about 400 x 0.5 = 0.2 N (TF1v: exactly 0.200 N per node
                                    #   in the settle; the end-of-step force exceeds it while a node lags its carried
                                    #   target, up to 0.2355 N at step 228, 75 of 304 steps), the upper end of what the
                                    #   evidence allows ("cap
                                    #   the tie force at ~0.1-0.2 N per node, or do not tie above s ~20-25 mm"): above that
                                    #   the ties straighten the canal beyond the BT evidence.  With the DEFAULT
                                    #   corpus_tie_follow false the bound does NOT hold: the start-of-step target adds k x
                                    #   the node's own motion in the step (unit test 0.39 N per node; TF1u net 7.35 N in L).
                                    #   The summary's force_bound_per_node_mN is k x max_offset in either case.
    corpus_canal_s_max=None,        # fem: drop tie nodes with canal_s above this (e.g. 25 or 33); None = all 39 (as
                                    #   TF1u / TF1v ran).  The BT scan supports ~25: above it the labelled canal is the
                                    #   left part of the cavity, still beside the real tandem at BT (see the S9 header).
    corpus_canal_engage_clip=True,  # fem: engage at min(canal_s, max schedule tip_s - 0.05).  MEASURED: TF0c's tip_s
                                    #   ends at 44.30 mm, so the 5 nodes with canal_s 44.8-46.6 would never engage.
    corpus_canal_axial="none",      # fem: "none" = no axial DRIVE: the target is the node's position (start of step, or
                                    #   carried: corpus_tie_follow) + the capped LATERAL correction.  It does NOT mean the
                                    #   node slides freely along the tube: the spring is isotropic and pulls toward that
                                    #   target in every direction (review HIGH-2: with corpus_tie_follow false the target
                                    #   sits at the start-of-step node, so the ties hold back the corpus's own motion --
                                    #   TF1u L: 7.35 N net, ~ 39 x 0.4 N/mm x 0.49 mm/step = 7.67 N, a rate artifact).
                                    #   "arclength" = also drive it along the tube to h* = L_iu - (tip_s - canal_s) (the
                                    #   straightened canal keeps its length; bounded by the same max offset per step).
                                    #   Host estimate: this stretches the corpus little (fundus +0.6 mm with the bottom
                                    #   held); corpus_pose_stretch_lam is the calibrated route to the BT stretch.
    corpus_tie_follow=False,        # eu2 (review HIGH-2), fem ONLY (a bool; refused with corpus_model "rigid", where there
                                    #   are no corpus ties).  False (DEFAULT, TF1u and the EUISO / EUP7 runs): the tie target
                                    #   is the START-OF-STEP node + the capped lateral correction (canal_tie_step as the
                                    #   cervix).  True: the node is first carried by this step's change of its pose target
                                    #   (corpus_pose_targets row k minus row k-1 at the node, as follow_surface), and the
                                    #   capped lateral correction is computed from that carried position: a canal riding
                                    #   with the uterus gets zero tie force, the cap acts only on the misfit.  The AXIAL
                                    #   part of the carry is kept (not projected out): the spring is isotropic, so dropping
                                    #   it IS the defect (0.433 of TF1u's 0.492 mm/step L lift is along the tube); with it
                                    #   the tie resists only the node's deviation from the carried motion within one step,
                                    #   re-anchored every step -- an elongation of the corpus along the tube of 1-2.5 mm
                                    #   over the ~135 C + L steps (the BT evidence) costs < 0.02 mm/step, < 10 mN per node.
                                    #   MEASURED in SOFA, isolated scene (EU2ISO_F = TF1u's corpus + this + canal_tie_follow
                                    #   "corpus" on the mapped attach -- a combination canal_tie_follow now refuses; the
                                    #   corpus is one-way coupled, so it is the same with the springs / predicted attach,
                                    #   EU3_C, to 0.0002 mm -- vs EUP7_E40_K20_H25, whose corpus equals TF1u's to 0.0 mm):
                                    #   bit-identical frames through step 120; corpus tie net force C 2.57 N (5.44), L 3.37
                                    #   (7.35), H 3.37
                                    #   (6.69); L pose max 1.57 mm (2.78), non-rigid serosa max 1.24 (2.06); L one body: tip /
                                    #   flange-in-corpus drift 0.06 / 0.04 mm (0.48 / 1.04), canal spread 0.17 (2.34);
                                    #   worst-frame strain p95 16.7 % (27.5), tets > 30 % 0.44 % (3.8); corpus min tet 0.783
                                    #   (0.735); C lower_canal_in_tube_frac min 1.0 (0.476).  Unchanged: uterus Dice 0.883
                                    #   (0.882), canal bands, upper residual 14.52 (14.46), best line 6.25 (6.27) -- the fix
                                    #   removes the artifact; it does not make the corpus stretch.  Full scene (TF1v, with
                                    #   canal_tie_follow "corpus" + springs / predicted attach, vs TF1u): the corpus equals
                                    #   the isolated EU3_C to 3e-9 mm; L tie force 7.35 -> 3.37 N net; L one body spread 2.34
                                    #   -> 0.17 mm, flange drift 1.04 -> 0.04 mm; strain p95 27.5 -> 16.7 %.
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
                canal=pose.get("insertion_path_canal"), tandem_first=pose.get("insertion_path_tandem_first"))


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


TF_ROW_TOL_MM = 1e-6        # regenerated vs recorded tandem-first rows (flange mm; axis rad)


def build_schedule_tandem_first(cfg, tgt):
    """G32 fix plan S5 / TF0: the tandem goes in FIRST and its tip follows the physician's tandem path.

    B (balloon) and P rows hold the tandem at the start of V (tip v_below_mm under the path's introitus end, the device
    outside the body; corpus at rest).  Then the rows of pose.json insertion_path_tandem_first, regenerated here by
    geom.tandem_first_path from the record's params (host and container run the same pure-numpy code):
      V  tip up the vaginal part of the path to O_true (<= 1 mm / step), tube axis = the chord of the path over the
         tube length behind the tip, corpus at rest;
      C  tip = T_C(w_r) tau(d) + w_r delta through the labelled lower canal (d 0 -> L_iu - d_F), T_C(w) = Trans(w L_C) o
         Rot(w theta about O_true): the corpus ROTATES about the os (L_C = 0 unless a flagged fallback of the gate
         started part of the lift in C); the last C row is the tandem at Trans(L_C - L) device_final;
      L  tandem and corpus translate TOGETHER by w_l (L - L_C) onto device_final and corpus.T_preBT_to_target.
    WHY: the plan's decomposition T_final = Trans(L) o Rot(theta about O_true) keeps the os (and with it the vault on the
    portio) in place while the tip travels, and puts the cranial lift after the tandem is in (R2: G32's screw lifted the
    os 9.7 mm before the tip was halfway).  The rows are the ones hybrid/tf_gate.py checked on the host (containment in
    the rest lumen, lower canal in the tube, per-step limits), so their COUNTS are the record's: one row per step,
    n_approach / n_insert / n_seat are not used.  The recorded rows are compared with the regenerated ones (flange and
    axis within TF_ROW_TOL_MM): a record written by other code than this geom is refused, not silently re-interpreted.
    Rows carry tip_s (the tip's arclength along the planned path from O_true = the depth d in C; S1b depth engagement),
    d_mm, w_r, w_l; s = (w_r + w_l) / 2 (1 at the final pose, as run_hybrid's reached_final_pose expects); ov_lag 0
    (no ovoid body)."""
    rec = tgt.get("tandem_first")
    if not rec:
        raise ValueError("insertion_path='tandem_first' needs pose.json insertion_path_tandem_first in the applicator dir "
                         "(python -P hybrid/tf_gate.py search --write)")
    if cfg.get("ovoid_mode") != "none":
        raise ValueError("insertion_path='tandem_first' moves the tandem body alone: set ovoid_mode 'none' (got %r); "
                         "a parked / seated ovoid body is TF1's (fix plan S7)" % cfg.get("ovoid_mode"))
    prm = rec["params"]
    if abs(float(rec.get("flange_shift_mm", np.nan)) - float(tgt["delta_mm"])) > 1e-9 or \
            float(np.abs(np.asarray(prm["F_fin"], float) - tgt["flange"]).max()) > 1e-3:
        raise ValueError("insertion_path_tandem_first was built for flange_shift_mm %s / flange %s, the run asks for %s / %s"
                         % (rec.get("flange_shift_mm"), prm["F_fin"], tgt["delta_mm"], tgt["flange"].tolist()))
    rows = geom.tandem_first_path(prm)
    ref = rec.get("rows")
    if ref:
        if len(ref) != len(rows):
            raise ValueError("insertion_path_tandem_first: %d recorded rows, %d regenerated" % (len(ref), len(rows)))
        dF = max(float(np.abs(np.asarray(q["F"], float) - r["F"]).max()) for q, r in zip(ref, rows))
        da = max(float(np.linalg.norm(np.asarray(q["tube_axis"], float) - r["tube_axis"])) for q, r in zip(ref, rows))
        if dF > TF_ROW_TOL_MM or da > TF_ROW_TOL_MM:
            raise ValueError("insertion_path_tandem_first rows differ from geom.tandem_first_path by %.3g mm / %.3g "
                             "(stale record: re-run tf_gate.py search --write)" % (dF, da))
    start = geom.tf_row_V(prm, float(prm["V_start_s"]))

    def pose_of(r):
        return dict(F=np.asarray(r["F"], float), tube_axis=np.asarray(r["tube_axis"], float),
                    R_rows=np.asarray(r["R_rows"], float), T_corpus=np.asarray(r["T_corpus"], float),
                    tip_s=float(r["tip_s"]), d_mm=float(r["d_mm"]), w_r=float(r["w_r"]), w_l=float(r["w_l"]))
    sched = []
    nB = int(cfg.get("n_balloon", 0) or 0)
    for k in range(1, nB + 1):
        sched.append(dict(pose_of(start), phase="B", u=0.0, s=0.0, bal=k / float(nB), ov_lag=0.0, stage="B"))
    for k in range(int(cfg["n_presettle"])):
        sched.append(dict(pose_of(start), phase="P", u=0.0, s=0.0, ov_lag=0.0, stage="P"))
    n = len(rows)
    for i, r in enumerate(rows):
        sched.append(dict(pose_of(r), phase=r["phase"], u=(i + 1) / float(n), s=0.5 * (r["w_r"] + r["w_l"]),
                          ov_lag=0.0, stage=r["phase"]))
    return sched


def build_schedule(cfg, tgt):
    """Per-step device / corpus kinematics.  u in [0, 1] is the pose-rule path parameter (pose.json insertion_path):
    the device flange is F(u) = F_final - (1 - u) * travel * axis, and the corpus follows the screw motion
    s(u) = smoothstep((u - u_ios) / (1 - u_ios)).  Phases: P (pre-settle), A (approach), T (insertion),
    D (ovoid seating), H (settle)."""
    if cfg.get("insertion_path", "rule") == "canal":
        return build_schedule_canal(cfg, tgt)
    if cfg.get("insertion_path", "rule") == "tandem_first":
        return build_schedule_tandem_first(cfg, tgt)
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
def add_deformable(root, b, inp, cfg, X0, static=False, collide=True):
    """One body.  `static=True` (cfg `static_bodies`) builds it as a NON-DEFORMABLE collision obstacle: the tet
    topology, the full-node `dofs` MechanicalObject and the surface collision models are kept exactly as for a
    deformable body -- so ctrl.X(b), run_hybrid.write_outputs and frame_cache see identical shapes and simply read
    u = 0 -- but there is no ODE solver, no mass, no FEM and (see build_scene) no constraint correction, so nothing
    integrates the body and it stays at its rest pose.  `collide=False` leaves out the surface's collision models
    (the elastic corpus with corpus_oar_contact "follow" / "off": its contact surface is a separate copy, or none)."""
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
            pk = _prox_kw(cfg, "vagina_" + side)        # S2 P5c: {} unless cfg collision_proximity_mm names it
            ch.addObject("TriangleCollisionModel", name="tri", group=gg, **pk)
            ch.addObject("LineCollisionModel", name="lin", group=gg, **pk)
            ch.addObject("PointCollisionModel", name="pnt", group=gg, **pk)
    else:
        g = list(grp[b])
        # a static body is an OBSTACLE: moving=False, simulated=False makes the contact one-sided, so the deformable
        # side alone carries the constraint and no constraint correction is needed on this node
        kw = dict(moving=False, simulated=False) if static else {}
        kw.update(_prox_kw(cfg, b))
        cn = s
        if b == "cervix" and cfg.get("cervix_collision_refine") and not static:
            cn = _add_refined_collision_surface(n, inp, cfg, X0)      # S2 P5b: the models go on the refined copy
        if collide:
            cn.addObject("TriangleCollisionModel", name="tri", group=g, **kw)
            cn.addObject("LineCollisionModel", name="lin", group=g, **kw)
            cn.addObject("PointCollisionModel", name="pnt", group=g, **kw)
    v = s.addChild("vis")
    col = list(inp["bodies"]["bodies"][b]["color"]) + [1.0]
    v.addObject("OglModel", name="ogl", color=col)
    v.addObject("IdentityMapping", name="vm", input="@../sdofs", output="@ogl")
    return n


def _prox_kw(cfg, key):
    """{proximity: mm} for a collision model named in cfg collision_proximity_mm (S2), else {} (the G32 graph)."""
    v = (cfg.get("collision_proximity_mm") or {}).get(key)
    return {} if v is None else dict(proximity=float(v))


def refined_surface(inp, cfg, X0c):
    """S2 P5b: the cervix surface as a triangle list over its OWN surface nodes (surface.obj faces through
    surface_obj_vertex_to_tet_node), with the triangles whose centroid lies within radius_mm of the rest vaginal
    wall (any wall node) split 1-to-4 `levels` times.  Returns (V (m, 3) rest positions, T triangles, info).
    The split triangles' new vertices sit on edges shared with unsplit neighbours (T-junctions): the surface is
    the same piecewise-linear shape, only its collision primitives are denser where the vault meets the portio."""
    P = inp["P"]
    ref = dict(levels=1, radius_mm=8.0)
    ref.update(cfg.get("cervix_collision_refine") or {})
    meta = inp["meta"]["cervix"]
    _, F = geom.read_obj("%s/cervix/surface.obj" % P["meshes"])
    s2n = np.asarray(meta["surface_obj_vertex_to_tet_node"], int)
    T = s2n[np.asarray(F, np.int64)]
    used = np.unique(T)
    rem = -np.ones(len(X0c), np.int64)
    rem[used] = np.arange(len(used))
    Tl, Xs = rem[T], np.asarray(X0c, float)[used]
    Xw = read_vtk_points("%s/vagina/tets.vtk" % P["meshes"])
    from scipy.spatial import cKDTree
    dist = cKDTree(Xw).query(Xs[Tl].mean(1))[0]
    sel = dist <= float(ref["radius_mm"])
    Ts, rounds = subdivide_tris(Tl[sel], int(ref["levels"]), n_vertices=len(Xs))
    V = apply_subdiv(Xs, rounds)
    Tr = np.concatenate([Tl[~sel], Ts])
    info = dict(levels=int(ref["levels"]), radius_mm=float(ref["radius_mm"]), n_tri=int(len(Tl)),
                n_tri_split=int(sel.sum()), n_tri_out=int(len(Tr)), n_vertices=int(len(V)),
                n_surface_nodes=int(len(used)), mapping="BarycentricMapping (cervix tets)")
    return V, Tr, info


def _add_refined_collision_surface(n, inp, cfg, X0c):
    """S2 P5b: a collision-only child of the cervix node carrying the refined surface (refined_surface), mapped from
    the cervix tets by a BarycentricMapping, so contacts on it act on the FEM nodes through the mapping exactly as the
    IdentityMapping `surf` does.  The FEM mesh is not re-meshed."""
    V, Tr, info = refined_surface(inp, cfg, X0c)
    rc = n.addChild("surf_fine")
    rc.addObject("MechanicalObject", name="sdofs", template="Vec3d", position=V.tolist())
    rc.addObject("MeshTopology", name="stopo", position=V.tolist(), triangles=Tr.tolist())
    rc.addObject("BarycentricMapping", name="bm", input="@../dofs", output="@sdofs")
    inp.setdefault("probe_info", {})["cervix_collision_refine"] = info
    return rc


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
    check_follow_cfg(cfg)                               # eu3: refuse lagged / silent-no-op follow cfgs before building
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
    corpus_fem = corpus_model(cfg) == "fem"
    if corpus_fem:
        # the ELASTIC corpus (cfg corpus_model "fem"), FIRST in the graph (its free motion is solved before the
        # cervix's).  That does NOT make the mapped attach target current: MEASURED (EUISO_B), the cervix reads
        # /corpus/iface/mo two steps stale, as with the rigid corpus (CFG attach_target; the log's attach_residual_mm /
        # iface_mo_stale_mm measure it); attach_target "predicted" is the fix
        nodes["corpus"] = add_deformable(root, "corpus", inp, cfg, X0["corpus"],
                                         collide=(cfg.get("corpus_oar_contact", "follow") == "two_way"))
    for b in DEFORMABLE:
        nodes[b] = add_deformable(root, b, inp, cfg, X0[b], static=(b in stat))

    iface = np.asarray(inp["meta"]["cervix"]["node_sets"]["interface_corpus"], int)
    if corpus_fem:
        # the cervix interface nodes carried by the corpus TETS (BarycentricMapping; 64 of 127 lie up to 2.1 mm outside
        # the corpus mesh and are extrapolated from the nearest tet -- exact at rest either way).  Same path
        # /corpus/iface/mo as the rigid copy, so _add_couplings "attach" / "springs" are unchanged.
        corp = None
        ifn = nodes["corpus"].addChild("iface")
        ifmo = ifn.addObject("MechanicalObject", name="mo", template="Vec3d", position=X0["cervix"][iface].tolist())
        ifn.addObject("BarycentricMapping", name="bm", input="@../dofs", output="@mo")
        defo = defo + ["corpus"]                        # solved, logged, checked; constraint correction last (below)
        # the same mapping on the host (SOFA's rule): the true attach residual and the "predicted" attach target
        corpus_bary = bary_weights(X0["corpus"], read_vtk_tets("%s/corpus/tets.vtk" % P["meshes"])[1],
                                   X0["cervix"][iface])
    else:
        # ---- rigid corpus (pose rule) + its rigid-mapped copy of the cervix interface nodes
        cdev = dict(corpus=list(inp["bodies"]["bodies"]["corpus"]["color"]) + [1.0])
        corp = add_rigid_parts(root, "corpus", [("surf", "%s/corpus/surface.obj" % P["meshes"])],
                               dict(surf=grp["corpus"]), dict(surf=cdev["corpus"]), rigid_pose(np.eye(3), np.zeros(3)))
        ifn = corp.addChild("iface")                   # local coords == preBT world (the corpus rigid starts at identity)
        ifmo = ifn.addObject("MechanicalObject", name="mo", template="Vec3d", position=X0["cervix"][iface].tolist())
        ifn.addObject("RigidMapping", name="rm", input="@../rig", output="@mo")

    # ---- rigid device: tandem (tube + shaft) and ovoids (seated separately, cfg ovoid_mode)
    dcol = dict(tube=[0.15, 0.15, 0.20, 1.0], shaft=[0.15, 0.15, 0.20, 1.0],
                ovoid_L=[0.35, 0.35, 0.42, 1.0], ovoid_R=[0.35, 0.35, 0.42, 1.0])
    # ovoid_mode "none": no ovoid body at all (a tandem-only applicator has no ovoid / rod / packing OBJ to load; a
    # MeshObjLoader on a missing file is an error at init).  Checked here so a wrong cfg fails at build time.
    no_ov = cfg["ovoid_mode"] == "none"
    tonly = bool(inp["app"].get("params", {}).get("tandem_only", {}).get("value", False))
    if tonly and not no_ov:
        raise ValueError("applicator %r is tandem-only (params.tandem_only): set ovoid_mode 'none' (got %r)"
                         % (cfg.get("applicator_dir"), cfg["ovoid_mode"]))
    if no_ov and (cfg.get("device_rods") or cfg.get("device_packing")):
        raise ValueError("ovoid_mode 'none' has no ovoid body to carry device_rods / device_packing: switch them off")
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
    ovoids = None
    if not no_ov:
        ov_pose = np.asarray(sched[0]["ov_pos"], float) if "ov_pos" in sched[0] else F0 - sched[0]["ov_lag"] * tgt["axis"]
        R_ov0 = ovoid_R_rows(sched[0]).T if "R_rows" in sched[0] else R0      # "rods": the body's own (final) frame
        dcol.update(rod_L=dcol["shaft"], rod_R=dcol["shaft"], packing=[0.9, 0.9, 0.8, 0.3])
        ovoids = add_rigid_parts(root, "ovoids", [(p, _pobj(p)) for p in ov_parts],
                                 dgrp, dcol, rigid_pose(R_ov0, ov_pose))

    # ---- couplings and supports
    ctx = dict(root=root, cfg=cfg, inp=inp, tgt=tgt, sched=sched, X0=X0, nodes=nodes, corpus=corp,
               tandem=tandem, ovoids=ovoids, iface=iface, Rdev=Rdev, targets=tg, supports=sup,
               extra=dict(inp.get("probe_info") or {}), deformable=defo, static=stat,
               corpus_fem=(nodes["corpus"] if corpus_fem else None), corpus_iface_mo=ifmo,
               corpus_iface_bary=(corpus_bary if corpus_fem else None))
    if no_ov:
        ctx["extra"]["ovoid_body"] = False
    if cfg.get("insertion_path", "rule") == "tandem_first":
        rec = tgt["tandem_first"]
        ctx["extra"]["tandem_first"] = dict(
            n_V=sum(1 for r in sched if r["phase"] == "V"), n_C=sum(1 for r in sched if r["phase"] == "C"),
            n_L=sum(1 for r in sched if r["phase"] == "L"), version=rec.get("version"), written=rec.get("written"),
            wall=rec.get("wall"), lift_in_C=rec.get("lift_in_C"), fallback=rec.get("fallback"))
    _add_couplings(ctx)
    _add_supports(ctx)
    if corpus_fem:
        _add_corpus_pose(ctx)                           # the elastic corpus's pose springs to the kinematic target
        _add_corpus_follow(ctx)                         # ... and the surface the OARs meet (corpus_oar_contact)
    _wall_diag_setup(ctx)                               # read-only per-step lumen diagnostics (wall runs only)
    _add_balloon(ctx)                                   # Stage 2b OAR pre-relaxation (cfg n_balloon > 0, wall runs)
    _add_sheet_penalty(ctx)                             # S2: OAR-vs-sheet monitor / P4 penalty (off by default)
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
    #     cfg attach_target "predicted": a controller-written copy instead of the mapped one (see CFG: the mapped copy is
    #     read two steps stale).  "mapped" (default) adds nothing here: the G32 / TF0c graph.
    obj1 = "@/corpus/iface/mo"
    at_mode = cfg.get("attach_target", "mapped")
    if at_mode not in ("mapped", "predicted"):
        raise ValueError("attach_target must be 'mapped' or 'predicted' (got %r)" % at_mode)
    if at_mode == "predicted":
        if cfg["attach_impl"] == "bilateral":
            raise ValueError("attach_target 'predicted' is for attach_impl 'attach' / 'springs' (bilateral constrains "
                             "the corpus itself)")
        ctx["corpus_iface_tgt"] = ctx["targets"].addObject("MechanicalObject", name="corpus_iface_tgt", template="Vec3d",
                                                           position=X0["cervix"][iface].tolist())
        obj1 = "@/targets/corpus_iface_tgt"
        ctx["extra"]["attach_target"] = "predicted (/targets/corpus_iface_tgt, written each step)"
    if cfg["attach_impl"] == "attach":
        cvx.addObject("AttachConstraint", name="attach_corpus", object1=obj1, object2="@/cervix/dofs",
                      indices1=list(range(len(iface))), indices2=[int(i) for i in iface], twoWay=False)
    elif cfg["attach_impl"] == "bilateral":
        # two-way (elastic corpus only): a Lagrangian constraint solved by the GenericConstraintSolver with both bodies'
        # compliances (each has a LinearSolverConstraintCorrection); zero violation at rest
        if ctx.get("corpus_fem") is None:
            raise ValueError("attach_impl 'bilateral' needs corpus_model 'fem' (a kinematic corpus has no compliance)")
        ctx["root"].addObject("BilateralInteractionConstraint", name="attach_corpus", template="Vec3d",
                              object1="@/corpus/iface/mo", object2="@/cervix/dofs",
                              first_point=list(range(len(iface))), second_point=[int(i) for i in iface])
    else:
        cvx.addObject("RestShapeSpringsForceField", name="attach_corpus", points=[int(i) for i in iface],
                      stiffness=[float(cfg["k_attach_mN_per_mm"])] * len(iface),
                      external_rest_shape=obj1, external_points=list(range(len(iface))))
    # (b) cervix.canal: dilation ties onto the tube, written per step.  The CORRECTION is lateral; the spring is
    #     isotropic, so without cfg canal_tie_follow the start-of-step target also holds the node against its own
    #     carried motion (review HIGH-2).  The set is cfg canal_tie_set (S1b); "canal_path" also carries each node's
    #     path arclength canal_s.
    tie_follow = canal_tie_follow_mode(cfg)             # validated at build time (a bad value fails here, not mid-run)
    ns_c = inp["meta"]["cervix"]["node_sets"]
    tie_set = cfg.get("canal_tie_set", "canal")
    if tie_set not in ns_c:
        raise ValueError("canal_tie_set %r is not a cervix node set (have %s); canal_path is written by "
                         "`python -P hybrid/tandem_path.py nodesets`" % (tie_set, sorted(ns_c)))
    canal = np.asarray(ns_c[tie_set], int)
    ctx["canal"] = canal
    ctx["canal_s"] = np.asarray(ns_c[tie_set + "_s"], float) if (tie_set + "_s") in ns_c else None
    if cfg.get("canal_engage", "radius") == "depth":
        # fail at build time, not silently at run time: depth engagement needs a per-node arclength and a schedule
        # that says where the tip is along the same path
        if cfg.get("canal_tie_mode", "dilate") != "centre":
            raise ValueError("canal_engage 'depth' is defined for canal_tie_mode 'centre' only")
        if ctx["canal_s"] is None or len(ctx["canal_s"]) != len(canal):
            raise ValueError("canal_engage 'depth' needs node set %r_s (per-node path arclength) of the same length as "
                             "%r" % (tie_set, tie_set))
        miss = [i for i, r in enumerate(ctx["sched"]) if r["phase"] not in ("B", "P") and r.get("tip_s") is None]
        if miss:
            raise ValueError("canal_engage 'depth' needs schedule rows carrying tip_s (the tip's arclength along the "
                             "tandem path); %d rows lack it, first %d (phase %s).  B/P rows may omit it (never engage)."
                             % (len(miss), miss[0], ctx["sched"][miss[0]]["phase"]))
    if (tie_set, cfg.get("canal_engage", "radius"), cfg.get("canal_tie_axis", "final")) != ("canal", "radius", "final"):
        ctx["extra"]["canal_tie_cfg"] = dict(set=tie_set, n=int(len(canal)), engage=cfg.get("canal_engage", "radius"),
                                             axis=cfg.get("canal_tie_axis", "final"),
                                             rod_axis=cfg.get("canal_rod_axis", "shaft"),
                                             canal_s_range=(None if ctx["canal_s"] is None else
                                                            [round(float(ctx["canal_s"].min()), 3),
                                                             round(float(ctx["canal_s"].max()), 3)]))
    if tie_follow:                                      # eu2: only when set, so the default summary is unchanged
        ctx["extra"].setdefault("canal_tie_cfg", {})["follow"] = tie_follow
    if cfg["canal_tie"] and len(canal):
        ctx["canal_tgt"] = ctx["targets"].addObject("MechanicalObject", name="canal_tgt", template="Vec3d",
                                                    position=X0["cervix"][canal].tolist())
        ctx["canal_ff"] = cvx.addObject("RestShapeSpringsForceField", name="canal_tie",
                                        points=[int(i) for i in canal], stiffness=[0.0] * len(canal),
                                        external_rest_shape="@/targets/canal_tgt",
                                        external_points=list(range(len(canal))))
    # (b2) the elastic corpus's own canal nodes (corpus_model "fem", corpus_canal_tie)
    if ctx.get("corpus_fem") is not None:
        _add_corpus_ties(ctx)
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
            # indices EMPTY through phase B (the driven wall passes through the cervix's portio as it opens and the
            # cervix must be able to push its top ring aside -- G31 first try crushed the cervix, vol ratio 0.09 at
            # step 15); the controller sets them to the apex nodes at the first non-B step.
            ctx["apex_fix"] = ctx["nodes"]["vagina"].addObject("FixedConstraint", name="apex_fix", fixAll=False, indices=[])
            ctx["apex_fix_on"] = False
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


def _add_corpus_ties(ctx):
    """(b2) corpus_model "fem": the corpus's own canal nodes (meta corpus node set cfg corpus_canal_tie_set, with their
    arclengths in "<set>_s") tied onto the tube exactly as canal_tie_mode "centre" ties the cervix's -- a lateral
    correction, the target at most corpus_canal_max_offset_mm from the node's start-of-step position (or, cfg
    corpus_tie_follow, from its position carried by this step's pose-target change), depth engagement, about the ROW's tube axis (the S1a
    fix; the corpus ties are new, so they never take the pre-S1 "final" axis).  Same pattern as (b): a solver-less target
    MO /targets/corpus_canal_tgt written by the controller + a RestShapeSpringsForceField in the corpus node with zero
    stiffness until a node engages.  The build fails, not the run, when the set or the rows' tip_s are missing."""
    cfg, inp, X0 = ctx["cfg"], ctx["inp"], ctx["X0"]
    ctx["corpus_canal"] = np.zeros(0, int)
    if not cfg.get("corpus_canal_tie", True):
        ctx["extra"].setdefault("corpus_fem", {})["canal_tie"] = dict(on=False)
        return
    ns_k = inp["meta"]["corpus"]["node_sets"]
    name = cfg.get("corpus_canal_tie_set", "canal_path")
    if name not in ns_k or (name + "_s") not in ns_k:
        raise ValueError("corpus_canal_tie_set %r (and %r_s) is not a corpus node set (have %s); it is written by "
                         "`python -P hybrid/tandem_path.py corpus_nodesets` (or set corpus_canal_tie false)"
                         % (name, name, sorted(ns_k)))
    idx = np.asarray(ns_k[name], int)
    cs = np.asarray(ns_k[name + "_s"], float)
    if len(cs) != len(idx):
        raise ValueError("corpus node sets %r (%d) and %r_s (%d) differ in length" % (name, len(idx), name, len(cs)))
    n_all = len(idx)
    smax = cfg.get("corpus_canal_s_max")
    if smax is not None:
        keep = cs <= float(smax)
        idx, cs = idx[keep], cs[keep]
    miss = [i for i, r in enumerate(ctx["sched"]) if r["phase"] not in ("B", "P") and r.get("tip_s") is None]
    if miss:
        raise ValueError("the corpus canal ties engage by depth and need schedule rows carrying tip_s (insertion_path "
                         "'tandem_first'); %d rows lack it, first %d (phase %s).  Set corpus_canal_tie false otherwise."
                         % (len(miss), miss[0], ctx["sched"][miss[0]]["phase"]))
    tips = [float(r["tip_s"]) for r in ctx["sched"] if r.get("tip_s") is not None]
    tmax = max(tips) if tips else float("nan")
    cs_eng = clip_canal_s(cs, tmax) if (cfg.get("corpus_canal_engage_clip", True) and tips) else cs.copy()
    axial = cfg.get("corpus_canal_axial", "none")
    if axial not in ("none", "arclength"):
        raise ValueError("corpus_canal_axial must be 'none' or 'arclength' (got %r)" % axial)
    ctx["corpus_canal"], ctx["corpus_canal_s"], ctx["corpus_canal_s_eng"] = idx, cs, cs_eng
    ctx["extra"].setdefault("corpus_fem", {})["canal_tie"] = dict(
        on=True, set=name, n=int(len(idx)), n_in_set=int(n_all), s_max=smax,
        canal_s_range=[round(float(cs.min()), 3), round(float(cs.max()), 3)] if len(cs) else None,
        schedule_tip_s_max=round(tmax, 3) if tips else None,
        n_clipped=int(np.sum(cs_eng < cs)), k_mN_per_mm=float(cfg["k_corpus_canal_mN_per_mm"]),
        max_offset_mm=float(cfg["corpus_canal_max_offset_mm"]), axial=axial,
        force_bound_per_node_mN=round(float(cfg["k_corpus_canal_mN_per_mm"]) * float(cfg["corpus_canal_max_offset_mm"]), 3))
    if corpus_tie_follow_on(cfg):                       # eu2: only when set, so TF1u's summary is unchanged
        ctx["extra"]["corpus_fem"]["canal_tie"]["follow"] = "pose_target_increment"
    if not len(idx):
        return
    ctx["corpus_canal_tgt"] = ctx["targets"].addObject("MechanicalObject", name="corpus_canal_tgt", template="Vec3d",
                                                       position=X0["corpus"][idx].tolist())
    ctx["corpus_canal_ff"] = ctx["corpus_fem"].addObject("RestShapeSpringsForceField", name="canal_tie",
                                                         points=[int(i) for i in idx], stiffness=[0.0] * len(idx),
                                                         external_rest_shape="@/targets/corpus_canal_tgt",
                                                         external_points=list(range(len(idx))))


def _add_corpus_pose(ctx):
    """corpus_model "fem": the gross pose.  Springs (RestShapeSpringsForceField "pose", per-node stiffness) from the
    cfg corpus_pose_set nodes (+ the interface_cervix bottom at k_corpus_bottom_mN_per_mm) to a solver-less target MO
    /targets/corpus_pose_tgt that the controller rewrites each step to corpus_pose_targets(X0, T_corpus(row), stretch):
    the same kinematic target the rigid corpus is placed on, so a stiff enough pose set reproduces the rigid corpus.
    Zero force at rest (the target starts at the rest positions)."""
    cfg, inp, X0 = ctx["cfg"], ctx["inp"], ctx["X0"]
    canal_pts = None
    if float(cfg.get("corpus_pose_exclude_canal_mm") or 0.0) > 0.0:
        import tandem_path as TP                        # numpy-only at import (py3.8-safe); reads inputs/tandem_path.npz
        T = TP.load()
        canal_pts = np.asarray(T["pts"], float)[np.asarray(T["s"], float) >= 0.0]
    sel = corpus_pose_nodes(inp["meta"]["corpus"], cfg, len(X0["corpus"]), X0["corpus"], canal_pts)
    idx, k = sel["idx"], sel["k"]
    tmo = ctx["targets"].addObject("MechanicalObject", name="corpus_pose_tgt", template="Vec3d",
                                   position=X0["corpus"][idx].tolist())
    ff = ctx["corpus_fem"].addObject("RestShapeSpringsForceField", name="pose", points=[int(i) for i in idx],
                                     stiffness=[float(v) for v in k], external_rest_shape="@/targets/corpus_pose_tgt",
                                     external_points=list(range(len(idx))))
    lam = float(cfg.get("corpus_pose_stretch_lam", 1.0))
    ctx["corpus_pose"] = dict(idx=idx, k=k, serosa=sel["serosa"], bottom=sel["bottom"], tgt_mo=tmo, ff=ff, lam=lam)
    e = ctx["extra"].setdefault("corpus_fem", {})
    e["pose"] = dict(sel["info"], stretch_lam=lam, total_k_N_per_mm=round(float(k.sum()) / 1000.0, 4),
                     tag=("CALIBRATED (stretch fitted to this patient's BT uterus)" if lam != 1.0 else "rigid target"))
    ff_ = int(cfg.get("frame_every") or 0)
    if ff_ > 0 and "corpus" not in (cfg.get("frame_u_bodies") or []):
        w = "frame_every %d without 'corpus' in frame_u_bodies: tf_metrics / label_views carry the canal through the " \
            "corpus by an approximate (Kabsch) fit per frame; set frame_u_bodies [\"cervix\", \"corpus\"]" % ff_
        e.setdefault("warnings", []).append(w)
        print("[scene_hybrid] WARNING " + w, flush=True)


def _add_corpus_follow(ctx):
    """corpus_model "fem", corpus_oar_contact "follow": a solver-less copy of the corpus surface that the OARs collide
    with -- surface.obj's faces over the corpus's surface nodes (s2n), the same collision groups, moving=True,
    simulated=False as the rigid corpus's surface.  At rest it IS the rigid collision surface (max |V - X0[s2n]| is
    recorded).  The controller rewrites it each step (follow_surface): the corpus's surface nodes at the start of the
    step plus this step's change of the pose target, so a corpus that follows its target is met exactly where the rigid
    one was.  One-way: nothing acts back on the corpus."""
    cfg = ctx["cfg"]
    mode = cfg.get("corpus_oar_contact", "follow")
    if mode not in ("follow", "two_way", "off"):
        raise ValueError("corpus_oar_contact must be 'follow', 'two_way' or 'off' (got %r)" % mode)
    ctx["corpus_follow"] = None
    ctx["extra"].setdefault("corpus_fem", {})["oar_contact"] = dict(mode=mode)
    if mode != "follow":
        return
    P = ctx["inp"]["P"]
    V, Fs = geom.read_obj("%s/corpus/surface.obj" % P["meshes"])
    s2n = np.asarray(ctx["inp"]["meta"]["corpus"]["surface_obj_vertex_to_tet_node"], int)
    Xs = ctx["X0"]["corpus"][s2n]
    nd = ctx["root"].addChild("corpus_follow")
    mo = nd.addObject("MechanicalObject", name="mo", template="Vec3d", position=Xs.tolist())
    nd.addObject("MeshTopology", name="mt", position=Xs.tolist(), triangles=np.asarray(Fs, int).tolist())
    g = list(groups_for(cfg)["corpus"])
    kw = dict(moving=True, simulated=False)
    kw.update(_prox_kw(cfg, "corpus"))
    nd.addObject("TriangleCollisionModel", name="tri", group=g, **kw)
    nd.addObject("LineCollisionModel", name="lin", group=g, **kw)
    nd.addObject("PointCollisionModel", name="pnt", group=g, **kw)
    ctx["corpus_follow"] = dict(node=nd, mo=mo, s2n=s2n)
    ctx["extra"]["corpus_fem"]["oar_contact"].update(
        n_vertices=int(len(s2n)), n_triangles=int(len(Fs)), groups=g,
        rest_vs_surface_obj_max_mm=round(float(np.abs(np.asarray(V, float) - Xs).max()), 6))


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
    # S2 P3: an optionally SUBDIVIDED copy (the first len(outer) vertices are the outer nodes themselves, the rest edge
    # midpoints rewritten from them every step); S2 P2: an optional per-model proximity.  Defaults: G32's balloon.
    tri_loc = rem[tri]
    lv = int(cfg.get("balloon_subdivide", 0) or 0)
    tri_b, rounds = subdivide_tris(tri_loc, lv) if lv > 0 else (tri_loc, [])
    P_start = apply_subdiv(X_start, rounds)
    pkw = {}
    if "balloon" in (cfg.get("collision_proximity_mm") or {}):
        pkw = dict(proximity=float(cfg["collision_proximity_mm"]["balloon"]))
    nd = ctx["root"].addChild("balloon")
    nd.addObject("MechanicalObject", name="mo", template="Vec3d", position=P_start.tolist())
    nd.addObject("MeshTopology", name="mt", position=P_start.tolist(), triangles=tri_b.tolist())
    models = [nd.addObject("TriangleCollisionModel", name="tri", group=g_on, moving=True, simulated=False, **pkw),
              nd.addObject("LineCollisionModel", name="lin", group=g_on, moving=True, simulated=False, **pkw),
              nd.addObject("PointCollisionModel", name="pnt", group=g_on, moving=True, simulated=False, **pkw)]
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
                          drive=bool(W_start is not None), rounds=rounds, cur=X_start.copy(), tri_loc=tri_loc,
                          station=k.copy(), t_lo=Tg[0].copy(), t_hi=Tg[-1].copy())
    if lv > 0 or pkw:
        ctx["extra"]["balloon_probe"] = dict(subdivide=lv, vertices=int(len(P_start)), triangles=int(len(tri_b)),
                                             proximity_mm=pkw.get("proximity"))
    ctx["extra"]["balloon"] = dict(n_steps=nB, nodes=int(len(outer)), triangles=int(len(tri)),
                                   start_semi_axes_mm=[A, B],
                                   start_radius_mm=dict(min=round(float(r0.min()), 3), max=round(float(r0.max()), 3)),
                                   travel_mm=dict(max=round(float(np.linalg.norm(Xo - X_start, axis=1).max()), 3),
                                                  mean=round(float(np.linalg.norm(Xo - X_start, axis=1).mean()), 3)),
                                   groups=dict(balloon_on=g_on, balloon_off=g_off, wall_outer_on=wall_on,
                                               wall_outer_off=wall_off))


def _add_sheet_penalty(ctx):
    """S2: the OAR-vs-sheet monitor (cfg oar_sheet_monitor) and the P4 penalty springs (oar_sheet_penalty_mN_per_mm
    > 0) on the surface nodes of cfg oar_sheet_bodies.  Same pattern as the canal tie: a solver-less target MO written
    by the controller + a RestShapeSpringsForceField with per-node stiffness (0 = inactive).  Added before the
    constraint corrections (build_scene adds those last)."""
    cfg = ctx["cfg"]
    k = float(cfg.get("oar_sheet_penalty_mN_per_mm", 0.0) or 0.0)
    if not (cfg.get("oar_sheet_monitor") or k > 0.0):
        return
    b = ctx.get("balloon")
    if b is None:
        raise ValueError("oar_sheet_monitor / oar_sheet_penalty need the balloon (vagina_model 'wall', n_balloon > 0)")
    inp, X0 = ctx["inp"], ctx["X0"]
    w = inp["meta"]["vagina"]["wall"]
    C = np.asarray(w["centreline"], float)
    # outward = away from the lumen centreline, fixed once from the REST sheet (the winding does not change)
    Xe = b["X_end"]
    rad = np.einsum("ij,ij->i", vertex_normals(Xe, b["tri_loc"]), Xe - C[b["station"]])
    sign = 1.0 if float(np.mean(rad)) >= 0.0 else -1.0
    bodies = {}
    for name in cfg.get("oar_sheet_bodies", []):
        if name not in ctx["deformable"]:
            continue
        idx = np.asarray(inp["meta"][name]["node_sets"]["surface_nodes"], int)
        e = dict(idx=idx)
        if k > 0.0:
            e["tgt"] = ctx["targets"].addObject("MechanicalObject", name="pen_" + name, template="Vec3d",
                                                position=X0[name][idx].tolist())
            e["ff"] = ctx["nodes"][name].addObject("RestShapeSpringsForceField", name="sheet_pen",
                                                   points=[int(i) for i in idx], stiffness=[0.0] * len(idx),
                                                   external_rest_shape="@/targets/pen_" + name,
                                                   external_points=list(range(len(idx))))
        bodies[name] = e
    ctx["sheet_pen"] = dict(bodies=bodies, k=k, margin=float(cfg["oar_sheet_margin_mm"]),
                            cap=float(cfg["oar_sheet_max_offset_mm"]), reach=25.0, sign=sign, n_ax=int(w["n_axial"]))
    ctx["extra"]["oar_sheet"] = dict(bodies={n: int(len(e["idx"])) for n, e in bodies.items()}, penalty_k=k,
                                     margin_mm=float(cfg["oar_sheet_margin_mm"]),
                                     max_offset_mm=float(cfg["oar_sheet_max_offset_mm"]),
                                     normal_outward_frac=round(float(np.mean(sign * rad > 0.0)), 4))


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
             n_sched=len(ctx["sched"]),
             phases=dict({p: sum(1 for r in ctx["sched"] if r["phase"] == p) for p in "BPATD"},     # + the tandem-first
                         **{p: sum(1 for r in ctx["sched"] if r["phase"] == p) for p in "VCL"     # phases, only when
                            if any(r["phase"] == p for r in ctx["sched"])}),                       # present (G32 as was)
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
    if ctx.get("corpus_fem") is None:
        s["bodies"]["corpus"] = dict(nodes=int(len(ctx["X0"]["corpus"])), role="rigid kinematic")
    else:
        e = ctx["extra"].get("corpus_fem", {})
        lam = float(cfg.get("corpus_pose_stretch_lam", 1.0))
        s["bodies"]["corpus"] = dict(
            nodes=int(len(ctx["X0"]["corpus"])), role="fem (elastic; gross pose by springs to the pose-rule target)",
            E_kPa=cfg["E_kPa"]["corpus"], nu=cfg["nu"]["corpus"],
            material=cfg.get("material_by_body", {}).get("corpus", cfg["material"]),
            pose=e.get("pose"), canal_tie=e.get("canal_tie"), oar_contact=e.get("oar_contact"),
            coupling=cfg["attach_impl"] + (" (two-way)" if cfg["attach_impl"] == "bilateral" else " (one-way)"),
            attach_target=cfg.get("attach_target", "mapped"),
            stretch_lam=lam, tags=(["CALIBRATED"] if lam != 1.0 else []) + ["S9 elastic corpus"])
        s["corpus_model"] = "fem"
    return s


# ----------------------------------------------------------------------------- pure helpers (no SOFA; host-testable)
# The canal tie (S1) and the contact probes (S2) keep their arithmetic here, out of the controller, so that
# hybrid/test_ties.py can run it on the host against synthetic rows and against the pre-S1 code.
def canal_tie_params(cfg, r_tube, L_iu):
    """The per-run constants of the canal tie, read once from the cfg."""
    return dict(rad=float(r_tube) + float(cfg["canal_slack_mm"]), L_iu=float(L_iu),
                centre=cfg.get("canal_tie_mode", "dilate") == "centre",
                below=float(cfg.get("canal_tie_below_mm", 0.0) or 0.0),
                engage=cfg.get("canal_engage", "radius"), engage_mm=float(cfg.get("canal_engage_mm", 3.0)),
                cap=float(cfg["canal_max_offset_mm"]), ramp=float(cfg["tie_ramp_steps"]),
                k=float(cfg["k_canal_mN_per_mm"]))


def rod_dir_app(app):
    """Unit direction of the tandem's vaginal rod in the APPLICATOR frame, pointing UP (towards the flange): minus
    landmarks.shaft_end_dir (applicator_v3: [0, -0.483, 0.875], the straight rod 28.9 deg from the tube).  Without
    that landmark, the tube axis itself."""
    lm = app.get("landmarks", {})
    if lm.get("shaft_end_dir") is not None:
        return geom.unit(-np.asarray(lm["shaft_end_dir"], float))
    return np.array([0.0, 0.0, 1.0])


def tie_axes(cfg, tgt, row, rod_app):
    """(at, ar): the unit lines the canal tie is measured about -- the tube axis through the flange, and the rod
    axis for the below-flange nodes.  cfg canal_tie_axis "final" returns exactly the arrays the pre-S1 code used
    (tgt tube_axis / tgt axis); "row" the current row's tube axis and rod axis (see CFG)."""
    if cfg.get("canal_tie_axis", "final") != "row":
        return tgt["tube_axis"], np.asarray(tgt["axis"], float)
    at = geom.unit(np.asarray(row["tube_axis"], float))
    if cfg.get("canal_rod_axis", "shaft") == "tube":
        return at, at
    return at, geom.unit(np.asarray(rod_app, float) @ np.asarray(row["R_rows"], float))   # rows convention


def canal_tie_step(X, F, at, ar, eng_step, k, p, tip_s=None, canal_s=None, carry=None):
    """One step of the canal tie: which nodes are engaged / active, their spring targets and stiffnesses.

    X (n, 3) the tie nodes' current positions; F the current flange; at, ar the unit tube and rod lines (tie_axes);
    eng_step (n,) int, the step each node engaged at (-1 = not yet), UPDATED IN PLACE (a node stays engaged); k this
    step's index; p = canal_tie_params(...); tip_s the row's tip arclength along the tandem path and canal_s (n,) the
    nodes' own (depth engagement only).  Returns dict(tgt, k, act, inside, d, canal_d).

    With p engage "radius" this is, operation for operation, the block HybridController._begin ran before S1 (so the
    default path is bit-identical; hybrid/test_ties.py compares the two): target = the node moved radially to the
    tube surface at its own axial level, at most `cap` from where it is; "centre" drives both ways, "dilate" only out.

    carry (n, 3) or None (eu2, cfg canal_tie_follow / corpus_tie_follow): this step's carried motion of the nodes
    (canal_tie_carry / the pose-target change).  Everything above is then evaluated at the CARRIED position X + carry:
    the target is where the node is carried to plus the capped lateral correction from there, so `cap` bounds the misfit
    only, and a node carried rigidly with the tube keeps zero spring force (inside / d / canal_d are measured there as
    well).  carry None = the arithmetic above, untouched (X is not even copied)."""
    if carry is not None:
        X = np.asarray(X, float) + np.asarray(carry, float)
    q = X - F
    s = q @ at
    lat = q - np.outer(s, at)
    d = np.linalg.norm(lat, axis=1)
    inside = (s >= 0.0) & (s <= p["L_iu"])              # the tube currently occupies this axial level
    centre, below = p["centre"], p["below"]
    if centre and below > 0.0:
        # the os region below the flange: lateral offset from the ROD line, not the tube
        sr = q @ ar
        latr = q - np.outer(sr, ar)
        dr = np.linalg.norm(latr, axis=1)
        low = (sr < 0.0) & (sr >= -below)
        inside = inside | low
        lat = np.where(low[:, None], latr, lat)
        d = np.where(low, dr, d)
        s = np.where(low, sr, s)
    if centre and p["engage"] == "depth":               # engage once the TIP has passed the node along the path
        reached = (np.asarray(canal_s, float) <= float(tip_s)) if tip_s is not None else np.zeros(len(X), bool)
        new = reached & (eng_step < 0)
    elif centre:                                        # engage only once the tube has REACHED the node
        new = inside & (d < p["rad"] + p["engage_mm"]) & (eng_step < 0)
    else:
        new = inside & (eng_step < 0)
    eng_step[new] = k
    act = (inside & (eng_step >= 0)) if centre else (inside & (d < p["rad"]))   # dilate: never pull inwards
    u = np.where(d[:, None] > 1e-9, lat / np.maximum(d, 1e-9)[:, None], _perp(at)[None, :])
    # (below the flange `u` is radial about the rod line; above it about the tube line -- both lateral)
    cap = p["cap"]
    off = np.clip(p["rad"] - d, -cap, cap) if centre else np.maximum(np.minimum(p["rad"] - d, cap), 0.0)
    tgt = np.where(act[:, None], X + off[:, None] * u, X)
    ramp = np.clip((k - eng_step + 1) / p["ramp"], 0.0, 1.0)
    kk = np.where(act, p["k"] * ramp, 0.0)
    # canal-to-tube distance of the nodes inside the tube's span, about the SAME line the tie aims at: the number that
    # says whether the tube is IN the canal or beside it (MEASURED G17: the centring tie engaged 0 nodes until u = 0.81
    # because every canal node was > tube radius + 3 mm from the tube axis until then)
    canal_d = (dict(n_in_span=int(inside.sum()), min=round(float(d[inside].min()), 3),
                    median=round(float(np.median(d[inside])), 3), max=round(float(d[inside].max()), 3))
               if inside.any() else dict(n_in_span=0))
    return dict(tgt=tgt, k=kk, act=act, inside=inside, d=d, canal_d=canal_d)


# ---- eu2 (review HIGH-2): the tie target carried with the tissue.  Pure, unit-tested in test_ties.py / test_corpus.py.
CANAL_TIE_FOLLOW = ("corpus", "tube")
CANAL_TIE_FOLLOW_ATTACH = ("springs", "predicted")      # (attach_impl, attach_target): the only lag-free attach (eu3)


def cfg_flag(cfg, key, default=False):
    """A boolean cfg key, checked: true / false (null = the default).  0 / 1, strings etc. are refused, so a typo is a
    build error rather than a silently different run."""
    m = cfg.get(key, default)
    if m is None:
        return bool(default)
    if not isinstance(m, bool):
        raise ValueError("%s must be true or false (got %r)" % (key, m))
    return m


def canal_tie_follow_mode(cfg):
    """cfg canal_tie_follow, checked: None (off: false / null / "off", the default) or one of CANAL_TIE_FOLLOW.

    A follow mode is REFUSED (ValueError, at build time and in the controller) unless
      - canal_tie is true (otherwise there are no cervix ties to carry: a silent no-op), and
      - the attach is lag-free: attach_impl "springs" with attach_target "predicted" (CANAL_TIE_FOLLOW_ATTACH).  The
        carry puts the tie nodes on row k's corpus pose, while "mapped" holds the cervix interface on row k-2 and
        AttachConstraint + "predicted" on row k-1 (see CFG attach_target); the junction between them is sheared
        (review eu2 HIGH-1: TF1n_noC, "corpus" on the mapped attach, aborted with inverted junction tets);
    and "tube" is refused with below-flange ties (canal_tie_mode "centre" + canal_tie_below_mm > 0): those nodes lie
    on the rod line, whose motion tube_lateral_increment does not describe."""
    m = cfg.get("canal_tie_follow", False)
    if m is None or m is False or (isinstance(m, str) and m == "off"):
        return None
    if not isinstance(m, str) or m not in CANAL_TIE_FOLLOW:
        raise ValueError("canal_tie_follow must be false, 'corpus' or 'tube' (got %r)" % (m,))
    if not cfg.get("canal_tie", True):
        raise ValueError("canal_tie_follow %r carries the cervix canal ties, but canal_tie is false (it would do nothing): "
                         "drop canal_tie_follow or set canal_tie true" % m)
    att = (cfg.get("attach_impl", "attach"), cfg.get("attach_target", "mapped"))
    if att != CANAL_TIE_FOLLOW_ATTACH:
        raise ValueError("canal_tie_follow %r needs the lag-free attach: attach_impl %r with attach_target %r (got %r / "
                         "%r).  The carry moves the canal ties with this row's corpus pose while a lagged attach holds "
                         "the cervix interface one or two rows behind, which shears the junction (TF1n_noC: 'corpus' "
                         "on the mapped attach aborted at step 241 with inverted junction tets)."
                         % ((m,) + CANAL_TIE_FOLLOW_ATTACH + att))
    if m == "tube" and cfg.get("canal_tie_mode", "dilate") == "centre" and \
            float(cfg.get("canal_tie_below_mm", 0.0) or 0.0) > 0.0:
        raise ValueError("canal_tie_follow 'tube' is not defined for below-flange ties (canal_tie_below_mm %r > 0: those "
                         "nodes are tied to the rod line, not the tube); use 'corpus' or canal_tie_below_mm 0"
                         % cfg.get("canal_tie_below_mm"))
    return m


def check_follow_cfg(cfg):
    """Build-time check of the eu2 keys (no graph object, no summary entry): canal_tie_follow_mode, corpus_tie_follow_on
    and log_canal_tie_force (a bool; refused with canal_tie false, where there is no tie force to log).  Returns
    (canal_tie_follow mode or None, corpus_tie_follow bool)."""
    mode = canal_tie_follow_mode(cfg)
    kf = corpus_tie_follow_on(cfg)
    if cfg_flag(cfg, "log_canal_tie_force") and not cfg.get("canal_tie", True):
        raise ValueError("log_canal_tie_force is true but canal_tie is false (no cervix tie force to log)")
    return mode, kf


def corpus_tie_follow_on(cfg):
    """cfg corpus_tie_follow, checked: a bool (default False).  True is refused where it would do nothing: with a rigid
    corpus (corpus_model "rigid" has no corpus canal ties) or with corpus_canal_tie false."""
    m = cfg_flag(cfg, "corpus_tie_follow")
    if m and corpus_model(cfg) != "fem":
        raise ValueError("corpus_tie_follow carries the elastic corpus's canal ties; corpus_model is %r, which has none "
                         "(drop corpus_tie_follow or set corpus_model 'fem')" % corpus_model(cfg))
    if m and not cfg.get("corpus_canal_tie", True):
        raise ValueError("corpus_tie_follow is true but corpus_canal_tie is false (no corpus ties to carry)")
    return m


def rigid_step_increment(X, T_prev, T_now):
    """This step's rigid motion at the points X: X carried by T_now o T_prev^-1, minus X.  T_prev / T_now = 4x4
    placements in the convention of corpus_pose_targets (x = X0 @ T[:3, :3]^T + T[:3, 3]).  Exactly zero when the two
    are equal (the corpus at rest: phases B, P, V and every settle row)."""
    X = np.asarray(X, float)
    Ta, Tb = np.asarray(T_prev, float), np.asarray(T_now, float)
    if np.array_equal(Ta, Tb):
        return np.zeros_like(X)
    return (X - Ta[:3, 3]) @ Ta[:3, :3] @ Tb[:3, :3].T + Tb[:3, 3] - X


def tube_lateral_increment(X, R_prev, F_prev, R_now, F_now, a):
    """The tandem's rigid motion (R_rows ROWS = applicator axes in world, F = flange; x = x_app @ R + F) from the
    previous row to this one at the points X, with its component along the unit line `a` (the tie line) removed: how the
    tube moves sideways at X.  The axial part is the tube sliding along the canal (the insertion itself), which tissue
    around a frictionless tube does not follow.  Exactly zero when the device pose is unchanged."""
    X = np.asarray(X, float)
    Ra, Rb = np.asarray(R_prev, float), np.asarray(R_now, float)
    Fa, Fb = np.asarray(F_prev, float), np.asarray(F_now, float)
    if np.array_equal(Ra, Rb) and np.array_equal(Fa, Fb):
        return np.zeros_like(X)
    D = (X - Fa) @ Ra.T @ Rb + Fb - X
    a = np.asarray(a, float)
    return D - np.outer(D @ a, a)


def canal_tie_carry(mode, X, r_prev, r_now):
    """The cervix tie nodes' carried motion this step (cfg canal_tie_follow `mode`, as canal_tie_follow_mode returns
    it), for canal_tie_step(carry=).  "corpus": rigid_step_increment of T_corpus row k-1 -> k at the nodes.  "tube":
    tube_lateral_increment about the ROW's tube axis (r_now tube_axis: the line the tube actually slides along,
    whatever line the tie aims at -- canal_tie_axis "final" included) + the part of that corpus increment along the
    same axis.  r_prev None (the first step): zero.  Every schedule row carries T_corpus, R_rows, F and tube_axis."""
    X = np.asarray(X, float)
    if mode is None:
        return None
    if r_prev is None:
        return np.zeros_like(X)
    Dc = rigid_step_increment(X, r_prev["T_corpus"], r_now["T_corpus"])
    if mode == "corpus":
        return Dc
    if mode == "tube":
        a = geom.unit(np.asarray(r_now["tube_axis"], float))
        Dt = tube_lateral_increment(X, r_prev["R_rows"], r_prev["F"], r_now["R_rows"], r_now["F"], a)
        return Dt + np.outer(Dc @ a, a)
    raise ValueError("canal_tie_follow must be false, 'corpus' or 'tube' (got %r)" % (mode,))


def tie_force_log(tgt, k, X_end, carry=None):
    """The spring force of a tie set at the END of a step, N (k mN/mm, target and node positions mm): the magnitude of
    the net force, the largest per-node force, the summed per-node magnitudes, and the largest carry (mm)."""
    f = np.asarray(k, float)[:, None] * (np.asarray(tgt, float) - np.asarray(X_end, float))
    fn = np.linalg.norm(f, axis=1)
    out = dict(net_N=round(float(np.linalg.norm(f.sum(0))) / 1000.0, 4),
               max_node_N=round(float(fn.max()) / 1000.0 if len(fn) else 0.0, 4),
               sum_abs_N=round(float(fn.sum()) / 1000.0, 4), n_loaded=int((np.asarray(k) > 0).sum()))
    if carry is not None:
        cn = np.linalg.norm(np.asarray(carry, float), axis=1)
        out["carry_max_mm"] = round(float(cn.max()) if len(cn) else 0.0, 4)
    return out


# ---- S9: the elastic corpus (corpus_model "fem").  Pure arithmetic, unit-tested in hybrid/test_corpus.py.
def corpus_model(cfg):
    """The cfg's corpus model, checked: "rigid" (default, the kinematic Rigid3d corpus) or "fem"."""
    m = cfg.get("corpus_model", "rigid")
    if m not in ("rigid", "fem"):
        raise ValueError("corpus_model must be 'rigid' or 'fem' (got %r)" % m)
    return m


def read_vtk_tets(path):
    """(points, tets) of a legacy ASCII VTK tet grid written by mesh_bodies (vagina_wall.read_vtk_legacy, numpy only)."""
    with open(path) as fh:
        tok = fh.read().split()
    i = tok.index("POINTS")
    n = int(tok[i + 1])
    P = np.array(tok[i + 3:i + 3 + 3 * n], float).reshape(n, 3)
    j = tok.index("CELLS", i)
    m = int(tok[j + 1])
    return P, np.array(tok[j + 3:j + 3 + 5 * m], int).reshape(m, 5)[:, 1:]


def bary_weights(P, T, Q):
    """Barycentric weights of the points Q in the tets (P, T) by SOFA v22.12's BarycentricMapperTetrahedronSetTopology
    rule: v = the tet's local coordinates, d = max(-v0, -v1, -v2, v0 + v1 + v2 - 1) (<= 0 inside); outside every tet
    (d > 0) the tet with the nearest centroid, extrapolated; the smallest d wins.  Returns (tet node indices (m, 4),
    weights (m, 4)).  MEASURED: reproduces SOFA's mapped /corpus/iface/mo of run EUISO_B to 1e-5 mm (the logged
    attach residual), and the rest points exactly (1e-14 mm)."""
    P, T = np.asarray(P, float), np.asarray(T, int)
    A = P[T]
    B = np.linalg.inv(np.transpose(A[:, 1:] - A[:, :1], (0, 2, 1)))
    C = A.mean(1)
    Q = np.atleast_2d(np.asarray(Q, float))
    idx, W = np.zeros((len(Q), 4), int), np.zeros((len(Q), 4))
    for i, q in enumerate(Q):
        v = np.einsum("tij,tj->ti", B, q - A[:, 0])
        d = np.maximum(np.maximum(-v[:, 0], -v[:, 1]), np.maximum(-v[:, 2], v.sum(1) - 1.0))
        d = np.where(d > 0.0, ((q - C) ** 2).sum(1), d)
        t = int(np.argmin(d))
        idx[i], W[i] = T[t], np.r_[1.0 - v[t].sum(), v[t]]
    return idx, W


def bary_apply(X, idx, W):
    """Points carried by bary_weights (idx, W) from the node positions X."""
    return np.einsum("ij,ijk->ik", np.asarray(W, float), np.asarray(X, float)[np.asarray(idx, int)])


def _polyline_dist(X, pts):
    """Distance of each X to the polyline pts (segment by segment; tandem_path.project without the arclength)."""
    X = np.atleast_2d(np.asarray(X, float))
    pts = np.asarray(pts, float)
    best = np.full(len(X), np.inf)
    for a, b in zip(pts[:-1], pts[1:]):
        ab = b - a
        l2 = float(ab @ ab)
        t = np.clip(((X - a) @ ab) / l2, 0.0, 1.0) if l2 > 0 else np.zeros(len(X))
        best = np.minimum(best, np.linalg.norm(X - (a + t[:, None] * ab), axis=1))
    return best


def corpus_pose_nodes(meta, cfg, n_nodes, X0=None, canal_pts=None):
    """The elastic corpus's pose springs: node indices (sorted) and per-node stiffness (mN/mm) from cfg
    corpus_pose_set / k_corpus_pose_mN_per_mm / k_corpus_bottom_mN_per_mm / corpus_pose_exclude_canal_mm (see CFG).
    meta = the corpus meta.json (node_sets surface_nodes, interface_cervix); X0 (n, 3) rest nodes and canal_pts (the
    rest labelled canal polyline) are needed only for the canal exclusion.  Returns dict(idx, k, serosa, bottom, info)."""
    ns = meta["node_sets"]
    surf = np.unique(np.asarray(ns["surface_nodes"], int))
    bot = np.unique(np.asarray(ns["interface_cervix"], int))
    serosa = np.setdiff1d(surf, bot)
    mode = cfg.get("corpus_pose_set", "serosa")
    if mode in ("serosa", "shell"):
        P = serosa.copy()
    elif mode == "surface":
        P = surf.copy()
    elif mode == "all":
        P = np.arange(int(n_nodes))
    else:
        raise ValueError("corpus_pose_set must be 'serosa', 'surface', 'all' or 'shell' (got %r)" % mode)
    k = float(cfg["k_fixed_mN_per_mm"]) if mode == "shell" else float(cfg["k_corpus_pose_mN_per_mm"])
    R = float(cfg.get("corpus_pose_exclude_canal_mm") or 0.0)
    n_excl = 0
    if R > 0.0:
        if X0 is None or canal_pts is None:
            raise ValueError("corpus_pose_exclude_canal_mm > 0 needs the rest nodes and the rest labelled canal")
        drop = _polyline_dist(np.asarray(X0, float)[P], canal_pts) <= R
        n_excl = int(drop.sum())
        P = P[~drop]
    kk = np.full(len(P), k)
    kb = cfg.get("k_corpus_bottom_mN_per_mm")
    n_bot = 0
    if kb is not None:
        kb = float(kb)
        kk[np.isin(P, bot)] = kb                        # a bottom node already in the set takes the bottom stiffness
        extra = np.setdiff1d(bot, P)
        P = np.r_[P, extra]
        kk = np.r_[kk, np.full(len(extra), kb)]
        n_bot = int(len(bot))
    o = np.argsort(P, kind="stable")
    P, kk = P[o], kk[o]
    info = dict(set=mode, n=int(len(P)), n_serosa=int(len(serosa)), n_surface=int(len(surf)), n_bottom_nodes=int(len(bot)),
                n_bottom_springs=n_bot, k_mN_per_mm=k, k_bottom_mN_per_mm=kb, exclude_canal_mm=R, n_excluded_near_canal=n_excl)
    return dict(idx=P, k=kk, serosa=serosa, bottom=bot, info=info)


def corpus_stretch_at(lam, row):
    """The row's stretch for corpus_pose_targets, or None when there is none (lam 1: the rigid target, bit for bit).
    lam_row = 1 + (lam - 1) w, w = the row's w_r (tandem-first: 0 through V, the rotation weight through C, 1 in L / H),
    else its screw fraction s; the axis is the row's tube axis."""
    lam = float(lam)
    if lam == 1.0:
        return None
    w = row.get("w_r")
    w = float(row["s"]) if w is None else float(w)
    return dict(lam=1.0 + (lam - 1.0) * w, axis=np.asarray(row["tube_axis"], float), w=w)


def corpus_pose_targets(X0, T, stretch=None, idx=None):
    """The pose target of the corpus nodes: X0 moved by the 4x4 T (x T[:3, :3]^T + T[:3, 3], exactly the rigid corpus's
    placement in run_hybrid / write_frame), then -- stretch = corpus_stretch_at(...) not None -- a volume-preserving
    stretch along stretch["axis"] about the axis-parallel line through the posed nodes' centroid c, anchored at h0 = the
    0.5 percentile of their axial coordinate h (the corpus bottom): h' = h0 + lam (h - h0), lateral offsets / sqrt(lam)
    (det = lam / sqrt(lam)^2 = 1).  The eu_design m5 construction.  idx: return only those nodes (the stretch is always
    computed on ALL nodes, so the anchor and centroid do not depend on the subset)."""
    T = np.asarray(T, float)
    X = np.asarray(X0, float) @ T[:3, :3].T + T[:3, 3]
    if stretch is not None and float(stretch["lam"]) != 1.0:
        lam = float(stretch["lam"])
        a = geom.unit(np.asarray(stretch["axis"], float))
        c = X.mean(0)
        h = (X - c) @ a
        h0 = float(np.percentile(h, 0.5))
        lat = (X - c) - np.outer(h, a)
        X = c + np.outer(h0 + lam * (h - h0), a) + lat / np.sqrt(lam)
    return X if idx is None else X[np.asarray(idx, int)]


def follow_surface(Xs_now, tgt_now, tgt_prev):
    """corpus_oar_contact "follow": where the OARs meet the corpus this step = its surface nodes now (end of the last
    step) plus this step's change of their pose target.  A corpus exactly on its target gives the target itself, i.e.
    the rigid corpus's collision surface."""
    return np.asarray(Xs_now, float) + (np.asarray(tgt_now, float) - np.asarray(tgt_prev, float))


def clip_canal_s(canal_s, tip_s_max, margin=0.05):
    """Depth engagement for nodes the schedule's tip never reaches: engage at min(canal_s, tip_s_max - margin)."""
    return np.minimum(np.asarray(canal_s, float), float(tip_s_max) - float(margin))


def corpus_tie_params(cfg, r_tube, L_iu):
    """canal_tie_params for the corpus canal ties: always "centre" (both ways onto the tube surface), depth engagement,
    no below-flange part, the corpus's own stiffness and per-step bound; radius / ramp shared with the cervix ties."""
    return dict(rad=float(r_tube) + float(cfg["canal_slack_mm"]), L_iu=float(L_iu), centre=True, below=0.0,
                engage="depth", engage_mm=float(cfg.get("canal_engage_mm", 3.0)),
                cap=float(cfg["corpus_canal_max_offset_mm"]), ramp=float(cfg["tie_ramp_steps"]),
                k=float(cfg["k_corpus_canal_mN_per_mm"]))


def axial_tie_targets(res, X, F, at, tip_s, canal_s, L_iu, cap):
    """corpus_canal_axial "arclength": the ACTIVE tie targets of canal_tie_step (res) also moved along the tube axis at
    towards h* = L_iu - (tip_s - canal_s) above the flange F (a straightened canal keeps its arclength behind the tip),
    at most `cap` per step.  Returns (targets, axial offsets); inactive nodes keep res["tgt"]."""
    X = np.asarray(X, float)
    h = (X - np.asarray(F, float)) @ np.asarray(at, float)
    dh = np.clip(float(L_iu) - (float(tip_s) - np.asarray(canal_s, float)) - h, -float(cap), float(cap))
    dh = np.where(res["act"], dh, 0.0)
    return res["tgt"] + dh[:, None] * np.asarray(at, float)[None, :], dh


def subdivide_tris(tri, levels, n_vertices=None):
    """`levels` rounds of 1-to-4 midpoint subdivision of a triangle list (orientation kept).  Returns (tri_new,
    rounds): rounds[i] = (e0, e1), the parent pair of every vertex round i appends, so apply_subdiv can rebuild the
    vertices from the original ones each step.  The subdivided surface is geometrically the SAME piecewise-linear
    surface (every new vertex is an edge midpoint); only the collision primitives get denser.  n_vertices = the
    size of the vertex array the triangles index (default max index + 1; pass it when `tri` is a SUBSET)."""
    T = np.asarray(tri, np.int64)
    n = int(n_vertices) if n_vertices is not None else (int(T.max()) + 1 if len(T) else 0)
    rounds = []
    for _ in range(int(levels)):
        E = np.sort(np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), axis=1)
        Eu, inv = np.unique(E, axis=0, return_inverse=True)
        inv = np.asarray(inv).reshape(-1)
        m = len(T)
        mab, mbc, mca = (n + inv[:m], n + inv[m:2 * m], n + inv[2 * m:])
        a, b, c = T[:, 0], T[:, 1], T[:, 2]
        T = np.concatenate([np.c_[a, mab, mca], np.c_[b, mbc, mab], np.c_[c, mca, mbc], np.c_[mab, mbc, mca]])
        rounds.append((Eu[:, 0].copy(), Eu[:, 1].copy()))
        n += len(Eu)
    return T, rounds


def apply_subdiv(X, rounds):
    """The vertices of a subdivide_tris surface from its original vertices X (the original ones first)."""
    P = np.asarray(X, float)
    for e0, e1 in rounds:
        P = np.concatenate([P, 0.5 * (P[e0] + P[e1])])
    return P


def vertex_normals(X, tri):
    """Area-weighted unit vertex normals of a triangle surface (orientation = the triangles' winding)."""
    X = np.asarray(X, float)
    T = np.asarray(tri, np.int64)
    fn = np.cross(X[T[:, 1]] - X[T[:, 0]], X[T[:, 2]] - X[T[:, 0]])
    N = np.zeros_like(X)
    for j in range(3):
        np.add.at(N, T[:, j], fn)
    return N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)


def sheet_signed_distance(Xq, Xs, N, station, n_ax, t_lo, t_hi, reach=25.0):
    """Signed distance of points Xq to a tube-like SHEET (vertices Xs, OUTWARD unit vertex normals N, station index of
    every vertex, n_ax stations): (Xq - x_j) . n_j with j the nearest sheet vertex.  Positive = outside the sheet.
    Points further than `reach` from every vertex, or beyond an end ring (their offset from x_j points out of the
    sheet along -t_lo at station 0 or +t_hi at station n_ax - 1), are not measured: valid False, sd NaN.
    Nearest-vertex, not nearest-triangle: at G32's ~2 mm outer-sheet spacing and ~20 mm radius the error of the
    NORMAL component is well under 0.1 mm; the lateral miss does not enter."""
    from scipy.spatial import cKDTree              # container scipy 1.10.1 / host scipy
    dist, j = cKDTree(np.asarray(Xs, float)).query(np.asarray(Xq, float), distance_upper_bound=float(reach))
    ok = np.isfinite(dist)
    sd = np.full(len(Xq), np.nan)
    if ok.any():
        jj = j[ok]
        v = np.asarray(Xq, float)[ok] - np.asarray(Xs, float)[jj]
        st = np.asarray(station)[jj]
        beyond = ((st == 0) & (v @ -np.asarray(t_lo, float) > 0.0)) | \
                 ((st == n_ax - 1) & (v @ np.asarray(t_hi, float) > 0.0))
        val = np.einsum("ij,ij->i", v, np.asarray(N, float)[jj])
        val[beyond] = np.nan
        sd[np.nonzero(ok)[0]] = val
    valid = np.isfinite(sd)
    return sd, np.where(valid, j, -1), valid


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
        self.tie_p = canal_tie_params(self.cfg, self.r_tube, self.L_iu)
        self.rod_app = rod_dir_app(c["inp"]["app"])
        self.tie_log = (self.cfg.get("canal_tie_set", "canal"), self.cfg.get("canal_engage", "radius"),
                        self.cfg.get("canal_tie_axis", "final")) != ("canal", "radius", "final")
        # eu2 (review HIGH-2): the cervix tie target carried with the tissue (cfg canal_tie_follow; None = off)
        self.tie_follow = canal_tie_follow_mode(self.cfg)
        self.tie_force_on = bool(self.tie_follow or cfg_flag(self.cfg, "log_canal_tie_force"))
        self.row_prev = None                            # the row of the previous step (the carry's start)
        self.canal_tie_last = None
        # S9: the elastic corpus (corpus_model "fem"); a rigid corpus leaves every line below inert
        self.corpus_fem = c.get("corpus_fem") is not None
        self.body_order = DEFORMABLE + (["corpus"] if self.corpus_fem else [])
        if self.corpus_fem:
            self.eng_step_k = np.full(len(c.get("corpus_canal", [])), -1, int)
            self.tie_p_k = corpus_tie_params(self.cfg, self.r_tube, self.L_iu)
            self.corpus_tgt_prev = c["X0"]["corpus"].copy()     # the pose target of the previous step (rest at k = 0)
            self.corpus_tgt = self.corpus_tgt_prev
            self.corpus_tgt_before = self.corpus_tgt_prev       # eu2: the target BEFORE this step's update
            self.corpus_tie_follow = corpus_tie_follow_on(self.cfg)
            self.corpus_tie = None
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

    def _canal_tie_force_row(self, Xc_end):
        """log row["canal_tie_force"] (cfg canal_tie_follow / log_canal_tie_force): the cervix ties' spring force at the
        END of the step -- tie_force_log of the targets and stiffnesses _begin wrote (canal_tie_last) against the
        cervix nodes Xc_end (all nodes; the tie nodes are picked here) -- and the follow mode."""
        ct = self.canal_tie_last
        return dict(tie_force_log(ct["tgt"], ct["k"], np.asarray(Xc_end, float)[self.ctx["canal"]], ct["carry"]),
                    follow=self.tie_follow or "off")

    def row_now(self):
        if self.k < len(self.sched):
            return self.sched[self.k]
        r = dict(self.sched[-1])
        r["phase"] = "H"
        return r

    def _set_rigid(self, node, R, t):
        node.rig.position.value = rigid_pose(R, t)

    # ---------------------------------------------------------------- S9: the elastic corpus (corpus_model "fem")
    def _corpus_pose(self, r, Tc):
        """This row's pose target (corpus_pose_targets: T_corpus, plus the calibrated stretch when cfg
        corpus_pose_stretch_lam != 1) into the pose springs' target MO; corpus_oar_contact "follow": the OARs' copy of
        the surface = the corpus now + this step's change of the target (follow_surface)."""
        c = self.ctx
        cp = c["corpus_pose"]
        tg = corpus_pose_targets(c["X0"]["corpus"], Tc, corpus_stretch_at(cp["lam"], r))
        cp["tgt_mo"].position.value = tg[cp["idx"]].tolist()
        fo = c.get("corpus_follow")
        if fo is not None or c.get("corpus_iface_tgt") is not None:
            Xk = self.X("corpus")
            if fo is not None:
                s2n = fo["s2n"]
                fo["mo"].position.value = follow_surface(Xk[s2n], tg[s2n], self.corpus_tgt_prev[s2n]).tolist()
            if c.get("corpus_iface_tgt") is not None:   # attach_target "predicted": the corpus now + this step's move
                c["corpus_iface_tgt"].position.value = bary_apply(follow_surface(Xk, tg, self.corpus_tgt_prev),
                                                                  *c["corpus_iface_bary"]).tolist()
        self.corpus_tgt_before = self.corpus_tgt_prev   # eu2: kept for corpus_tie_follow (this step's target change)
        self.corpus_tgt_prev = self.corpus_tgt = tg

    def _corpus_ties(self, r):
        """The corpus canal ties for this row: canal_tie_step about the row's tube axis (no below-flange part), depth
        engagement on the clipped canal_s, optionally the arclength axial drive (corpus_canal_axial).  cfg
        corpus_tie_follow (eu2): the nodes are first carried by this step's pose-target change (corpus_tgt minus
        corpus_tgt_before at the tie nodes, what follow_surface adds), and the lateral correction / axial drive start
        from there."""
        c = self.ctx
        X = self.X("corpus")[c["corpus_canal"]]
        at = geom.unit(np.asarray(r["tube_axis"], float))
        F = np.asarray(r["F"], float)
        carry = None
        if self.corpus_tie_follow:
            ii = c["corpus_canal"]
            carry = np.asarray(self.corpus_tgt, float)[ii] - np.asarray(self.corpus_tgt_before, float)[ii]
        res = canal_tie_step(X, F, at, at, self.eng_step_k, self.k, self.tie_p_k, tip_s=r.get("tip_s"),
                             canal_s=c["corpus_canal_s_eng"], carry=carry)
        tgt = res["tgt"]
        if self.cfg.get("corpus_canal_axial", "none") == "arclength" and r.get("tip_s") is not None:
            Xa = X if carry is None else X + carry
            tgt, _ = axial_tie_targets(res, Xa, F, at, r["tip_s"], c["corpus_canal_s"], self.L_iu, self.tie_p_k["cap"])
        c["corpus_canal_tgt"].position.value = tgt.tolist()
        c["corpus_canal_ff"].stiffness.value = res["k"].tolist()
        self.corpus_tie = dict(n_active=int(res["act"].sum()), n_engaged=int((self.eng_step_k >= 0).sum()),
                               canal_d=res["canal_d"], tgt=np.asarray(tgt, float), k=np.asarray(res["k"], float))
        if carry is not None:
            self.corpus_tie["carry"] = carry

    def _corpus_log(self, X):
        """log.jsonl row["corpus_fem"]: how far the corpus is from its pose target (pose springs), how non-rigid it is
        (serosa residual after the best rigid fit to rest), its total volume ratio, the attach residual (cervix
        interface nodes vs their corpus-mapped copies: > 0 means the attach lags or is soft), the ties and the net forces
        of the pose springs and the ties on the corpus (spring force k (target - x) at the END of the step, N)."""
        c = self.ctx
        X0 = c["X0"]["corpus"]
        cp = c["corpus_pose"]
        P = cp["idx"]
        dP = X[P] - self.corpus_tgt[P]
        nP = np.linalg.norm(dP, axis=1)
        ser = cp["serosa"]
        R, t = geom.kabsch(X0[ser], X[ser])
        e = np.linalg.norm(X0[ser] @ R.T + t - X[ser], axis=1)
        out = dict(pose_rms_mm=round(float(np.sqrt((nP ** 2).mean())), 4), pose_max_mm=round(float(nP.max()), 4),
                   pose_net_force_N=round(float(np.linalg.norm((-cp["k"][:, None] * dP).sum(0))) / 1000.0, 4),
                   nonrigid_serosa_rms_mm=round(float(np.sqrt((e ** 2).mean())), 4),
                   nonrigid_serosa_max_mm=round(float(e.max()), 4),
                   kabsch_rotation_deg=round(float(geom.rot_angle_deg(R)), 3))
        if c["tets"].get("corpus") is not None:
            out["vol_ratio_total"] = round(float(_tet_vol(X, c["tets"]["corpus"]).sum() / c["vol0"]["corpus"].sum()), 5)
        # attach: the cervix interface nodes against the corpus points they belong to NOW (host barycentric map of the
        # corpus nodes, SOFA's weights) -- and how stale SOFA's mapped copy /corpus/iface/mo reads (see attach_target)
        Xm = bary_apply(X, *c["corpus_iface_bary"])
        out["attach_residual_mm"] = round(float(np.linalg.norm(Xm - self.X("cervix")[c["iface"]], axis=1).max()), 5)
        try:
            Xi = np.asarray(c["corpus_iface_mo"].position.value, float)
            out["iface_mo_stale_mm"] = round(float(np.linalg.norm(Xi - Xm, axis=1).max()), 5)
        except Exception as ex:                         # a read-out: never let it stop the run
            out["iface_mo_stale_mm"] = "n/a (%s)" % type(ex).__name__
        ti = self.corpus_tie
        if ti is not None:
            Xt = X[c["corpus_canal"]]
            out.update(n_ties_active=ti["n_active"], n_ties_engaged=ti["n_engaged"], canal_d_mm=ti["canal_d"],
                       tie_net_force_N=round(float(np.linalg.norm((ti["k"][:, None] * (ti["tgt"] - Xt)).sum(0))) / 1000.0, 4))
            if ti.get("carry") is not None:             # eu2 corpus_tie_follow: the per-node force and the carry
                f = tie_force_log(ti["tgt"], ti["k"], Xt, ti["carry"])
                out.update(tie_follow=True, tie_max_node_force_N=f["max_node_N"], tie_sum_abs_force_N=f["sum_abs_N"],
                           tie_carry_max_mm=f["carry_max_mm"])
        return out

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
        if not self.corpus_fem:
            self._set_rigid(c["corpus"], Tc[:3, :3], Tc[:3, 3])
            if c.get("corpus_iface_tgt") is not None:   # attach_target "predicted": this row's pose, exactly
                c["corpus_iface_tgt"].position.value = (c["X0"]["cervix"][c["iface"]] @ Tc[:3, :3].T + Tc[:3, 3]).tolist()
        else:                                           # elastic corpus: its pose target (and the OARs' copy of it)
            self._corpus_pose(r, Tc)
        Rr = np.asarray(r["R_rows"], float)             # PER-STEP orientation (constant when rotation is "off")
        self._set_rigid(c["tandem"], Rr.T, r["F"])      # rigid_pose wants columns = applicator axes, hence .T
        if c.get("ovoids") is not None:                 # (ovoid_mode "none": no ovoid body)
            self._set_rigid(c["ovoids"], ovoid_R_rows(r).T, ovoid_origin(r, a))   # its own frame in "rods" mode
        if c.get("balloon") is not None:
            self._balloon(r)
        # (2) velocity scaling (quasi-static relaxation during the settle phase)
        sc = float(cfg["settle_vel_scale"] if r["phase"] == "H" else cfg["motion_vel_scale"])
        if sc != 1.0:
            for b in self.deformable:
                mo = c["nodes"][b].dofs
                mo.velocity.value = (np.array(mo.velocity.value, dtype=float) * sc).tolist()
        # (2b) S2 P4 / monitor: OAR surface nodes against the sheet they are loaded by, AFTER the sheet has moved
        if c.get("sheet_pen") is not None:
            self._sheet_penalty()
        # (3) canal ties: target = the node moved radially to the tube surface at its own axial level, at most
        #     canal_max_offset_mm from where it is now (bounded tie force; the full move is reached over several
        #     steps).  "dilate" moves outward only; "centre" also pulls a node beyond the tube radius back onto it.
        #     The arithmetic is canal_tie_step (pure, unit-tested); tie_axes picks the line (cfg canal_tie_axis, S1a).
        #     cfg canal_tie_follow (eu2): the nodes are first carried by this step's rigid motion (canal_tie_carry), so
        #     the cap bounds the misfit only; off (default) = the node's start-of-step position, bit for bit.
        self.n_canal_active = 0
        if c.get("canal_ff") is not None and len(c["canal"]):
            at, ar = tie_axes(cfg, self.tgt, r, self.rod_app)
            X = self.X("cervix")[c["canal"]]
            carry = canal_tie_carry(self.tie_follow, X, self.row_prev, r) if self.tie_follow else None
            res = canal_tie_step(X, np.asarray(r["F"], float), at, ar, self.eng_step, self.k, self.tie_p,
                                 tip_s=r.get("tip_s"), canal_s=c.get("canal_s"), carry=carry)
            c["canal_tgt"].position.value = res["tgt"].tolist()
            c["canal_ff"].stiffness.value = res["k"].tolist()
            self.n_canal_active = int(res["act"].sum())
            self.canal_d = res["canal_d"]
            if self.tie_force_on:                       # eu2: read back at the end of the step (_end)
                self.canal_tie_last = dict(tgt=np.asarray(res["tgt"], float), k=np.asarray(res["k"], float), carry=carry)
        self.row_prev = r                               # eu2: the next step's carry starts from this row
        # (3a) S9: the elastic corpus's own canal ties (same arithmetic, the row's tube axis, depth engagement)
        if self.corpus_fem and c.get("corpus_canal_ff") is not None:
            self._corpus_ties(r)
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
            if c.get("apex_fix") is not None:
                want = r["phase"] != "B"
                if want != c["apex_fix_on"]:
                    c["apex_fix"].indices.value = [int(i) for i in L["idx"]] if want else []
                    c["apex_fix_on"] = want
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
            self._sheet_write(b, b["X_start"] + w * (b["X_end"] - b["X_start"]))
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
            self._sheet_write(b, self.X("vagina")[b["idx"]])
            if not b.get("inner_on_done") and len(b["wall_models"]) > 3:
                b["inner_on_done"] = True
                b["wall_group_seen"] = [_set_group(m, g) for m, g in zip(b["wall_models"][3:], b["wall_on_list"][3:])]
        elif b["active"] and not b["released"]:
            b["released"] = True
            b["w"] = 1.0
            self._sheet_write(b, b["X_end"])
            b["balloon_group_seen"] = [_set_group(m, b["g_off"]) for m in b["models"]]
            b["wall_group_seen"] = [_set_group(m, g) for m, g in zip(b["wall_models"], b["wall_on_list"])]

    def _sheet_write(self, b, X):
        """Place the balloon: X = its outer-node positions; a subdivided copy (cfg balloon_subdivide) gets its edge
        midpoints rebuilt from them.  Keeps X as the sheet the OAR monitor / penalty measures against."""
        b["cur"] = np.asarray(X, float)
        b["node"].mo.position.value = apply_subdiv(b["cur"], b["rounds"]).tolist()

    def _sheet_measure(self):
        """Signed distance of every OAR surface node (cfg oar_sheet_bodies) to the sheet the OARs are loaded by: the
        balloon as last placed (phase B, and afterwards in "follow" mode), or the wall's own outer sheet once a
        "release" balloon is off.  Returns {body: (Xb, sd, j, ok)} and the sheet's outward vertex normals, or None
        while the balloon is not active."""
        c, S = self.ctx, self.ctx["sheet_pen"]
        b = c["balloon"]
        if not b["active"]:
            return None, None
        Xs = self.X("vagina")[b["idx"]] if b["released"] else b["cur"]
        N = S["sign"] * vertex_normals(Xs, b["tri_loc"])
        out = {}
        for name, e in S["bodies"].items():
            Xb = self.X(name)[e["idx"]]
            sd, j, ok = sheet_signed_distance(Xb, Xs, N, b["station"], S["n_ax"], b["t_lo"], b["t_hi"], S["reach"])
            out[name] = (Xb, sd, j, ok)
        return out, N

    def _sheet_penalty(self):
        """S2 P4 (cfg oar_sheet_penalty_mN_per_mm > 0), at the START of a step, after the sheet has moved: a node
        closer than oar_sheet_margin_mm to the sheet, or behind it, is sprung towards `margin` in front of it along
        the sheet normal, at most oar_sheet_max_offset_mm from where it is now (bounded force, as the canal tie).
        KNOWN DEFECT (review M6, measured in probe PRB4): each spring is switched fully on or off every step and its
        target re-anchored, so after phase B the pushed nodes chatter with a 2-step period.  It is the only option
        that kept every organ vertex in front of the sheet through B (P0-P4), but do not use it past B until the
        switching is smoothed (ramped stiffness or hysteresis on the margin)."""
        S = self.ctx["sheet_pen"]
        if S["k"] <= 0.0:
            return
        m, N = self._sheet_measure()
        self.n_pushed = {}
        for name, e in S["bodies"].items():
            if m is None:                                # balloon not active yet: springs off
                e["ff"].stiffness.value = [0.0] * len(e["idx"])
                continue
            Xb, sd, j, ok = m[name]
            push = ok & (np.nan_to_num(sd, nan=np.inf) < S["margin"])
            off = np.zeros(len(Xb))
            off[push] = np.minimum(S["margin"] - sd[push], S["cap"])
            tg = Xb.copy()
            tg[push] += off[push, None] * N[j[push]]
            e["tgt"].position.value = tg.tolist()
            e["ff"].stiffness.value = np.where(push, S["k"], 0.0).tolist()
            self.n_pushed[name] = int(push.sum())

    def _sheet_monitor(self):
        """The in-run crossing monitor, at the END of a step (the state a frame shows): per OAR, nodes measured,
        nodes behind the sheet (sd < 0) and more than 0.5 mm behind it (the S2 gate's threshold), and the minimum."""
        m, _ = self._sheet_measure()
        if m is None:
            return None
        log = {}
        for name, (Xb, sd, j, ok) in m.items():
            v = sd[ok]
            log[name] = dict(n_measured=int(ok.sum()), n_lt_0=int((v < 0.0).sum()), n_lt_m0p5=int((v < -0.5).sum()),
                             sd_min=(round(float(v.min()), 3) if len(v) else None))
            if name in getattr(self, "n_pushed", {}):
                log[name]["n_pushed"] = self.n_pushed[name]
        return log

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
        for b in self.body_order:                       # DEFORMABLE, + "corpus" when it is elastic (S9)
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
                if cfg.get("log_min_vol_where"):        # S6 diagnostics: where the worst element is
                    j = int(np.argmin(vr))
                    Tj = c["tets"][b][j]
                    disp[b]["min_vol_where"] = dict(tet=j, nodes=[int(i) for i in Tj],
                                                    centroid=np.round(X[Tj].mean(axis=0), 2).tolist(),
                                                    n_lt_0p6=int((vr < 0.6).sum()), n_lt_0p8=int((vr < 0.8).sum()))
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
        if r.get("w_r") is not None:                    # tandem-first rows (S5): where the plan puts the tip
            row["tf"] = dict(tip_s=round(float(r["tip_s"]), 4), d_mm=round(float(r["d_mm"]), 4),
                             w_r=round(float(r["w_r"]), 6), w_l=round(float(r["w_l"]), 6))
        if self.tie_log:                                # S1 switches in use: the engagement record (G33: ties vs u)
            ts = r.get("tip_s")
            row["canal_tie"] = dict(n_engaged=int((self.eng_step >= 0).sum()), n_nodes=int(len(self.eng_step)),
                                    tip_s=(None if ts is None else round(float(ts), 3)))
        if self.canal_tie_last is not None and finite:  # eu2 (canal_tie_follow / log_canal_tie_force): end-of-step force
            row["canal_tie_force"] = self._canal_tie_force_row(self.X_prev["cervix"])
        if self.corpus_fem and finite:                  # S9: the elastic corpus against its target, and its ties
            row["corpus_fem"] = self._corpus_log(self.X_prev["corpus"])
        elif c.get("corpus_iface_tgt") is not None and finite:     # rigid + attach_target "predicted": the residual
            Tc = np.asarray(r["T_corpus"], float)
            Xm = c["X0"]["cervix"][c["iface"]] @ Tc[:3, :3].T + Tc[:3, 3]
            row["attach"] = dict(target="predicted", residual_mm=round(float(np.linalg.norm(
                Xm - self.X_prev["cervix"][c["iface"]], axis=1).max()), 5))
        if c.get("sheet_pen") is not None and finite:   # S2 monitor: OAR surface nodes vs the sheet, post-solve
            row["oar_sheet"] = self._sheet_monitor()
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
        stop = cfg.get("stop_after_phase")
        if stop and not self.done and r["phase"] == stop:    # the probes: end after the last row of that phase
            nxt = self.sched[self.k + 1]["phase"] if self.k + 1 < len(self.sched) else "H"
            if nxt != stop:
                self.done, self.status = True, "stopped_after_%s" % stop
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
