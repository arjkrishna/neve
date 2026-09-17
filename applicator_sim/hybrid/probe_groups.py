"""Probe (container): can a collision model's `group` be rewritten at runtime, and does the broad phase honour it?

Two overlapping triangles: A kinematic (moving, not simulated), B a one-element deformable with a constraint
correction.  Contacts are counted from the constraint solver.  Expected: contacts > 0 with disjoint groups,
0 once A is given B's group, > 0 again when restored.  Prints one JSON line."""
import json
import Sofa
import Sofa.Simulation

root = Sofa.Core.Node("root")
root.dt = 0.01
root.gravity = [0.0, 0.0, 0.0]
root.addObject("RequiredPlugin", name="p", pluginName="Sofa.Component")
root.addObject("FreeMotionAnimationLoop", name="fal")
root.addObject("GenericConstraintSolver", name="gcs", maxIterations=200, tolerance=1e-4, computeConstraintForces=True)
root.addObject("CollisionPipeline", name="pipe")
root.addObject("BruteForceBroadPhase", name="bf")
root.addObject("BVHNarrowPhase", name="bvh")
root.addObject("LocalMinDistance", name="lmd", alarmDistance=2.0, contactDistance=0.5, angleCone=0.0)
root.addObject("CollisionResponse", name="resp", response="FrictionContactConstraint", responseParams="mu=0.0")

a = root.addChild("A")
a.addObject("MechanicalObject", name="mo", template="Vec3d", position=[[-10, -10, 0], [10, -10, 0], [0, 10, 0]])
a.addObject("MeshTopology", name="mt", position=[[-10, -10, 0], [10, -10, 0], [0, 10, 0]], triangles=[[0, 1, 2]])
a.addObject("TriangleCollisionModel", name="tri", group=[1], moving=True, simulated=False)
a.addObject("LineCollisionModel", name="lin", group=[1], moving=True, simulated=False)
a.addObject("PointCollisionModel", name="pnt", group=[1], moving=True, simulated=False)

b = root.addChild("B")
b.addObject("EulerImplicitSolver", name="ode", rayleighStiffness=0.1, rayleighMass=0.1)
b.addObject("CGLinearSolver", name="ls", iterations=50, tolerance=1e-6, threshold=1e-6)
pos = [[-10, -10, 0.3], [10, -10, 0.3], [0, 10, 0.3]]
b.addObject("MechanicalObject", name="mo", template="Vec3d", position=pos)
b.addObject("MeshTopology", name="mt", position=pos, triangles=[[0, 1, 2]])
b.addObject("UniformMass", name="mass", totalMass=1.0)
b.addObject("TriangleCollisionModel", name="tri", group=[2])
b.addObject("LineCollisionModel", name="lin", group=[2])
b.addObject("PointCollisionModel", name="pnt", group=[2])
b.addObject("UncoupledConstraintCorrection", name="cc")

Sofa.Simulation.init(root)


def step():
    Sofa.Simulation.animate(root, root.dt.value)
    return int(root.gcs.currentNumConstraints.value)


def set_group(m, groups):
    how = "list"
    try:
        m.group.value = [int(g) for g in groups]
    except Exception as e:  # noqa: BLE001
        how = "string (%s)" % type(e).__name__
        m.group.value = " ".join(str(int(g)) for g in groups)
    return how, str(m.group.value)


def show(m):
    return dict(value=str(m.group.value), string=m.findData("group").getValueString())


sem = dict(initial=show(a.tri))
a.tri.group.value = [2]
sem["after_list_2"] = show(a.tri)
a.tri.group.value = [3, 4]
sem["after_list_3_4"] = show(a.tri)
a.tri.group.value = "5 6"
sem["after_string_5_6"] = show(a.tri)
a.tri.group.value = []
sem["after_empty_list"] = show(a.tri)
a.tri.group.value = ""
sem["after_empty_string"] = show(a.tri)
a.tri.group.value = "1"
sem["after_string_1"] = show(a.tri)
print("[probe_groups] semantics " + json.dumps(sem), flush=True)
out = dict(n_disjoint=step())
out["n_disjoint_2"] = step()
hows = [set_group(m, [2]) for m in (a.tri, a.lin, a.pnt)]
out["set_how"] = hows[0][0]
out["read_back_shared"] = hows[0][1]
out["n_shared"] = step()
out["n_shared_2"] = step()
hows = [set_group(m, [1]) for m in (a.tri, a.lin, a.pnt)]
out["read_back_restored"] = hows[0][1]
out["n_restored"] = step()
out["n_restored_2"] = step()
out["verdict"] = ("runtime group switch HONOURED" if out["n_disjoint_2"] > 0 and out["n_shared_2"] == 0 and out["n_restored_2"] > 0
                  else "NOT honoured / inconclusive")
print("[probe_groups] " + json.dumps(out), flush=True)
