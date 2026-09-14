"""Probe (CONTAINER): can a SofaPython3 v22.12 Python ForceField (Vec3d) act on FEM nodes with StaticSolver + CG?
A: RSSFF isotropic tie (reference).  B: Python FF, isotropic (must equal A).  C: Python FF, lateral-only (P = I - z z^T).
Units mm, mN.  python3 /scratch/probe_pyff.py"""
import json, time, traceback
import numpy as np
import Sofa, Sofa.Core, Sofa.Simulation

K_TIE = 50.0
SHIFT = np.array([1.0, 0.5, 2.0])
out = {}


class PyTie(Sofa.Core.ForceFieldVec3d):
    def __init__(self, *args, **kwargs):
        Sofa.Core.ForceFieldVec3d.__init__(self, *args, **kwargs)
        self.nodes = np.zeros(0, int); self.F0 = np.zeros((0, 3)); self.Kb = np.zeros((0, 3, 3)); self.xref = np.zeros((0, 3))
        self.ncall = dict(f=0, df=0, k=0, e=0); self.kf_seen = []

    def addForce(self, m, f, x, v):
        self.ncall["f"] += 1
        X = np.asarray(x.value)[self.nodes]
        F = self.F0 - np.einsum("nij,nj->ni", self.Kb, X - self.xref)
        with f.writeableArray() as wa:
            wa[self.nodes] += F

    def addDForce(self, m, df, dx):
        self.ncall["df"] += 1
        kf = m["kFactor"] if isinstance(m, dict) else float(m.kFactor())
        if len(self.kf_seen) < 3:
            self.kf_seen.append(float(kf))
        D = np.asarray(dx.value)[self.nodes]
        with df.writeableArray() as wa:
            wa[self.nodes] -= kf * np.einsum("nij,nj->ni", self.Kb, D)

    def addKToMatrix(self, m, nNodes, nDofs):
        self.ncall["k"] += 1
        return np.zeros((nDofs, nDofs))

    def getPotentialEnergy(self, m, x):
        self.ncall["e"] += 1
        return 0.0


def build(mode):
    root = Sofa.Core.Node("root"); root.gravity = [0, 0, 0]; root.dt = 1.0
    root.addObject("RequiredPlugin", pluginName="Sofa.Component"); root.addObject("DefaultAnimationLoop")
    tg = root.addChild("tg")
    b = root.addChild("b")
    b.addObject("StaticSolver", newton_iterations=3, absolute_correction_tolerance_threshold=1e-6,
                relative_correction_tolerance_threshold=1e-8, absolute_residual_tolerance_threshold=1e-6,
                relative_residual_tolerance_threshold=1e-8)
    b.addObject("CGLinearSolver", iterations=500, tolerance=1e-14, threshold=1e-18)
    b.addObject("RegularGridTopology", name="g", n=[5, 5, 9], min=[0, 0, 0], max=[16, 16, 32])
    mo = b.addObject("MechanicalObject", name="dofs", template="Vec3d")
    b.addObject("HexahedronFEMForceField", youngModulus=30.0, poissonRatio=0.45, method="large")
    n = 5 * 5 * 9
    b.addObject("RestShapeSpringsForceField", name="found", points=list(range(n)), stiffness=[0.2] * n)
    ff = None; tmo = None
    S = list(range(5 * 5 * 4, 5 * 5 * 5))   # a mid slab of nodes
    if mode == "A":
        tmo = tg.addObject("MechanicalObject", name="t", template="Vec3d", position=[[0, 0, 0]] * len(S))
        b.addObject("RestShapeSpringsForceField", name="tie", points=S, external_rest_shape="@/tg/t",
                    external_points=list(range(len(S))), stiffness=[K_TIE] * len(S))
    else:
        ff = b.addObject(PyTie(name="pytie"))
    Sofa.Simulation.init(root)
    X0 = np.array(mo.position.value, copy=True)
    return root, mo, ff, tmo, np.array(S), X0


for mode in ("A", "B", "C"):
    try:
        root, mo, ff, tmo, S, X0 = build(mode)
        T = X0[S] + SHIFT
        walls = []
        for it in range(6):
            X = np.array(mo.position.value, copy=True)
            if mode == "A":
                tmo.position.value = T.tolist()
            else:
                P = np.eye(3) if mode == "B" else np.eye(3) - np.outer([0, 0, 1.0], [0, 0, 1.0])
                ff.nodes = S; ff.xref = X[S]; ff.Kb = np.repeat((K_TIE * P)[None], len(S), 0)
                ff.F0 = (K_TIE * (T - X[S])) @ P.T
            t = time.perf_counter(); Sofa.Simulation.animate(root, 1.0); walls.append(1000 * (time.perf_counter() - t))
        X = np.array(mo.position.value)
        out[mode] = dict(ok=True, mean_disp_tie=(X[S] - X0[S]).mean(0).round(5).tolist(), umax=float(np.linalg.norm(X - X0, axis=1).max()),
                         ms=np.round(walls, 1).tolist(), ncall=(ff.ncall if ff else None), kf=(ff.kf_seen if ff else None))
        out[mode + "_X"] = X
    except Exception:
        out[mode] = dict(ok=False, err=traceback.format_exc()[-1500:])
if "A_X" in out and "B_X" in out:
    out["maxdiff_A_B_mm"] = float(np.abs(out["A_X"] - out["B_X"]).max())
for k in [k for k in out if k.endswith("_X")]:
    del out[k]
print("PROBE", json.dumps(out, indent=1))
