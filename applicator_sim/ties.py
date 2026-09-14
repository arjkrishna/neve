"""Anisotropic distributed node ties as a SofaPython3 ForceField (CONTAINER, SOFA v22.12, py3.8).  Units mm, mN.

Why: RestShapeSpringsForceField is isotropic, so inside one implicit solve a lateral (frictionless) tie still resists
motion ALONG the rod with its full stiffness; the per-step re-projection then releases the slide only a little at a
time (measured: the hold phase creeps with the max-dx node moving axially, axial fraction ~1.0).  This force field
applies, on the FEM nodes themselves (F1: no forces on mapped nodes), the linearisation at the start of the step of
the lumped DNT springs with a per-canal-point projector P_i:
    f_j(x) = F0_j - Kb_j (x_j - xref_j),   F0_j = sum_i w_ij k_i P_i (t_i - c_i),   Kb_j = sum_i w_ij k_i P_i
lateral ties: P_i = I - a a^T (no axial stiffness -> free sliding inside the solve); tip push: P_i = I.
It is matrix-free (addForce + addDForce), so it needs CGLinearSolver: the v22.12 binding offers addKToMatrix only as
a dense matrix, which a direct solver would need.  Probe (proto/build/probe_pyff.py): with P = I it equals
RestShapeSpringsForceField to 1.7e-14 mm; G0.6 repeats this on the real body.
The interpreter segfaults at exit once such an object has existed (SofaPython3 teardown); scripts end with os._exit.
"""
import numpy as np

import Sofa.Core


class NodeTieFF(Sofa.Core.ForceFieldVec3d):
    def __init__(self, *args, **kwargs):
        Sofa.Core.ForceFieldVec3d.__init__(self, *args, **kwargs)
        self.nodes = np.zeros(0, int)
        self.xref = np.zeros((0, 3))
        self.F0 = np.zeros((0, 3))
        self.Kb = np.zeros((0, 3, 3))
        self.n_addforce = 0
        self.n_adddforce = 0

    def set_linearisation(self, nodes, xref, F0, Kb):
        self.nodes = np.asarray(nodes, int)
        self.xref = np.asarray(xref, float)
        self.F0 = np.asarray(F0, float)
        self.Kb = np.asarray(Kb, float)

    def addForce(self, m, f, x, v):
        self.n_addforce += 1
        if not len(self.nodes):
            return
        X = np.asarray(x.value)[self.nodes]
        F = self.F0 - np.einsum("nij,nj->ni", self.Kb, X - self.xref)
        with f.writeableArray() as wa:
            wa[self.nodes] += F

    def addDForce(self, m, df, dx):
        self.n_adddforce += 1
        if not len(self.nodes):
            return
        kf = m["kFactor"] if isinstance(m, dict) else float(m.kFactor())
        D = np.asarray(dx.value)[self.nodes]
        with df.writeableArray() as wa:
            wa[self.nodes] -= kf * np.einsum("nij,nj->ni", self.Kb, D)   # K = df/dx = -Kb

    def addKToMatrix(self, m, nNodes, nDofs):
        raise RuntimeError("NodeTieFF is matrix-free; use CGLinearSolver (tie_impl='rssff' for direct solvers)")

    def getPotentialEnergy(self, m, x):
        return 0.0
