#!/usr/bin/env python3
"""The three fields of the v3 union, as a 1-D profile across one vessel.

Cuts a line through a centerline point and plots what each field says at every
distance along it, so f_tube / f_real / f_capsule can be told apart by what
they do rather than by their definitions.

    python reports/make/fig_union_fields.py   -> reports/figs/union_fields.png
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figs")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False})

MISR, FLOOR = 1.5, 1.0
R_TUBE = max(MISR, FLOOR)
R_CAP = 1.8 * MISR + 1.0                 # 3.7 mm
LEFT_WALL, RIGHT_WALL = -1.15, 2.45      # the real lumen: pinched left, bulging right
SIDE_LO, SIDE_HI = 4.3, 5.6              # a side vessel in the same label, not part of this anatomy

x = np.linspace(-6.5, 7.0, 1400)
f_tube = R_TUBE - np.abs(x)
f_cap = R_CAP - np.abs(x)
# the real surface: inside the main lumen, and inside a detached side vessel
f_main = np.minimum(x - LEFT_WALL, RIGHT_WALL - x)
f_side = np.minimum(x - SIDE_LO, SIDE_HI - x)
f_real = np.maximum(f_main, f_side)
f_clipped = np.minimum(f_real, f_cap)
f_union = np.maximum(f_tube, f_clipped)

fig, (ax, bx) = plt.subplots(2, 1, figsize=(10.5, 7.0), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1.15]})

ax.axhline(0, color="#333333", lw=1.1)
ax.plot(x, f_tube, color="#1f77b4", lw=2.0, label="f_tube      distance to the tube of radius max(MISR, floor) = 1.5 mm")
ax.plot(x, f_real, color="#ff7f0e", lw=2.0, label="f_real      distance to the patient's segmented surface")
ax.plot(x, f_cap, color="#7f7f7f", lw=1.6, ls="--", label="f_capsule   distance to a generous tube, 1.8 x MISR + 1 mm = 3.7 mm")
ax.plot(x, f_union, color="#2ca02c", lw=3.2, alpha=0.85, label="result      max( f_tube , min( f_real , f_capsule ) )")

for v, c in ((LEFT_WALL, "#ff7f0e"), (RIGHT_WALL, "#ff7f0e"), (-R_TUBE, "#1f77b4"), (R_TUBE, "#1f77b4")):
    ax.axvline(v, color=c, lw=0.7, ls=":", alpha=0.6)

# where the surface ends up: zero crossings of the union
zc = x[:-1][np.sign(f_union[:-1]) != np.sign(f_union[1:])]
ax.plot(zc, np.zeros_like(zc), "o", color="#2ca02c", ms=9, zorder=5)
for z in zc:
    ax.annotate("%.2f" % z, (z, 0), textcoords="offset points", xytext=(0, -16),
                ha="center", fontsize=8.5, color="#2ca02c", weight="bold")

ax.annotate("real lumen PINCHES to 1.15 mm here;\nthe tube is wider, so THE TUBE WINS\n(the 1.0 mm floor doing its job)",
            xy=(-1.5, 0.05), xytext=(2.6, 4.6), fontsize=8.6, color="#1f77b4",
            arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.0))
ax.annotate("real lumen BULGES to 2.45 mm;\nthe real surface is wider, so IT WINS",
            xy=(2.45, 0.12), xytext=(3.1, 3.1), fontsize=8.6, color="#e07b39",
            arrowprops=dict(arrowstyle="->", color="#e07b39", lw=1.0))
ax.annotate("a side vessel in the SAME label, outside\nthe capsule -> CLIPPED, never reaches zero",
            xy=(4.95, 0.45), xytext=(2.0, -3.0), fontsize=8.6, color="#666666",
            arrowprops=dict(arrowstyle="->", color="#666666", lw=1.0))

ax.set_ylim(-4.2, 6.0)
ax.set_ylabel("field value (mm)\npositive = inside")
ax.legend(fontsize=8.2, loc="upper left", framealpha=0.95, borderpad=0.6)
ax.set_title("The three fields of the v3 union, along one line across a vessel.  The surface is drawn where the green curve crosses zero.",
             fontsize=10)

# which term governs, as a band
bx.set_ylim(0, 1); bx.set_yticks([])
gov = np.where(f_tube >= f_clipped, 0, np.where(f_real <= f_cap, 1, 2))
cols = {0: "#1f77b4", 1: "#ff7f0e", 2: "#7f7f7f"}
names = {0: "f_tube governs", 1: "f_real governs", 2: "f_capsule clips"}
i = 0
seen = set()
while i < len(x) - 1:
    j = i
    while j + 1 < len(x) and gov[j + 1] == gov[i]:
        j += 1
    lab = names[gov[i]] if gov[i] not in seen else None
    seen.add(gov[i])
    bx.axvspan(x[i], x[j], color=cols[gov[i]], alpha=0.35, label=lab)
    i = j + 1
bx.set_xlabel("distance along the line, from the centerline point (mm)")
bx.set_ylabel("which\nterm wins", fontsize=8.5)
bx.legend(fontsize=8.2, ncol=3, loc="upper center")

fig.tight_layout()
p = os.path.join(OUT, "union_fields.png")
fig.savefig(p, dpi=170, bbox_inches="tight", facecolor="white")
print("wrote", p, "| surface at", np.round(zc, 2))
