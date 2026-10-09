# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v6\wall_d17\genus_cmp.py (= /sp/v6/wall_d17/genus_cmp.py in the build containers)
# genus of the D17 junction surfaces: v5-rule run vs plateau run vs local (one closed component each) -> /v6w/d17/genus_cmp.json
import glob, json, os
import pyvista as pv
def topo(f):
    m = pv.read(f).clean()
    m = m.extract_surface().triangulate()
    conn = m.connectivity()
    nc = int(conn["RegionId"].max()) + 1 if m.n_points else 0
    e = m.extract_all_edges()
    V, E, F = m.n_points, e.n_cells, m.n_cells
    chi = V - E + F
    return {"components": nc, "chi": chi, "genus_if_one": (2 - chi) / 2 if nc == 1 else None}
out = {}
for u in sorted(os.listdir("/v6w/d17/local")):
    for s in "rl":
        row = {}
        for tag in ("v5rule", "plateau", "local"):
            f = "/v6w/d17/%s/%s/%sjunction_stageA.vtp" % (tag, u, s)
            row[tag] = topo(f) if os.path.isfile(f) else None
        out["%s:%s" % (u, s.upper())] = row
tot = {t: sum((r[t] or {}).get("genus_if_one") or 0 for r in out.values()) for t in ("v5rule", "plateau", "local")}
more = [k for k, r in out.items() if r["local"] and r["v5rule"] and (r["local"]["genus_if_one"] or 0) > (r["v5rule"]["genus_if_one"] or 0)]
less = [k for k, r in out.items() if r["local"] and r["v5rule"] and (r["local"]["genus_if_one"] or 0) < (r["v5rule"]["genus_if_one"] or 0)]
res = {"genus_sum": tot, "local_more_than_v5rule": more, "local_less_than_v5rule": less, "sides": out}
json.dump(res, open("/v6w/d17/genus_cmp.json", "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "sides"}, indent=1))
for k, r in out.items():
    print(k, *[(r[t] or {}).get("genus_if_one") for t in ("v5rule", "plateau", "local")])
