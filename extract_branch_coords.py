"""Extract LCCA/LVA/RCCA/RVA branch centerline coordinates (vessel-CS) from env_train.yml."""
import yaml
import sys

class _AnyTagLoader(yaml.SafeLoader):
    pass

def _ignore_any_tag(loader, tag_suffix, node):
    if isinstance(node, yaml.ScalarNode):
        return loader.construct_scalar(node)
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node)
    return loader.construct_mapping(node)

yaml.add_multi_constructor("tag:yaml.org,2002:python/", _ignore_any_tag, Loader=_AnyTagLoader)
yaml.add_multi_constructor("!", _ignore_any_tag, Loader=_AnyTagLoader)

with open(
    "saved/eve_paper/neurovascular/full/mesh_ben/2026-04-27_173014_env5_rl7_ckpttest28/env_train.yml",
    "r",
) as f:
    cfg = yaml.load(f, Loader=_AnyTagLoader)


def walk(node, path=""):
    if isinstance(node, dict):
        if "name" in node and "coordinates" in node:
            yield node["name"], node["coordinates"]
        for k, v in node.items():
            yield from walk(v, f"{path}.{k}")
    elif isinstance(node, list):
        for i, item in enumerate(node):
            yield from walk(item, f"{path}[{i}]")


named = {"Centerline curve - LCCA.mrk", "Centerline curve - LVA.mrk",
         "Centerline curve - RCCA.mrk", "Centerline curve - RVA.mrk"}

found = {}
for name, coords in walk(cfg):
    if name in named:
        found[name] = coords

for name, coords in found.items():
    short = name.split(" - ")[1].split(".")[0]
    pts = [(c[0], c[1], c[2]) for c in coords]
    print(f"{short}: {len(pts)} pts")
    print(f"  first 3: {[tuple(round(x,1) for x in p) for p in pts[:3]]}")
    print(f"  last 3: {[tuple(round(x,1) for x in p) for p in pts[-3:]]}")
    # Compute z bounds
    zs = [p[2] for p in pts]
    print(f"  z range: [{min(zs):.1f}, {max(zs):.1f}]")
