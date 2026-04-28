"""Map each (target tracking3d) → branch by transforming the target back to
vessel-CS, then finding the closest named centerline point."""
import sys
import numpy as np
import yaml
import re
import glob
import os


def _get_rot_matrix(image_rot_zx):
    rot_z = -image_rot_zx[0] * np.pi / 180
    rot_x = -image_rot_zx[1] * np.pi / 180
    rotation_matrix_z = np.array(
        [[np.cos(rot_z), -np.sin(rot_z), 0],
         [np.sin(rot_z), np.cos(rot_z), 0],
         [0, 0, 1]],
    )
    rotation_matrix_x = np.array(
        [[1, 0, 0],
         [0, np.cos(rot_x), -np.sin(rot_x)],
         [0, np.sin(rot_x), np.cos(rot_x)]],
    )
    return rotation_matrix_z @ rotation_matrix_x


def tracking3d_to_vessel_cs(array, image_rot_zx, image_center):
    rot_matrix = _get_rot_matrix(image_rot_zx)
    image_center = np.array(image_center)
    image_center_rot_cs = rot_matrix @ image_center.T
    new_array = np.array(array) + image_center_rot_cs
    return (rot_matrix.T @ new_array.T).T


# image_rot_zx and image_center from env_train.yml
IMAGE_ROT_ZX = (20.0, 5.0)
IMAGE_CENTER = (0.0, 0.0, 0.0)


# ---- Load named branch centerlines from env_train.yml ----
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


def walk(node):
    if isinstance(node, dict):
        if "name" in node and "coordinates" in node:
            yield node["name"], node["coordinates"]
        for v in node.values():
            yield from walk(v)
    elif isinstance(node, list):
        for it in node:
            yield from walk(it)


with open(
    "saved/eve_paper/neurovascular/full/mesh_ben/2026-04-27_173014_env5_rl7_ckpttest28/env_train.yml",
    "r",
) as f:
    cfg = yaml.load(f, Loader=_AnyTagLoader)

named_centerlines = {}
for name, coords in walk(cfg):
    if name and name.startswith("Centerline curve - "):
        short = name.split(" - ")[1].split(".")[0]
        named_centerlines[short] = np.array([[float(c[0]), float(c[1]), float(c[2])] for c in coords])

print("Named centerlines (vessel-CS):")
for n, pts in named_centerlines.items():
    print(f"  {n}: {len(pts)} pts, x∈[{pts[:,0].min():.1f},{pts[:,0].max():.1f}], "
          f"y∈[{pts[:,1].min():.1f},{pts[:,1].max():.1f}], "
          f"z∈[{pts[:,2].min():.1f},{pts[:,2].max():.1f}]")


def classify_target(target_track3d):
    """Map target (in tracking3d frame) → nearest named branch."""
    target_v = tracking3d_to_vessel_cs(
        np.array(target_track3d, dtype=np.float32),
        IMAGE_ROT_ZX,
        IMAGE_CENTER,
    )
    best_name = None
    best_d = 1e18
    best_idx = None
    for name, pts in named_centerlines.items():
        d2 = np.sum((pts - target_v) ** 2, axis=1)
        i = int(np.argmin(d2))
        if d2[i] < best_d:
            best_d = d2[i]
            best_name = name
            best_idx = i
    return best_name, np.sqrt(best_d), best_idx, target_v


# ---- Apply to all EPISODE_START targets ----
RE_EP_START = re.compile(
    r"EPISODE_START \| ep=(\d+) \| .* pid=(\d+) \| target=\(([\d.\-]+),([\d.\-]+),([\d.\-]+)\)"
)

LOG_DIR = ("saved/eve_paper/neurovascular/full/mesh_ben/"
           "2026-04-27_173014_env5_rl7_ckpttest28/diagnostics/logs_subprocesses")

eps = []
for path in sorted(glob.glob(os.path.join(LOG_DIR, "worker_*.log"))):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = RE_EP_START.search(line)
            if m:
                ep, pid, x, y, z = m.groups()
                eps.append((int(pid), int(ep), float(x), float(y), float(z)))

print(f"\nClassifying {len(eps)} episode targets...")
from collections import Counter
counts = Counter()
sample = {}
for pid, ep, x, y, z in eps:
    name, dist, idx, tv = classify_target((x, y, z))
    counts[name] += 1
    sample.setdefault(name, []).append((pid, ep, (x, y, z), tuple(round(v, 1) for v in tv), dist, idx))

print("\nBranch distribution:")
for n, c in counts.most_common():
    print(f"  {n}: {c}")

print("\nSamples per branch (target_track3d → target_vessel_cs, distance to nearest, point index):")
for n in counts:
    print(f"\n  {n}:")
    for s in sample[n][:5]:
        pid, ep, t3d, tvc, d, idx = s
        print(f"    pid={pid} ep={ep} t3d=({t3d[0]:.1f},{t3d[1]:.1f},{t3d[2]:.1f}) → tvc={tvc} d={d:.2f}mm idx={idx}/{len(named_centerlines[n])}")
