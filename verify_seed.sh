#!/bin/bash
# Verify saved/rcca_proc_heatup/seed.npz before launching rcca_procedural_v3a.
# Expectations from HANDOFF_RCCA_V3A.md §2:
#   67,776,360 bytes | md5 54fe108c45ec46978d8987c00b09e2b7
#   282,310 transitions / 480 episodes / meta_buckle_coef=0.5
SEED="saved/rcca_proc_heatup/seed.npz"
EXP_BYTES=67776360
EXP_MD5=54fe108c45ec46978d8987c00b09e2b7

fail() { echo "FAIL: $1"; exit 1; }

[ -f "$SEED" ] || fail "$SEED not found. Download it and place it there."

BYTES=$(stat -c %s "$SEED")
echo "size : $BYTES (expected $EXP_BYTES)"
[ "$BYTES" = "$EXP_BYTES" ] || fail "size mismatch — download is truncated or the wrong file."

echo "md5  : computing (~68MB, a few seconds)..."
MD5=$(md5sum "$SEED" | cut -d' ' -f1)
echo "md5  : $MD5 (expected $EXP_MD5)"
[ "$MD5" = "$EXP_MD5" ] || fail "md5 mismatch — corrupt or wrong seed. Do NOT train on it."

# Inspect contents with the image's numpy (no host numpy needed).
echo "npz  : inspecting contents inside the docker image..."
docker run --rm -v "$(pwd)/saved:/results" eve-training-fixed python3 - <<'PY'
import numpy as np
d = np.load("/results/rcca_proc_heatup/seed.npz", allow_pickle=True)
print("keys:", sorted(d.files)[:12], "..." if len(d.files) > 12 else "")
for k in ("meta_buckle_coef", "buffer_len", "episodes_received"):
    if k in d.files:
        print(f"{k}: {d[k]}")
for k in sorted(d.files):
    a = d[k]
    if getattr(a, "shape", ()) and len(a.shape) >= 1 and a.shape[0] > 1000:
        print(f"{k}: shape={a.shape} dtype={a.dtype}")
        break
PY

echo
echo "SEED OK — safe to launch:  bash launch_rcca_procedural_v3a.sh"
