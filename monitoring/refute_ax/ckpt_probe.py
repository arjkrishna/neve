import torch, sys, numpy as np
p="/d/Arjun/workspace/neve/saved/eve_paper/neurovascular/full/mesh_ben/2026-07-25_022443_rcca_p2_teacher_v1bp/checkpoints/checkpoint2002292.everl"
d=torch.load(p,map_location="cpu",weights_only=False)
print(type(d), list(d.keys())[:20] if hasattr(d,'keys') else "")
