import os, json, sys, numpy as np
out={}
# ---------- 1. checkpoint first-layer sensitivity ----------
import torch
CK="/opt/eve_training/results/eve_paper/neurovascular/full/mesh_ben/2026-07-25_022443_rcca_p2_teacher_v1bp/checkpoints/checkpoint2002292.everl"
d=torch.load(CK,map_location="cpu")
def walk(o,pref="",depth=0):
    if depth>3: return
    if isinstance(o,dict):
        for k,v in o.items():
            if isinstance(v,torch.Tensor): print("T",pref+"/"+str(k),tuple(v.shape))
            else: walk(v,pref+"/"+str(k),depth+1)
walk(d)
