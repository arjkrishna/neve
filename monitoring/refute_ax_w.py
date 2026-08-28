import torch, numpy as np
CK="/opt/eve_training/results/eve_paper/neurovascular/full/mesh_ben/2026-07-25_022443_rcca_p2_teacher_v1bp/checkpoints/checkpoint2002292.everl"
d=torch.load(CK,map_location="cpu")
W=d["network_state_dicts"]["policy"]["body._input_layer.weight"].numpy()  # (256,125)
print("policy input layer", W.shape)
cn=np.linalg.norm(W,axis=0)
order=np.argsort(-cn)
rank={int(j):int(np.where(order==j)[0][0])+1 for j in range(len(cn))}
BASE=46
names={BASE+0:"g0 d_rem",BASE+7:"g7 on_correct_path",BASE+22:"g22 ep_step_norm",
       BASE+47:"g47 radius_now",BASE+48:"g48 radius_ahead",BASE+49:"g49 clearance_norm",
       101+22:"priv22 offcorridor"}
print(f"\ncolumn L2 norms: mean {cn.mean():.4f} med {np.median(cn):.4f} p05 {np.percentile(cn,5):.4f} p95 {np.percentile(cn,95):.4f} max {cn.max():.4f}")
print("\nidx  name                     ||W[:,j]||   rank/125")
for j in sorted(names):
    print(f"{j:4d} {names[j]:24s} {cn[j]:9.4f}   {rank[j]:3d}")
# measured host->cohort median feature deltas
delta={BASE+47:0.0250, BASE+48:0.0151, BASE+49:-0.0006, BASE+7:0.155}
print("\ninduced first-layer preactivation perturbation ||W[:,j]*dx||:")
tot=np.zeros(W.shape[0])
for j,dx in delta.items():
    v=W[:,j]*dx; tot+=v
    print(f"  {names[j]:24s} dx={dx:+.4f}  ||dz||={np.linalg.norm(v):.4f}")
print(f"  calibre trio (47,48,49) combined ||dz|| = {np.linalg.norm(sum(W[:,j]*delta[j] for j in [BASE+47,BASE+48,BASE+49])):.4f}")
print(f"  on_correct_path alone      ||dz|| = {np.linalg.norm(W[:,BASE+7]*delta[BASE+7]):.4f}")
# reference: perturbation from a 0.1 change in each single feature
ref=np.array([np.linalg.norm(W[:,j]*0.1) for j in range(W.shape[1])])
print(f"\nreference: ||W[:,j]*0.1|| over all 125 inputs: med {np.median(ref):.4f} p95 {np.percentile(ref,95):.4f}")
