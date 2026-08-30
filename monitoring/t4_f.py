import pickle, numpy as np, collections
D=pickle.load(open("_t4_rows.pkl","rb")); T=D["T"]; H=D["H"]
mh={r["seed"]:r for r in H}
epsT={e["seed"]:e for e in pickle.load(open("_t4_teacher.pkl","rb"))}
epsH={e["seed"]:e for e in pickle.load(open("_t4_heur.pkl","rb"))}
gt=[r for r in T if r["grafted"]]
print("cmd_action dims sample (teacher succ):")
e=epsT[gt[0]["seed"]]; C=np.array(e["cmd"]); print(C[:3], "shape",C.shape)
print("cmd col ranges over one ep:", C.min(0).round(2), C.max(0).round(2))
# correlate cmd cols with delta_ins
d0=np.array(e["dins0"]); d1=np.array(e["dins1"])
for j in range(4):
    print(" col",j,"corr d0 %.3f corr d1 %.3f"%(np.corrcoef(C[:,j],d0)[0,1], np.corrcoef(C[:,j],d1)[0,1]))
