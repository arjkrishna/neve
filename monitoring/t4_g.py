import pickle, numpy as np, collections
D=pickle.load(open("_t4_rows.pkl","rb")); T=D["T"]; H=D["H"]
epsT={e["seed"]:e for e in pickle.load(open("_t4_teacher.pkl","rb"))}
epsH={e["seed"]:e for e in pickle.load(open("_t4_heur.pkl","rb"))}
cs=[]
for sd,e in epsT.items():
    C=np.array(e["cmd"]); d1=np.array(e["dins1"]); d0=np.array(e["dins0"])
    if len(C)<5: continue
    cs.append((np.corrcoef(C[:,2],d1)[0,1], np.corrcoef(C[:,0],d0)[0,1],
               np.nanmax(np.abs(d1-0.132*C[:,2]))))
cs=np.array(cs)
print("teacher: corr(cmd2,d1) min %.6f ; corr(cmd0,d0) min %.4f ; max |d1-0.132*cmd2| %.4f"%(
    np.nanmin(cs[:,0]), np.nanmin(cs[:,1]), np.nanmax(cs[:,2])))
cs=[]
for sd,e in epsH.items():
    C=np.array(e["cmd"]); d1=np.array(e["dins1"]); d0=np.array(e["dins0"])
    if len(C)<5: continue
    cs.append((np.corrcoef(C[:,2],d1)[0,1], np.nanmax(np.abs(d1-0.132*C[:,2]))))
cs=np.array(cs)
print("heuristic: corr(cmd2,d1) min %.6f ; max resid %.4f"%(np.nanmin(cs[:,0]),np.nanmax(cs[:,1])))
