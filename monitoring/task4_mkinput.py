import pickle, json, os, collections
SP=r"C:/Users/akrish41/AppData/Local/Temp/claude/d--Arjun-workspace-neve/81b186b6-3a3f-4f63-8491-2172316ef81f/scratchpad"
D=pickle.load(open(os.path.join(SP,"h0.pkl"),"rb")); eps,steps,snap=D["eps"],D["steps"],D["snap"]
def t3(s): return [float(x) for x in s.strip("()").split(",")]
h0=[]
for k in sorted(eps):
    st=eps[k]["start"]; ss=steps[k]
    succ = 1 if snap[k]["cls"]=="success" else 0
    rec=dict(pid=k[0],ep=k[1],seed=int(st["seed"]),mesh=st["mesh_fp"],succ=succ,
             tgt=t3(st["target"]), path_len=float(ss[0]["path_len"]),
             fb=eps[k]["outcome"]["final_branch"] if eps[k]["outcome"] else None,
             n=len(ss))
    if not succ:
        rec["tip"]=[t3(s["tip3d"]) for s in ss]
        rec["projs"]=[float(s["proj_s"]) for s in ss]
        rec["local_r"]=[float(s["local_r"]) for s in ss]
        rec["curbr"]=[s["cur_branch"] for s in ss]
        rec["nearest"]=[s["nearest_named"] for s in ss]
        rec["dtgt"]=[float(s["d_tgt"]) for s in ss]
        rec["ins"]=[[float(x) for x in s["inserted"].strip("[]").split(",")] for s in ss]
        rec["onpath"]=[int(s["on_path"]) for s in ss]
        rec["xt"]=[float(s["xt_true"]) for s in ss]
    h0.append(rec)
teach=json.load(open(os.path.join(SP,"teach_20260828_045651.json")))
json.dump(dict(h0=h0,teach=teach), open(r"D:/Arjun/workspace/neve/monitoring/t4ak_input.json","w"))
print("h0",len(h0),"succ",sum(r["succ"] for r in h0),"teach",len(teach),"succ",sum(r["succ"] for r in teach))
print("size MB", os.path.getsize(r"D:/Arjun/workspace/neve/monitoring/t4ak_input.json")/1e6)
