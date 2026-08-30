import json, csv, os, collections
B="saved/eve_paper/neurovascular/full/mesh_ben/2026-07-25_022443_rcca_p2_teacher_v1bp/checkpoints/"
def load(d, ts):
    jl={}
    for line in open(B+d+"/episodes_official_%s.jsonl"%ts):
        r=json.loads(line); jl[r["seed"]]=r
    cs={}
    for r in csv.DictReader(open(B+d+"/episodes.csv")):
        cs[int(r["seed"])]=r
    return jl,cs
for d,ts,lab in [("eval_anatomies_checkpoint2002292","20260828_053306","POLICY"),
                 ("eval_anatomies_checkpoint0","20260828_062606","HEUR")]:
    jl,cs=load(d,ts)
    print(lab, "jsonl n=",len(jl), "csv n=",len(cs), "seed overlap=", len(set(jl)&set(cs)))
    mism=sum(1 for s in jl if s in cs and int(cs[s]["success"])!=int(bool(jl[s]["success"])))
    print("  success mismatch csv-vs-jsonl:", mism, " jsonl succ=",sum(1 for s in jl if jl[s]["success"]),
          " csv succ=",sum(int(r["success"]) for r in cs.values()))
    print("  mtime episodes.csv", os.path.getmtime(B+d+"/episodes.csv"))
