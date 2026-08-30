import json, csv, collections
B="saved/eve_paper/neurovascular/full/mesh_ben/2026-07-25_022443_rcca_p2_teacher_v1bp/checkpoints/"
def load(d, ts):
    ep={}
    for line in open(B+d+"/episodes_official_%s.jsonl"%ts):
        r=json.loads(line); ep[r["seed"]]={"succ":int(bool(r["success"])),"steps":r["steps"]}
    for r in csv.DictReader(open(B+d+"/episodes.csv")):
        s=int(r["seed"])
        ep[s].update(anat=r["anatomy"], plen=float(r["path_len_mm"]), sec=r["section"])
    return ep
P=load("eval_anatomies_checkpoint2002292","20260828_053306")
H=load("eval_anatomies_checkpoint0","20260828_062606")
# matched?
mm=[s for s in P if P[s]["anat"]!=H[s]["anat"] or abs(P[s]["plen"]-H[s]["plen"])>1e-6]
print("seed-matched anatomy+target mismatches:", len(mm), "of", len(P))
GT=166.91
g=[s for s in P if P[s]["plen"]>GT]
print("grafted episodes:", len(g), " policy succ:", sum(P[s]["succ"] for s in g),
      " heur succ:", sum(H[s]["succ"] for s in g))
by=collections.defaultdict(lambda:[0,0,0])
for s in g:
    a=P[s]["anat"]; by[a][0]+=1; by[a][1]+=P[s]["succ"]; by[a][2]+=H[s]["succ"]
for a in sorted(by):
    n,p,h=by[a]; print("%-14s n=%2d pol=%d (%5.1f%%) heur=%d (%5.1f%%)"%(a,n,p,100*p/n,h,100*h/n))
json.dump({a:by[a] for a in by}, open("monitoring/h2diff_grafted.json","w"), indent=1)
json.dump({str(s):{"P":P[s],"H":H[s]} for s in P}, open("monitoring/h2diff_alleps.json","w"))
