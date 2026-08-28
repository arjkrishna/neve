import pickle,json,statistics as st,math
from collections import defaultdict,Counter
recs=pickle.load(open("D:/Arjun/workspace/neve/monitoring/_t2_recs.pkl","rb"))
raw=open("D:/Arjun/workspace/neve/monitoring/_t2_geom.out").read()
G=json.loads(raw.split("@@@JSON@@@")[1].strip())
TERM=8.0
def mid(runs,L): return [r for r in runs if r["s1"]<L-TERM]
print("=== A. EXACT MID-VESSEL BLOCKAGE INVENTORY (non-terminal = run ends >8mm before terminus) ===")
print(f"{'anat':<12}{'L':>7}{'min_d':>7}  | {'w0.18 mid':>28} | {'s0.30 mid':>28} | {'c0.35 mid':>28}")
for a in sorted(G):
    g=G[a]; L=g["L"]
    cells=[]
    for k in ("w018","s030","c035"):
        m=mid(g[k],L)
        cells.append(", ".join(f"{r['s0']:.1f}-{r['s1']:.1f}@{r['min_d']:.2f}" for r in m) or "-")
    print(f"{a:<12}{L:>7.1f}{min(g['d']):>7.3f}  | {cells[0]:>28} | {cells[1]:>28} | {cells[2]:>28}")

print()
print("=== B. TERMINAL (end-cap) EXTENT: how far back from L does clearance fall below each thr ===")
print(f"{'anat':<12}{'L':>7}{'cap_w018':>10}{'cap_s030':>10}{'cap_c035':>10}   (mm of terminus below thr)")
for a in sorted(G):
    g=G[a]; L=g["L"]; row=[]
    for k in ("w018","s030","c035"):
        t=[r for r in g[k] if r["s1"]>=L-TERM]
        row.append(L-min(r["s0"] for r in t) if t else 0.0)
    print(f"{a:<12}{L:>7.1f}{row[0]:>10.2f}{row[1]:>10.2f}{row[2]:>10.2f}")

# per-anatomy arrest vs geometry
print()
print("=== C. PER-ANATOMY: modal failure arrest depth vs nearest mid-vessel clearance minimum ===")
def modal(vals,bw=4.0):
    best=(None,-1)
    for c in vals:
        n=sum(1 for x in vals if c-bw/2<=x<=c+bw/2)
        if n>best[1]: best=(c,n)
    inw=[x for x in vals if best[0]-bw/2<=x<=best[0]+bw/2]
    return st.median(inw),best[1]
by=defaultdict(list)
for r in recs: by[r["mesh"]].append(r)
print(f"{'anat':<12}{'n':>4}{'nf':>4}{'mode_s':>8}{'n@mode':>7}{'sd':>7}{'rng':>15}  {'blk030':>9}{'d_to_blk':>9}  {'blk018':>9}{'blk035':>9}  {'L':>7}{'mode-L':>8}")
rowsC=[]
for a in sorted(by):
    rs=by[a]; fs=[r for r in rs if not r["succ"]]
    g=G[a]; L=g["L"]
    m030=mid(g["s030"],L); m018=mid(g["w018"],L); m035=mid(g["c035"],L)
    b030=min(r["s_at_min"] for r in m030) if m030 else None
    b018=min(r["s_at_min"] for r in m018) if m018 else None
    b035=min(r["s_at_min"] for r in m035) if m035 else None
    if not fs:
        print(f"{a:<12}{len(rs):>4}{0:>4}       -                             "
              f"  {('%.1f'%b030) if b030 else '-':>9}{'-':>9}  {('%.1f'%b018) if b018 else '-':>9}{('%.1f'%b035) if b035 else '-':>9}  {L:>7.1f}")
        continue
    d=sorted(r["max_s"] for r in fs); md,cnt=modal(d)
    sd=st.pstdev(d) if len(d)>1 else 0.0
    dblk=(md-b030) if b030 is not None else None
    print(f"{a:<12}{len(rs):>4}{len(fs):>4}{md:>8.1f}{cnt:>7}{sd:>7.1f}{('%.0f-%.0f'%(d[0],d[-1])):>15}  "
          f"{('%.1f'%b030) if b030 is not None else '-':>9}{('%+.1f'%dblk) if dblk is not None else '-':>9}  "
          f"{('%.1f'%b018) if b018 is not None else '-':>9}{('%.1f'%b035) if b035 is not None else '-':>9}  {L:>7.1f}{md-L:>8.1f}")
    rowsC.append((a,md,b030,b035,L,len(fs),sd))
pickle.dump({"G":G,"rowsC":rowsC},open("D:/Arjun/workspace/neve/monitoring/_t2_geo.pkl","wb"))
