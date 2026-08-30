import glob,re,sys
from collections import Counter
PR=re.compile(r"proj_s=([-0-9.]+)"); NN=re.compile(r"nearest_named=(\S+)")
CB=re.compile(r"cur_branch=(.*?) \| local_r"); PL=re.compile(r"path_len=([0-9.]+)")
EP=re.compile(r"entries_passed=(\d+)"); DP=re.compile(r"daughters_passed=(\d+)")
bands=[(0,20),(20,35),(35,40),(40,50),(50,80),(140,160),(160,180),(180,220)]
d={b:Counter() for b in bands}; e={b:Counter() for b in bands}
for p in sorted(glob.glob(sys.argv[1]+"/worker_*.log"))[:4]:
    for line in open(p,errors="replace"):
        if " STEP |" not in line: continue
        m=PR.search(line)
        if not m: continue
        v=float(m.group(1))
        for b in bands:
            if b[0]<=v<b[1]:
                nn=NN.search(line); cb=CB.search(line); ep=EP.search(line); dp=DP.search(line)
                d[b][(nn.group(1) if nn else "?", cb.group(1).strip() if cb else "?")]+=1
                e[b][(ep.group(1) if ep else "?", dp.group(1) if dp else "?")]+=1
                break
for b in bands:
    print(b, d[b].most_common(3), "| entries/daughters:", e[b].most_common(2))
