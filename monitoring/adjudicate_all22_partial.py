import re,glob,collections,os
D=r"d:\Arjun\workspace\neve\saved\eve_paper\neurovascular\full\mesh_ben\2026-07-25_022443_rcca_p2_teacher_v1bp\checkpoints\eval_anatomies_checkpoint2002292\logs\20260828_053306"
starts={}   # (pid,ep) -> dict
plen={}     # (pid,ep) -> path_len
out={}      # (pid,ep) -> (reason, grader)
for f in glob.glob(os.path.join(D,"*.log")):
    for line in open(f,errors="ignore"):
        if "EPISODE_START" in line:
            d=dict(re.findall(r"(\w+)=([^\s|]+)",line))
            starts[(d['pid'],d['ep'])]=d
        elif "EPISODE_OUTCOME" in line:
            d=dict(re.findall(r"(\w+)=([^\s|]+)",line))
            out[(d['pid'],d['ep'])]=(d['reason'],d.get('grader_success'),int(d['steps']))
        elif line.startswith("2026") and "| STEP |" in line:
            m=re.search(r"pid=(\d+).*?\bep=(\d+)",line)
            mm=re.search(r"\bep=(\d+).*?pid=(\d+).*?path_len=([\d.]+)",line)
            if mm:
                k=(mm.group(2),mm.group(1))
                plen[k]=float(mm.group(3))
print("starts",len(starts),"outcomes",len(out),"plen",len(plen))
def sec(p): return 'CCA' if p<146 else ('ICA-mid' if p<210 else 'siphon')
tot=collections.Counter(); suc=collections.Counter()
anat=collections.defaultdict(lambda:[0,0])
rows=[]
for k,(reason,g,steps) in out.items():
    s = 1 if reason=='success' else 0
    st=starts.get(k)
    a=st['mesh_fp'] if st else '?'
    p=plen.get(k)
    anat[a][0]+=1; anat[a][1]+=s
    if p is not None:
        tot[sec(p)]+=1; suc[sec(p)]+=s
    rows.append((a,p,s,steps,reason))
n=len(out); ns=sum(1 for r in rows if r[2])
print("overall %d/%d = %.1f%%"%(ns,n,100*ns/n))
for k in ['CCA','ICA-mid','siphon']:
    print(k, "%d/%d"%(suc[k],tot[k]), "%.1f%%"%(100*suc[k]/tot[k]) if tot[k] else "")
print("n anatomies", len(anat))
for a in sorted(anat, key=lambda x:(anat[x][1]/max(1,anat[x][0]))):
    c=anat[a]; print("  %-14s %d/%d  %.0f%%"%(a,c[1],c[0],100*c[1]/max(1,c[0])))
# cap sensitivity
for cap in (400,500,600):
    c=sum(1 for r in rows if r[2] and r[3]<=cap)
    print("cap",cap,"%d/%d = %.1f%%"%(c,n,100*c/n))
# seam split, offset 33.31 -> seam path_len 166.9
pre=[r for r in rows if r[1] is not None and r[1]<166.9]
post=[r for r in rows if r[1] is not None and r[1]>=166.9]
print("pre-seam %d/%d"%(sum(r[2] for r in pre),len(pre)), "post-seam %d/%d"%(sum(r[2] for r in post),len(post)))
