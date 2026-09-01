import json,os,re,math,random,glob
from collections import defaultdict
SCR=r"C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-workspace-neve\81b186b6-3a3f-4f63-8491-2172316ef81f\scratchpad"
AN=r"D:\Arjun\workspace\neve\carotid_data\anatomies"
CK8=r"D:\Arjun\workspace\neve\monitoring\ak_check8_out"

names=sorted(os.listdir(AN))
prov={}
mismatch=[]
for n in names:
    p=json.load(open(os.path.join(AN,n,"provenance.json")))
    lo,si=p["lower"],p["siphon"]
    prov[n]=(lo,si,p)
    # directory name: case_<lower>_<side>__topcow_mr_NNN[_L]
    dlo,dsi=n.split("__")
    dsi="topcow_"+dsi if not dsi.startswith("topcow") else dsi
    if dlo!=lo or dsi!=si: mismatch.append((n,lo,si))
print("N",len(names),"prov mismatch vs dirname:",len(mismatch))
for m in mismatch[:20]: print("  MISMATCH",m)

LOW={n:prov[n][0] for n in names}
SIP={n:prov[n][1] for n in names}
lowc=defaultdict(int); sipc=defaultdict(int)
for n in names: lowc[LOW[n]]+=1; sipc[SIP[n]]+=1
print("lower donors",len(lowc),"sizes",sorted(set(lowc.values())))
print("siphon donors",len(sipc),"sizes",sorted(set(sipc.values())))
# crossing structure
pairs=set((LOW[n],SIP[n]) for n in names)
print("distinct (lower,siphon) pairs",len(pairs))

D=defaultdict(dict)   # defect -> name -> 0/1
AUX={}

# ---------- check5b.out : mesh health ----------
rows5={}
for line in open(os.path.join(SCR,"check5b.out")):
    if line.startswith("ROW "):
        r=json.loads(line[4:]); rows5[r["name"]]=r
print("check5b rows",len(rows5))
for n in names:
    r=rows5[n]
    D["NONMANIFOLD_ge3"][n]=1 if r["main_nm"]>=3 else 0
    D["NONMANIFOLD_ge2"][n]=1 if r["main_nm"]>=2 else 0
    stub=any(60<=c["cells"]<=80 and 340<=c["area"]<=395 and 41<=c["diag"]<=44 for c in r["comps"])
    D["RVA_STUB_MISSING"][n]=0 if stub else 1
    mid=any(7<=c["cells"]<=12 for c in r["comps"][1:])
    D["MID_FRAGMENT"][n]=1 if mid else 0
    D["OPENBOUND_main_ge3"][n]=1 if r["main_ob"]>=3 else 0

# ---------- check6a.txt : RCCA blockage ----------
def parse6a(path):
    out={}
    for line in open(path):
        if not line.startswith("ROW,"): continue
        f=line.strip().split(",")
        nm=f[1]; L=float(f[2]); nst=int(f[3]); mn=float(f[4])
        rec={"L":L,"min":mn}
        for seg in f[6:]:
            p=seg.split("|"); tag=p[0]; nblk=int(p[1])
            runs=[]
            if p[3]!="-":
                for rr in p[3].split("+"):
                    m=re.match(r"([\d.]+)@([\d.]+)-([\d.]+)#(\d+)",rr)
                    runs.append((float(m.group(1)),float(m.group(2)),float(m.group(3)),int(m.group(4))))
            rec[tag]={"nblk":nblk,"runs":runs,"termonly":(int(p[4]) if p[4]!="-" else 1)}
        out[nm]=rec
    return out
r6=parse6a(os.path.join(SCR,"check6a.txt"))
print("check6a rows",len(r6))
for n in names:
    r=r6[n]; L=r["L"]
    for tag,key in (("gw","018"),("sofa","030"),("cath","035")):
        s=r[tag]
        D["RCCA_MIDVESSEL_"+key][n]=1 if (s["nblk"]>0 and s["termonly"]==0) else 0
        D["RCCA_ANYBLOCK_"+key][n]=1 if s["nblk"]>0 else 0
        deep=any((L-st)>40.0 for _,st,_,_ in s["runs"])
        D["RCCA_DEEP40_"+key][n]=1 if deep else 0
    D["RCCA_NEGCLEAR"][n]=1 if r["min"]<0 else 0
    D["RCCA_MIN_lt020"][n]=1 if r["min"]<0.20 else 0
    AUX[n]={"rcca_min":r["min"],"L":L}

# ---------- reca6b_v2.txt ----------
hdr=None; reca={}
for line in open(os.path.join(SCR,"reca6b_v2.txt")):
    f=line.split()
    if not f: continue
    if f[0]=="HDR": hdr=f[1:]; continue
    if f[0]!="RECA": continue
    v=f[1:]
    d=dict(zip(hdr,v))
    reca[v[0]]=d
print("reca rows",len(reca))
for n in names:
    d=reca[n]
    sdiv=float(d["s_div"])
    D["RECA_MID_018"][n]=1 if "M" in d["cls0.18"] else 0
    D["RECA_MID_035"][n]=1 if "M" in d["cls0.35"] else 0
    D["RECA_TIP_OUTSIDE"][n]=1 if float(d["tip_sd"])<0 else 0
    D["RECA_WEDGE_035_lt10mm"][n]=1 if (float(d["d0.35"])-sdiv)<10.0 else 0
    D["RECA_WEDGE_065_lt10mm"][n]=1 if (float(d["d0.65"])-sdiv)<10.0 else 0

# ---------- check8 fusion ----------
w2={r["name"]:r for r in json.load(open(os.path.join(CK8,"check8_wall2.json")))}
v2={r["name"]:r for r in json.load(open(os.path.join(CK8,"check8_v2.json")))}
print("check8 rows",len(w2),len(v2))
for n in names:
    a=w2[n]; b=v2[n]
    D["ECA_TIP_FUSED_ICA"][n]=1 if len(a.get("extra_zero_runs",[]))>0 else 0
    D["RCCA_RVA_FUSED"][n]=1 if len(a.get("rcca_rva_zero_runs",[]))>0 else 0
    D["RCCA_RVA_wall_lt030"][n]=1 if a.get("rcca_rva_wall_min",9)<0.30 else 0
    D["AUDIT22_lt090"][n]=1 if b["aud_min"]<0.90 else 0
    D["RCCA_RECA_declared_lt090"][n]=1 if b["reap_p3"]<0.90 else 0

# ---------- handedness (deterministic in the lower donor) ----------
for n in names:
    D["LOWER_LEFT_UNMIRRORED"][n]=1 if LOW[n].endswith("_left") else 0

json.dump({k:dict(v) for k,v in D.items()},open(os.path.join(SCR,"ck15_defects.json"),"w"))
json.dump({"low":LOW,"sip":SIP},open(os.path.join(SCR,"ck15_donors.json"),"w"))
for k in sorted(D): print("DEFECT %-26s n=%d" % (k,sum(D[k].values())))
