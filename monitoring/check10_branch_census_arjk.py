import os, json, hashlib, glob, sys
import numpy as np

ROOT = r"D:\Arjun\workspace\neve\carotid_data\anatomies"
HOST = r"D:\Arjun\workspace\neve\eve_bench\data\dualdevicenav\Centrelines_comb"

def raw_md5(p):
    return hashlib.md5(open(p,'rb').read()).hexdigest()

def positions(p):
    d = json.load(open(p))
    m = d['markups'][0]
    return np.array([c['position'] for c in m['controlPoints']], dtype=float), m['coordinateSystem']

def geo_md5(P):
    return hashlib.md5(P.tobytes()).hexdigest()

anats = sorted(os.listdir(ROOT))
print("N anatomies:", len(anats))

# branch name census
namesets = {}
for a in anats:
    fs = tuple(sorted(os.path.basename(f) for f in glob.glob(os.path.join(ROOT,a,'Centrelines_comb','*.mrk.json'))))
    namesets.setdefault(fs, []).append(a)
print("distinct branch-name sets:", len(namesets))
for fs, aa in namesets.items():
    print("  nfiles=%d  n_anat=%d" % (len(fs), len(aa)))
BRANCHES = list(list(namesets.keys())[0])
for b in BRANCHES: print("   ", b)

hostfiles = {os.path.basename(f): f for f in glob.glob(os.path.join(HOST,'*.mrk.json'))}
print("host nfiles:", len(hostfiles))
print("host-only:", sorted(set(hostfiles)-set(BRANCHES)))
print("anat-only:", sorted(set(BRANCHES)-set(hostfiles)))

CACHE = {}
def load(a,b):
    k=(a,b)
    if k not in CACHE:
        p = os.path.join(ROOT,a,'Centrelines_comb',b)
        P,cs = positions(p)
        CACHE[k]=(raw_md5(p),geo_md5(P),P,cs)
    return CACHE[k]

rows=[]
for b in BRANCHES:
    rawh={}; geoh={}; npts={}
    for a in anats:
        r,g,P,cs = load(a,b)
        rawh.setdefault(r,[]).append(a)
        geoh.setdefault(g,[]).append(a)
        npts.setdefault(len(P),0); npts[len(P)]+=1
    hp = hostfiles.get(b)
    hostraw=hostgeo=None
    if hp:
        HP,_ = positions(hp)
        hostraw = raw_md5(hp); hostgeo = geo_md5(HP)
    n_raw=len(rawh); n_geo=len(geoh)
    # how many anatomies match host geometry
    nhost = len(geoh.get(hostgeo,[])) if hostgeo else 0
    nhostraw = len(rawh.get(hostraw,[])) if hostraw else 0
    rows.append((b,n_raw,n_geo,nhostraw,nhost,dict(npts)))
    print("BRANCH %-34s distinct_raw=%-4d distinct_geo=%-4d match_host_raw=%-4d match_host_geo=%-4d npts=%s"
          % (b,n_raw,n_geo,nhostraw,nhost,sorted(npts.items())))
    if n_geo>1 and n_geo<=6:
        for g,aa in sorted(geoh.items(), key=lambda kv:-len(kv[1])):
            print("     geo-group n=%d e.g. %s" % (len(aa), aa[0]))

import pickle
pickle.dump({'branches':BRANCHES,'anats':anats}, open(r"C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-workspace-neve\81b186b6-3a3f-4f63-8491-2172316ef81f\scratchpad\c10.pkl",'wb'))
