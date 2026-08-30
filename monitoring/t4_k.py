import pickle, numpy as np, collections
GEO=pickle.load(open("_t4_geom.pkl","rb"))
ks=[k for k in sorted(GEO) if k.split("_")[-1] in
    "001 002 003 004 005 006 007 008 010 011 012 016 017 018 020 021 022 023 024 025 026 027".split()]
print("cohort n=",len(ks))
q=np.arange(0,200,0.25)
def stack(f): return np.array([np.interp(q,GEO[k]["q"],GEO[k][f]) for k in ks])
Rc=stack("Rc5"); B=stack("b10")
print(" s     Rc5 mean  min..max   sd     bend10 mean min..max  sd")
for s in (5,11.5,20,40,60,80,100,110,117.5,125,130,133.6,140,150,160,170,180,195):
    i=int(s/0.25)
    print(f"{s:6.1f} {Rc[:,i].mean():8.2f} {Rc[:,i].min():6.2f}..{Rc[:,i].max():7.2f} {Rc[:,i].std():7.3f}   "
          f"{B[:,i].mean():7.1f} {B[:,i].min():6.1f}..{B[:,i].max():6.1f} {B[:,i].std():6.2f}")
