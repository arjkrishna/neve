import glob,json,os
import numpy as np
exec(open("monitoring/attack2_holdout_geometry.py").read().split("def analyse")[0].replace("main()",""))
M=4;OFF=33.5
def menger(p,m):
    a=p[:-2*m];b=p[m:-m];c=p[2*m:]
    ab=np.linalg.norm(b-a,axis=1);bc=np.linalg.norm(c-b,axis=1);ca=np.linalg.norm(a-c,axis=1)
    ar=0.5*np.linalg.norm(np.cross(b-a,c-a),axis=1);den=ab*bc*ca
    return np.where(den>1e-12,4*ar/np.maximum(den,1e-12),0.0)
def prof(tag):
    f=os.path.join(ANAT,tag,"Centrelines_comb",RCCA_FILE)
    p,r=read_curve(f);s=arclength(p);k=menger(p,M);sk=s[M:len(p)-M]
    return s,r,sk,k,p
for tag,lo,hi in (("topcow_mr_023",160,215),("topcow_mr_004",125,165),("topcow_mr_017",145,170)):
    s,r,sk,k,p=prof(tag)
    print("=== %s   L=%.1f ==="%(tag,s[-1]))
    print("  s_mm  path_len   r_mm   Rc_mm")
    for i in range(len(sk)):
        if lo<=sk[i]<=hi and abs(sk[i]-round(sk[i]))<0.6 and int(round(sk[i]))%2==0:
            ri=np.interp(sk[i],s,r)
            print("  %5.1f   %6.1f   %5.2f  %6.2f"%(sk[i],sk[i]+OFF,ri,1/max(k[i],1e-9)))
    print()
