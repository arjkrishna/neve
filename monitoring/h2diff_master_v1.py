import json, math, os
import numpy as np
M="monitoring/"
GEO=json.load(open(M+"h2diff_geom.json")); ROWS=GEO["rows"]
CLR=json.load(open(M+"h2diff_clear.json"))
PROF=json.load(open(M+"h2diff_prof.json"))
# consistency: local centerline L vs container L
print("L agreement local-vs-container (mm):")
dd=[]
for a in ROWS:
    k="HOST" if a=="HOST" else a
    dd.append(abs(ROWS[a]["rcca_len"]-CLR[k]["L"]))
print("  max |dL| = %.4f"%max(dd))
# merge
T={}
for a in ROWS:
    r=dict(ROWS[a]); c=CLR[a]
    for k,v in c.items():
        if k!="L": r["clr_"+k if not k.startswith("clr") else k]=v
    T[a]=r
json.dump(T,open(M+"h2diff_table.json","w"),indent=1)
KEYS=["rcca_len","graft_len","tort_graft","tort_w40max","tort_h2","Rc_min","Rc_p05","Rc_p25","Rc_med",
      "n_Rc_lt5","n_Rc_lt8","n_Rc_lt12","bend_max","bend_p90","turn_cum","turn_per_mm","turn_net","turn_eff",
      "n_infl","tors_cum","tors_mean","planarity","frac_top24","turn_top24",
      "r_min","r_p05","r_med","r_term","r_n_lt_066",
      "clr_min","clr_min_nonterm","clr_p05","clr_p25","clr_med","clr_n_lt_cath","clr_n_lt_sofa","clr_f_lt_cath","clr_rdec_min","clr_minus_r_med"]
hdr="%-9s"%"anat"+"".join("%10s"%k[:10] for k in KEYS)
print("\n=== FULL GEOMETRY TABLE (graft region s_RCCA>=133.6) ===")
print(hdr)
order=["HOST"]+sorted([a for a in T if a!="HOST"])
for a in order:
    print("%-9s"%a.replace("topcow_",""),end="")
    print("".join("%10.3f"%T[a].get(k,float('nan')) for k in KEYS))
# host percentile rank of cohort
print("\n=== HOST vs COHORT (n=22) ===")
print("%-14s %10s %10s %10s %10s %6s"%("measure","HOST","coh_med","coh_min","coh_max","#coh more extreme than host"))
for k in KEYS:
    v=[T[a][k] for a in T if a!="HOST"]; h=T["HOST"][k]
    print("%-14s %10.3f %10.3f %10.3f %10.3f    n>%.0f  n<%.0f"%(k,h,np.median(v),min(v),max(v),sum(1 for x in v if x>h),sum(1 for x in v if x<h)))
