import csv, math
from collections import defaultdict

P = "saved/eve_paper/neurovascular/full/mesh_ben/2026-07-25_022443_rcca_p2_teacher_v1bp/checkpoints/eval_anatomies_checkpoint2002292/episodes.csv"
rows = list(csv.DictReader(open(P)))
for r in rows:
    r['path_len_mm'] = float(r['path_len_mm']); r['success'] = int(r['success']); r['steps']=int(r['steps'])
print("N rows", len(rows), "successes", sum(r['success'] for r in rows))

def wilson(k, n, z=1.96):
    if n == 0: return (float('nan'),)*3
    p = k/n; d = 1+z*z/n
    c = (p + z*z/(2*n))/d
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return 100*p, 100*max(0,c-h), 100*min(1,c+h)

def fmt(k,n):
    p,lo,hi = wilson(k,n); return "%d/%d = %.1f%% [%.1f, %.1f]" % (k,n,p,lo,hi)

anat = defaultdict(list)
for r in rows: anat[r['anatomy']].append(r)
print("\nper-anatomy:")
for a in sorted(anat):
    v = anat[a]; print("  %-14s %s  plen %.1f-%.1f" % (a, fmt(sum(x['success'] for x in v), len(v)),
          min(x['path_len_mm'] for x in v), max(x['path_len_mm'] for x in v)))

# mr_025 split at plen 200.06 (s_RCCA 166.75)
m = anat['topcowmr025']
print("\nmr_025 episodes:", sorted([(x['path_len_mm'], x['success'], x['steps']) for x in m]))
prox = [x for x in m if x['path_len_mm'] < 200.06]; dist = [x for x in m if x['path_len_mm'] >= 200.06]
print("  prox", fmt(sum(x['success'] for x in prox), len(prox)), " dist", fmt(sum(x['success'] for x in dist), len(dist)))

def subset(pred):
    v = [r for r in rows if pred(r)]; return sum(x['success'] for x in v), len(v)

bands = [("overall", lambda r: True),
         ("siphon", lambda r: r['section']=='siphon'),
         ("ICA-mid", lambda r: r['section']=='ICA-mid'),
         ("CCA", lambda r: r['section']=='CCA'),
         ("167-200", lambda r: 166.91 < r['path_len_mm'] < 200),
         ("200-240", lambda r: 200 <= r['path_len_mm'] < 240),
         (">=240", lambda r: r['path_len_mm'] >= 240)]

EXCL = {  # episode-level exclusion sets by rule
 "raw / wire 0.18mm (none)": lambda r: False,
 "mr_025 only (catheter 0.35 / co-located)": lambda r: r['anatomy']=='topcowmr025' and r['path_len_mm']>=200.06,
 "mr_025 + mr_004 (contactDistance 0.30)": lambda r: (r['anatomy']=='topcowmr025' and r['path_len_mm']>=200.06) or (r['anatomy']=='topcowmr004' and r['path_len_mm']>=252.8),
}
print()
for name, ex in EXCL.items():
    print(name)
    for bn, bp in bands:
        k,n = subset(lambda r, bp=bp, ex=ex: bp(r) and not ex(r))
        print("   %-10s %s" % (bn, fmt(k,n)))

# Fisher exact
def lchoose(n,k):
    return math.lgamma(n+1)-math.lgamma(k+1)-math.lgamma(n-k+1)
def fisher2(a,b,c,d):
    n=a+b+c+d; r1=a+b; c1=a+c
    p0=math.exp(lchoose(r1,a)+lchoose(c+d,c)-lchoose(n,c1))
    tot=0.0
    lo=max(0,c1-(c+d)); hi=min(r1,c1)
    for x in range(lo,hi+1):
        p=math.exp(lchoose(r1,x)+lchoose(c+d,c1-x)-lchoose(n,c1))
        if p<=p0*(1+1e-9): tot+=p
    return min(1.0,tot)
def fisher1_greater(a,b,c,d):
    n=a+b+c+d; r1=a+b; c1=a+c
    tot=0.0
    for x in range(a,min(r1,c1)+1):
        tot+=math.exp(lchoose(r1,x)+lchoose(c+d,c1-x)-lchoose(n,c1))
    return min(1.0,tot)

print("\nmr_025 prox-vs-dist one-sided Fisher p = %.5f" % fisher1_greater(4,0,0,5))
print("Teacher vs H0, all depths 4 holdout: 90/98 vs 75/98  two-sided p = %.4f" % fisher2(90,8,75,23))
print("  past-seam uncorrected 47/55 vs 31/42  p = %.4f" % fisher2(47,8,31,11))
print("  past-seam H0 trim-2   47/55 vs 31/40  p = %.4f" % fisher2(47,8,31,9))
print("  past-seam cath0.35    44/52 vs 28/38  p = %.4f" % fisher2(44,8,28,10))
for lab,(k,n) in [("teacher all",(90,98)),("H0 all",(75,98)),("teacher past-seam",(47,55)),("H0 past-seam",(31,42))]:
    print("  %-20s %s" % (lab, fmt(k,n)))

# ceiling
print("\nceiling: 1 - 97/4214 = %.2f%%   (mr_025 only, catheter 0.35)" % (100*(1-97/4214)))
print("normalised teacher: raw 75.0 / 97.70 = %.1f%% of achievable" % (100*0.75/0.9770))
print("corrected 76.7 / 97.70 = %.1f%%" % (100*(165/215)/0.9770))
