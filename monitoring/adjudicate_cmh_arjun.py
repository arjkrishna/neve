import math
# strata: (teacher_succ, teacher_n, h0_succ, h0_n) per anatomy, all depths, 4 holdouts
S = {'mr_004':(24,27,13,17), 'mr_008':(26,26,27,30), 'mr_017':(28,29,16,18), 'mr_023':(12,16,19,33)}
num=den=0.0; Rs=Ss=0.0
for a,(ts,tn,hs,hn) in S.items():
    a11,a12,a21,a22 = ts,tn-ts,hs,hn-hs; n=tn+hn
    num += a11 - (tn*(ts+hs))/n
    den += (tn*hn*(ts+hs)*(n-ts-hs))/(n*n*(n-1)) if n>1 else 0
    Rs += a11*a22/n; Ss += a12*a21/n
chi = (abs(num)-0.5)**2/den
# chi2 1df p
p = math.erfc(math.sqrt(chi/2))
print("all-depth anatomy-stratified CMH chi2=%.3f p=%.4f  MH OR=%.3f" % (chi,p,Rs/Ss if Ss else float('inf')))
# direct standardisation, pooled weights
wt = {a:(S[a][1]+S[a][3]) for a in S}; W=sum(wt.values())
t = sum(wt[a]*S[a][0]/S[a][1] for a in S)/W
h = sum(wt[a]*S[a][2]/S[a][3] for a in S)/W
print("anatomy-standardised all-depth: teacher %.1f%% vs H0 %.1f%%, +%.1f pp" % (100*t,100*h,100*(t-h)))
