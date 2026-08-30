import json, sys
ev=[json.loads(l) for l in open(sys.argv[1])]
sh=[e for e in ev if e["k"] in ("soft","hard")]
def corr(a,b):
    n=len(a); ma=sum(a)/n; mb=sum(b)/n
    ca=sum((x-ma)**2 for x in a)**.5; cb=sum((x-mb)**2 for x in b)**.5
    return sum((a[i]-ma)*(b[i]-mb) for i in range(n))/(ca*cb+1e-12)
print("corr(slack_release, retract) = %.4f"%corr([e["slack_release"] for e in sh],[e["retract"] for e in sh]))
print("corr(slack_rise, gw_fed)     = %.4f"%corr([e["slack_rise"] for e in sh],[e["gw_fed"] for e in sh]))
print("corr(slack_rise, fold_max)   = %.4f"%corr([e["slack_rise"] for e in sh],[float(e["fold_max"]) for e in sh]))
print("corr(slack_rise, adv_25)     = %.4f"%corr([e["slack_rise"] for e in sh],[e["adv_25"] for e in sh]))
print("corr(retract, adv_25)        = %.4f"%corr([e["retract"] for e in sh],[e["adv_25"] for e in sh]))
print("ratio slack_release/retract: med %.3f"%sorted(e["slack_release"]/max(e["retract"],1e-9) for e in sh)[len(sh)//2])
print()
print("FINE adv_25 histogram, 0.5mm bins 0..16")
for i in range(32):
    lo=i*0.5; c=sum(1 for e in sh if lo<=e["adv_25"]<lo+0.5)
    print("  [%5.1f,%5.1f) %3d %s"%(lo,lo+0.5,c,"#"*c))
print("  >=16.0 %3d"%sum(1 for e in sh if e["adv_25"]>=16))
print()
print("FINE adv_50 histogram, 1mm bins 0..20")
for i in range(20):
    c=sum(1 for e in sh if i<=e["adv_50"]<i+1)
    print("  [%4d,%4d) %3d %s"%(i,i+1,c,"#"*c))
print("  >=20 %3d"%sum(1 for e in sh if e["adv_50"]>=20))
print()
print("joint: adv_25 bucket x slack_rise bucket x restall")
for lab,f in [("adv25<2",lambda e:e["adv_25"]<2),("2<=adv25<10",lambda e:2<=e["adv_25"]<10),("adv25>=10",lambda e:e["adv_25"]>=10)]:
    g=[e for e in sh if f(e)]
    rs=sum(1 for e in g if e.get("restall_dp") is not None and e["restall_dp"]<=2.0)
    print("  %-12s n=%2d  succ=%.3f  med_slack_rise=%5.2f med_fold_max=%4.1f med_retract=%6.2f restall_same=%d med_adv_end=%6.1f"
          %(lab,len(g),sum(1 for e in g if e["succ"])/max(1,len(g)),
            sorted(e["slack_rise"] for e in g)[len(g)//2] if g else 0,
            sorted(float(e["fold_max"]) for e in g)[len(g)//2] if g else 0,
            sorted(e["retract"] for e in g)[len(g)//2] if g else 0, rs,
            sorted(e["adv_end"] for e in g)[len(g)//2] if g else 0))
print()
print("restall(same station, dp<=2mm, gap<=20) vs adv:")
for lab,f in [("restall_same",lambda e:e.get("restall_dp") is not None and e["restall_dp"]<=2.0),
              ("restall_far",lambda e:e.get("restall_dp") is not None and e["restall_dp"]>2.0),
              ("no_restall",lambda e:e.get("restall_dp") is None)]:
    g=[e for e in sh if f(e)]
    if not g: print("  %-13s n=0"); continue
    print("  %-13s n=%2d med_adv25=%6.2f med_adv_end=%6.2f succ=%.3f"%(lab,len(g),
        sorted(e["adv_25"] for e in g)[len(g)//2],sorted(e["adv_end"] for e in g)[len(g)//2],
        sum(1 for e in g if e["succ"])/len(g)))
print()
print("low-load events (slack_rise<1): n=%d"%sum(1 for e in sh if e["slack_rise"]<1))
for e in sh:
    if e["slack_rise"]<1.5:
        print("   seed=%s k=%s ret=%6.2f rise=%5.2f gw_fed=%5.2f foldmax=%3d rel=%5.2f adv25=%6.2f advend=%6.2f succ=%d"
              %(e["seed"],e["k"],e["retract"],e["slack_rise"],e["gw_fed"],e["fold_max"],e["slack_release"],e["adv_25"],e["adv_end"],e["succ"]))
