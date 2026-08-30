import json,sys,collections
OFF=43.85
def q(v,p):
    if not v: return float('nan')
    v=sorted(v); i=min(len(v)-1,int(round(p*(len(v)-1))))
    return v[i]
def med(v): return q(v,0.5)
def load(t): return [json.loads(l) for l in open('monitoring/cell3_%s.jsonl'%t)]
TAGS=sys.argv[1:]
for t in TAGS:
    E=load(t)
    n=len(E); tot=sum(e['steps'] for e in E)
    print("="*78); print(t, " n=%d steps=%d median_ep_len=%.0f  p90=%.0f max=%d"%(n,tot,med([e['steps'] for e in E]),q([e['steps'] for e in E],.9),max(e['steps'] for e in E)))
    print("  success %d/%d = %.1f%%"%(sum(1 for e in E if e['succ']),n,100*sum(1 for e in E if e['succ'])/n))
    for ss in [4,6,8,12]:
        k=str(ss)
        ns=sum(len(e['ev'][k]) for e in E)
        print("   stuck_steps=%-3d stalls=%-4d /ep=%.3f  /1000steps=%.2f  %%eps>=1stall=%.1f"%(
            ss,ns,ns/n,1000*ns/tot,100*sum(1 for e in E if e['ev'][k])/n))
    ev=[x for e in E for x in e['ev']['12']]
    mix=collections.Counter(x['k'] for x in ev)
    tota=len(ev)
    print("  CANON(12) recovery mix: "+"  ".join("%s=%d(%.1f%%)"%(k,mix[k],100*mix[k]/tota if tota else 0) for k in ['grind','soft','hard','unrec']))
    res=[x for x in ev if x['k']!='unrec']
    print("  resolved fraction = %d/%d = %.1f%%"%(len(res),tota,100*len(res)/tota if tota else 0))
    r=[x['r'] for x in res]
    print("  retraction of resolved (mm): median %.2f  p90 %.2f  max %.2f"%(med(r),q(r,.9),max(r) if r else float('nan')))
    # conditional success
    g0=[e for e in E if not e['ev']['12']]
    gr=[e for e in E if e['ev']['12'] and all(x['k']!='unrec' for x in e['ev']['12'])]
    gu=[e for e in E if any(x['k']=='unrec' for x in e['ev']['12'])]
    for lab,g in [('no stall',g0),('stalled+all recovered',gr),('stalled, >=1 unrecovered',gu)]:
        s=sum(1 for e in g if e['succ'])
        print("  P(success | %-24s) = %d/%d = %s"%(lab,s,len(g),'%.1f%%'%(100*s/len(g)) if g else 'n/a'))
    mr=[e['maxrun'] for e in E]
    print("  longest low-adv-while-pushing run: median %.0f p90 %.0f max %d  %%eps>=4 %.1f  %%eps>=12 %.1f"%(
        med(mr),q(mr,.9),max(mr),100*sum(1 for v in mr if v>=4)/n,100*sum(1 for v in mr if v>=12)/n))
    on=[x['onset_proj'] for x in ev]
    if on:
        bands=collections.Counter()
        for v in on:
            b=int((v-OFF)//25)*25
            bands[b]+=1
        print("  onset proj_s: median %.1f p10 %.1f p90 %.1f   (s_RCCA median %.1f)"%(med(on),q(on,.1),q(on,.9),med(on)-OFF))
        print("  onset s_RCCA 25mm bands: "+"  ".join("[%d,%d)=%d"%(b,b+25,bands[b]) for b in sorted(bands)))
    # section split
    print("  --- by section ---")
    for sec in ['CCA','ICA-mid','siphon']:
        G=[e for e in E if e['sec']==sec]
        if not G: continue
        st=sum(e['steps'] for e in G); evs=[x for e in G for x in e['ev']['12']]
        m=collections.Counter(x['k'] for x in evs)
        s=sum(1 for e in G if e['succ'])
        rr=[x['r'] for x in evs if x['k']!='unrec']
        g0=[e for e in G if not e['ev']['12']]; gu=[e for e in G if any(x['k']=='unrec' for x in e['ev']['12'])]
        gr=[e for e in G if e['ev']['12'] and not any(x['k']=='unrec' for x in e['ev']['12'])]
        f=lambda g:'%.0f%%(%d/%d)'%(100*sum(1 for e in g if e['succ'])/len(g),sum(1 for e in g if e['succ']),len(g)) if g else 'n/a'
        print("   %-8s n=%2d succ=%.1f%%  medlen=%.0f  steps=%d  stalls=%d /ep=%.2f /1k=%.2f  %%eps_stall=%.0f"%(
            sec,len(G),100*s/len(G),med([e['steps'] for e in G]),st,len(evs),len(evs)/len(G),1000*len(evs)/st,
            100*sum(1 for e in G if e['ev']['12'])/len(G)))
        print("            mix g/s/h/u = %d/%d/%d/%d  resolved %.0f%%  retr med %.1f p90 %.1f | P(succ|nostall)=%s P(succ|rec)=%s P(succ|unrec)=%s"%(
            m['grind'],m['soft'],m['hard'],m['unrec'],
            100*(len(evs)-m['unrec'])/len(evs) if evs else 0, med(rr) if rr else float('nan'), q(rr,.9) if rr else float('nan'),
            f(g0),f(gr),f(gu)))
        if evs:
            o=[x['onset_proj'] for x in evs]
            print("            onset proj_s med %.1f (s_RCCA %.1f) p10 %.1f p90 %.1f"%(med(o),med(o)-OFF,q(o,.1),q(o,.9)))
    # failure reasons
    rc=collections.Counter((e['reason'] or 'NO_OUTCOME_LINE') for e in E if not e['succ'])
    print("  failure reasons (unsuccessful eps):",dict(rc))
