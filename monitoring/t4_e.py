import pickle, numpy as np
eps=pickle.load(open("_t4_teacher.pkl","rb"))
by={e["seed"]:e for e in eps}
for sd in (900105, 900069, 900020):
    e=by[sd]
    ps=np.array(e["projs"])-33.31; i0=np.array(e["ins0"]); i1=np.array(e["ins1"])
    d1=np.array(e["dins1"]); fo=np.array(e["fold"]); sl=np.array(e["slack"])
    print(f"=== seed {sd} {e['mesh']} n={len(ps)} tgt_pl={e['path_len']:.1f} (s={e['path_len']-33.31:.1f})")
    idx=list(range(0,len(ps),40))+[len(ps)-1]
    for i in idx:
        print(f"  t={i:4d} s={ps[i]:7.1f} ins=[{i0[i]:7.1f},{i1[i]:7.1f}] d1={d1[i]:+6.2f} fold={fo[i]:3d} slack={sl[i]:6.1f}")
    print(f"  ins1: min {i1.min():.1f} max {i1.max():.1f} final {i1[-1]:.1f}; ins0 max {i0.max():.1f}")
    print(f"  s: max {ps.max():.1f} at t={int(np.argmax(ps))}; frac steps with s>5: {(ps>5).mean():.2f}")
    print(f"  d1 push frac {(d1>0).mean():.2f} mean|d1| {np.abs(d1).mean():.2f}")
