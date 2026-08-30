import json,collections
rows=json.load(open(r"D:/Arjun/workspace/neve/monitoring/_t2_eval_rows.json"))
# are missing-outcome episodes the last per (worker,eval)?
byw=collections.defaultdict(list)
for r in rows: byw[(r["worker"],r["eval"])].append(r)
miss_last=0; miss_other=0
for k,v in byw.items():
    v.sort(key=lambda r:int(r["ep"]))
    for i,r in enumerate(v):
        if not r["reason"]:
            if i==len(v)-1: miss_last+=1
            else: miss_other+=1
print("missing-outcome that are last-in-worker:",miss_last," other:",miss_other)
print("eps per worker per eval:",collections.Counter(len(v) for v in byw.values()))
# correlate reason with term/trunc
c=collections.Counter((r["reason"],r["last_term"],r["last_trunc"]) for r in rows if r["reason"])
for k,v in sorted(c.items(),key=lambda x:-x[1]): print(k,v)
print("--- reasons ---",collections.Counter(r["reason"] for r in rows))
print("--- grader_success vs reason ---",collections.Counter((r["reason"],r["grader_success"]) for r in rows if r["reason"]))
# missing-outcome episodes: their term/trunc/steps
print("--- missing rows summary ---")
for r in rows:
    if not r["reason"]:
        print(r["eval"],r["worker"],r["ep"],"seed",r["seed"],r["mesh"],"steps",r["n_steps"],"term",r["last_term"],"trunc",r["last_trunc"],"d_tgt",r["last_d_tgt"],"cum_r",r["cum_reward"])
