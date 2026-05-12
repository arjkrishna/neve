"""Cross-daughter episode analysis for RL_IMPROV_8 — RVA / RCCA / LVA 500-ep runs."""
import os, re, glob
import numpy as np
from collections import Counter, defaultdict


def analyze(run_dir, daughter):
    log_dir = os.path.join(run_dir, "diagnostics", "logs_subprocesses")
    eps = {}
    for path in sorted(glob.glob(os.path.join(log_dir, "worker_*.log"))):
        target = None
        cur_pid, cur_ep = None, None
        ep_state = None
        for line in open(path, 'r', encoding='utf-8', errors='ignore'):
            if 'EPISODE_START' in line:
                if ep_state:
                    eps[(cur_pid, cur_ep)] = ep_state
                m_pid = re.search(r'pid=(\d+)', line); cur_pid = m_pid.group(1) if m_pid else None
                m_ep = re.search(r'ep=(\d+)', line); cur_ep = m_ep.group(1) if m_ep else None
                m_t = re.search(r'target=\(([^)]+)\)', line)
                target = np.array([float(x) for x in m_t.group(1).split(',')]) if m_t else None
                ep_state = {'target': target, 'final_R': None, 'abort': None, 'steps': None,
                    'max_arc_daughter': 0, 'max_z_daughter': 0,
                    'min_d_target': 1e9, 'ever_daughter': False, 'last_br': '?', 'final_tip': None,
                    'term_true': False, 'n_off_path': 0, 'entry_rewards': 0, 'pos_rewards': 0,
                    'step_count': 0, 'arc_progress_rewards': 0, 'first_daughter_step': None}
                continue
            if not ep_state: continue
            if 'EPISODE_END' in line:
                m_r = re.search(r'total_reward=([-0-9.]+)', line)
                m_a = re.search(r'heur_abort=(\w+)', line)
                m_st = re.search(r'steps=(\d+)', line)
                if m_r: ep_state['final_R'] = float(m_r.group(1))
                if m_a: ep_state['abort'] = m_a.group(1)
                if m_st: ep_state['steps'] = int(m_st.group(1))
                continue
            if 'STEP' not in line: continue
            ep_state['step_count'] += 1
            if 'term=True' in line: ep_state['term_true'] = True
            m_tip = re.search(r'tip3d=\(([^)]+)\)', line)
            m_br = re.search(r'cur_branch=([^|]+)', line)
            m_arc = re.search(r'arc_past_d=([\d.]+|inf)', line)
            m_onp = re.search(r'on_path=(\d)', line)
            m_r = re.search(r' reward=([-0-9.]+)', line)
            m_es = re.search(r'ep_step=(\d+)', line)
            if m_tip: ep_state['final_tip'] = m_tip.group(1)
            if m_onp and m_onp.group(1) == '0': ep_state['n_off_path'] += 1
            if m_r:
                r = float(m_r.group(1))
                if r > 0.5: ep_state['entry_rewards'] += 1
                if 0.001 < r < 0.5: ep_state['arc_progress_rewards'] += 1
                if r > 0.001: ep_state['pos_rewards'] += 1
            if m_br:
                br = m_br.group(1).strip()
                ep_state['last_br'] = br
                if daughter + '.mrk' in br:
                    if not ep_state['ever_daughter']:
                        ep_state['ever_daughter'] = True
                        ep_state['first_daughter_step'] = int(m_es.group(1)) if m_es else None
                    if m_arc and m_arc.group(1) != 'inf':
                        try:
                            a = float(m_arc.group(1))
                            if a > ep_state['max_arc_daughter']: ep_state['max_arc_daughter'] = a
                        except: pass
                    if m_tip:
                        try:
                            z = float(m_tip.group(1).split(',')[2])
                            if z > ep_state['max_z_daughter']: ep_state['max_z_daughter'] = z
                        except: pass
            if m_tip and target is not None:
                tip = np.array([float(x) for x in m_tip.group(1).split(',')])
                d = float(np.linalg.norm(tip - target))
                if d < ep_state['min_d_target']: ep_state['min_d_target'] = d
        if ep_state:
            eps[(cur_pid, cur_ep)] = ep_state
    return eps


runs = {
    "RVA":  (r"d:/neve/.claude/worktrees/rl_improv_8/saved/eve_paper/neurovascular/full/mesh_ben/2026-05-11_202046_env5_rl8_RVA_v1_500ep",  "RVA"),
    "RCCA": (r"d:/neve/.claude/worktrees/rl_improv_8/saved/eve_paper/neurovascular/full/mesh_ben/2026-05-10_033729_env5_rl8_RCCA_v1_500ep", "RCCA"),
    "LVA":  (r"d:/neve/.claude/worktrees/rl_improv_8/saved/eve_paper/neurovascular/full/mesh_ben/2026-05-11_042358_env5_rl8_LVA_v2_500ep",  "LVA"),
}


def sample_show(cat_name, cat, n=2):
    print(f"\n  --- {cat_name} sample ---")
    for (pid, ep), v in cat[:n]:
        target = v['target']
        tip = v.get('final_tip', '?')
        print(f"    pid={pid:>3} ep={ep:>3} steps={v['steps']:>3} R={v['final_R']:>6.2f} abort={v['abort']:<22} | "
              f"target=({target[0]:.1f},{target[1]:.1f},{target[2]:.1f}) "
              f"final_tip=({tip})")
        print(f"      min_d_to_target={v['min_d_target']:.2f}mm  max_arc_in_daughter={v['max_arc_daughter']:.1f}mm  "
              f"max_z={v['max_z_daughter']:.1f}  "
              f"entry+1s={v['entry_rewards']} arc_progress_rs={v['arc_progress_rewards']} "
              f"off_path={v['n_off_path']}/{v['step_count']}")


for label, (run, daughter) in runs.items():
    print(f"\n{'='*78}\n=== {label} 500-ep run ===\n{'='*78}")
    eps = analyze(run, daughter)
    ended = {k: v for k, v in eps.items() if v['steps'] is not None and v['final_R'] is not None}

    successes = [(k, v) for k, v in ended.items() if v['term_true']]
    near = [(k, v) for k, v in ended.items() if not v['term_true'] and v['ever_daughter'] and v['min_d_target'] < 10]
    stuck = [(k, v) for k, v in ended.items() if not v['term_true'] and v['ever_daughter'] and v['min_d_target'] >= 10]
    wbt = [(k, v) for k, v in ended.items() if v['abort'] == 'wrong_branch_timeout']
    wedge = [(k, v) for k, v in ended.items() if not v['ever_daughter'] and v['abort'] == 'none']
    fold = [(k, v) for k, v in ended.items() if v['abort'] == 'wire_fold_stall']

    print(f"  Total ended: {len(ended)}")
    print(f"  SUCCESS (term=True):                              {len(successes):>4}")
    print(f"  NEAR-MISS in daughter (no term, min_d<10mm):      {len(near):>4}")
    print(f"  STUCK in daughter (no term, min_d>=10mm):         {len(stuck):>4}")
    print(f"  WRONG_BRANCH_TIMEOUT (deflected):                 {len(wbt):>4}")
    print(f"  TRUNK WEDGE (no abort, never reached daughter):   {len(wedge):>4}")
    print(f"  WIRE FOLD STALL:                                  {len(fold):>4}")

    if successes: sample_show("SUCCESS", successes, 3)
    if near: sample_show("NEAR-MISS in daughter", near, 3)
    if stuck: sample_show("STUCK deep (no term, far from target)", stuck, 3)
    if wbt: sample_show("WRONG_BRANCH_TIMEOUT", wbt, 2)
    if wedge: sample_show("TRUNK WEDGE", wedge, 2)
    if fold: sample_show("WIRE FOLD STALL", fold, 2)

    # Reward distribution by category
    print(f"\n  --- Reward distribution per category (mean ± std, count) ---")
    for cname, cat in [("SUCCESS", successes), ("NEAR-MISS", near), ("STUCK", stuck),
                       ("WBT", wbt), ("TRUNK_WEDGE", wedge), ("FOLD", fold)]:
        if cat:
            rs = [v['final_R'] for k, v in cat]
            print(f"    {cname:<14} mean R = {np.mean(rs):>7.2f} ± {np.std(rs):>5.2f}  (n={len(rs)})")
