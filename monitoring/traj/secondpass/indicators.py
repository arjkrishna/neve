"""indicators.py -- measurable per-episode indicators for two behaviours of the
RCCA navigation policies: CATHETER-LED advancement and COIL / BUCKLE formation.

Three entry points (pandas / numpy / scipy only):

    catheter_led(row_or_series)   -> dict   catheter-tip-ahead-of-wire-tip indicators
    coil_from_features(row)       -> dict   slack-only coil indicators (log-derived episodes)
    coil_from_record(rec)         -> dict   polyline coil geometry (recorded replay episodes)

Geometry facts these rest on (eve + env5, verified in the source):

* Both devices share ONE beam. ``fluoroscopy.tracking3d`` is the whole beam,
  tip first; ``device_trackings3d[k]`` is the proximal ``inserted_length[k]``
  mm of it.  Hence the catheter tip is ahead of the guidewire tip exactly when
  ``ins_cath > ins_gw`` and the lead in mm is ``ins_cath - ins_gw`` (it is the
  arclength between the two tips along the beam, not a projection proxy).
* ``proj_s`` (the STEP log's ``proj``) is the planned-path arclength of the
  FRONTIER tip (``tracking3d[0]`` = the tip of whichever device is further
  in).  So the log's guidewire slack ``ins_gw - proj_s`` is negative while
  the catheter leads (frontier ahead of the wire tip), and ``cath_slack`` =
  ``ins_cath - proj(catheter tip)`` (env5 ``_compute_cath_slack_mm``).
* Recorded polylines (``traj_record.py``) are 64 nodes per device, float16,
  tip first, resampled by arclength.  float16 at z ~ 400-600 mm quantises to
  0.25-0.5 mm, so raw per-node turning angles are noise (thousands of degrees
  on a clean wire).  Every curvature-type measure here first resamples the
  polyline at ``ds`` mm and smooths it with a Gaussian of ``sigma`` mm along
  arclength; the self-closure loop test is robust to the quantisation as is.

Definitions (see saved/traj/secondpass/INDICATORS.md for the validation):

CATHETER-LED
    lead(t)          = ins_cath(t) - ins_gw(t)                        [mm]
    lead_final       = lead at the terminal step  (= cath_final - gw_final,
                       so it is computable from the features jsonl alone)
    adv_led20        = share of the frontier's forward progress (sum of
                       positive d proj_s) gained while lead > 20 mm
    catheter-led SUCCESS  := success and lead_final > 20 mm
                       (the catheter is the leading device when the target is
                       reached; eye-label precision 0.89 / recall 0.86 on the
                       66 blind snapshots, recall 0.92 of the atlas
                       "cathled-sprint" type, 0.8 % of clean successes)
    catheter-led EPISODE  := adv_led20 >= 0.5 (any outcome; needs the series)
                       (recall 0.92 of cathled-sprint, 0.99 of shove-cap,
                       0.59 of shove-fight, 0.8 % of clean successes)

COIL / BUCKLE
    project rule     := gw_slack_max > 100 mm or cath_slack_max > 50 mm
                       (eye-label precision 0.94 / recall 0.89, 18 coils)
    loop (polyline)  := two nodes of the smoothed device polyline within
                       prox_mm (5) of each other whose arclength separation
                       exceeds sep_mm (25): a closed loop exists.  Planned
                       paths never come closer than ~7.6 mm to themselves at
                       that separation (15 carotid anatomies), clean wires
                       13-25 mm, loops < 1 mm.
    multi-loop       := >= 2 connected components of closure pairs in the
                       (i, j) node plane (a double winding or two loops)
    bow (kink)       := no loop but the device stores >= 20 mm of excess
                       length (slack); the near-target buckle is a bow at
                       prog >= 0.9
    grade            := none < bow < loop < multi-loop
    slack-only proxy for "a loop is present" (episodes without polylines):
                       gw_slack > 50 mm or cath_slack > 50 mm.  Per step,
                       P(closed loop | stored length) on 22.6k recorded steps:
                       gw 40-50 mm 0.32, 50-75 0.75, 75-100 0.99, >100 1.00;
                       cath 40-50 0.04, 50-75 0.36, 75-100 0.82, >100 1.00;
                       either device <= 30 mm 0.00.  Per episode (max over
                       steps) the project rule and this proxy both find the
                       7 looped episodes of 214 with no false positive; the
                       proxy differs from the project rule only for wire
                       slack 50-100 mm, where a loop is present 75-99 % of
                       the time.

Host sessions are log-derived (tip only): every catheter-led indicator and
every slack-only coil indicator is computable there for the sessions whose
logs carry the catheter fields (host_topbrain_v1/v2/v3, host_p2_teacher_*);
the converted sessions (host_v1bp_*, host_tbv1r1_*, tb22_*) have neither
catheter insertion nor catheter slack, so only the guidewire-slack rule
applies.  Polyline grades need recorded episodes (none exist for the host).
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional

import numpy as np

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None

# ----------------------------------------------------------------- thresholds
LEAD_FINAL_MM = 20.0        # catheter-led success: catheter tip > 20 mm beyond the wire tip at the end
LEAD_ADV_MM = 20.0          # catheter-led episode: progress counted while lead > 20 mm ...
ADV_LED_FRAC = 0.5          # ... and that share of the forward progress is >= 0.5
GW_COIL_MM = 100.0          # project coil rule (buckle_clear_b_coil_v1): guidewire slack
CATH_COIL_MM = 50.0         #                                             catheter slack
GW_LOOP_PROXY_MM = 50.0     # slack-only proxy for "closed loop present" (either device)
CATH_LOOP_PROXY_MM = 50.0
BOW_MM = 20.0               # stored length at which a bow / buckle is called
MULTI_MM = 200.0            # slack-only proxy for >= 2 loops
LOOP_PROX_MM = 5.0          # polyline self-closure distance
LOOP_SEP_MM = 25.0          # ... at this minimum arclength separation
RESAMPLE_DS_MM = 2.0
SMOOTH_SIGMA_MM = 3.0
GRADES = ("none", "bow", "loop", "multi-loop")


def _f(x, default=float("nan")) -> float:
    try:
        if x is None: return default
        v = float(x)
        return v if v == v else default
    except (TypeError, ValueError):
        return default


def _get(row, *names, default=None):
    for n in names:
        if isinstance(row, dict):
            if n in row and row[n] is not None: return row[n]
        else:
            try:
                v = row[n]
                if v is not None and not (isinstance(v, float) and v != v): return v
            except (KeyError, IndexError, TypeError):
                pass
    return default


# =========================================================== catheter-led
def catheter_led(row_or_series, lead_final_mm: float = LEAD_FINAL_MM, lead_adv_mm: float = LEAD_ADV_MM,
                 adv_frac: float = ADV_LED_FRAC) -> Dict[str, Any]:
    """Catheter-led indicators for one episode.

    Accepts either
      * a FEATURES row (dict / pandas row from ``<tag>.features.jsonl``) --
        uses the scalars ``cath_final``, ``gw_final``, ``cath_lead_max``,
        ``cath_lead_frac50``, ``success``; or
      * a per-step SERIES: a dict with arrays ``ins_gw``/``ins_cath``/``proj_s``
        (a ``traj_render.load_record`` record works as is) or ``gw``/``cath``/
        ``proj`` in mm (a slice of ``<tag>.series.npz`` divided by 10), plus
        optional ``path_len``/``pl`` and ``success``.

    Returns a dict with
      lead_final       mm, catheter tip minus guidewire tip at the last step
      lead_max         mm, max over the episode
      lead_frac0/10/20 fraction of steps with lead > 0 / 10 / 20 mm (series only)
      lead_at50/at90   lead at the first step the frontier reaches 50 % / 90 %
                       of the path (series only; NaN if never)
      adv_led10/20     share of forward progress gained while lead > 10 / 20 mm
      cath_led_success bool: success and lead_final > lead_final_mm
      cath_led_episode bool: adv_led20 >= adv_frac  (None when no series)
      source           "features" or "series"
    NaN / None where a field is unavailable (e.g. converted TopBrain-v1 logs
    carry no catheter insertion at all).
    """
    r = row_or_series
    is_series = False
    if isinstance(r, dict):
        is_series = any(k in r for k in ("ins_gw", "ins_cath")) or (
            "gw" in r and "cath" in r and np.ndim(r.get("gw")) >= 1 and np.asarray(r["gw"]).ndim == 1)
    out: Dict[str, Any] = dict(source="series" if is_series else "features")
    if not is_series:
        cf, gf = _f(_get(r, "cath_final")), _f(_get(r, "gw_final"))
        lead_final = cf - gf if (cf == cf and gf == gf) else float("nan")
        lead_max = _f(_get(r, "cath_lead_max"))
        succ = _get(r, "success")
        out.update(lead_final=lead_final, lead_max=lead_max, lead_frac50=_f(_get(r, "cath_lead_frac50")),
                   lead_frac0=float("nan"), lead_frac10=float("nan"), lead_frac20=float("nan"),
                   lead_at50=float("nan"), lead_at90=float("nan"), adv_led10=float("nan"), adv_led20=float("nan"),
                   cath_led_success=(bool(succ) and lead_final > lead_final_mm) if (succ is not None and lead_final == lead_final) else None,
                   cath_led_episode=None)
        return out
    # ---- series
    if "ins_gw" in r:
        G = np.asarray(r["ins_gw"], float); C = np.asarray(r["ins_cath"], float); P = np.asarray(r["proj_s"], float)
    else:
        G = np.asarray(r["gw"], float); C = np.asarray(r["cath"], float); P = np.asarray(r["proj"], float)
    n = len(G)
    if n == 0 or not np.isfinite(C).any():
        out.update(lead_final=float("nan"), lead_max=float("nan"), cath_led_success=None, cath_led_episode=None)
        return out
    pl = _f(_get(r, "path_len", "pl"))
    if not (pl > 0): pl = max(float(np.nanmax(P)), 1.0)
    L = C - G
    dP = np.diff(P, prepend=P[:1]); adv = np.clip(dP, 0, None); adv_tot = float(adv.sum())
    succ = _get(r, "success")
    if succ is not None and np.ndim(succ) > 0: succ = bool(np.asarray(succ).any())
    lead_final = float(L[-1]); lead_max = float(np.nanmax(L))

    def at(q):
        hit = np.nonzero(P >= q * pl)[0]
        return float(L[hit[0]]) if len(hit) else float("nan")

    adv10 = float((adv * (L > 10)).sum() / adv_tot) if adv_tot > 0 else float("nan")
    adv20 = float((adv * (L > lead_adv_mm)).sum() / adv_tot) if adv_tot > 0 else float("nan")
    out.update(lead_final=lead_final, lead_max=lead_max, lead_min=float(np.nanmin(L)),
               lead_frac0=float((L > 0).mean()), lead_frac10=float((L > 10).mean()), lead_frac20=float((L > 20).mean()),
               lead_frac50=float((L > 50).mean()), lead_at50=at(0.5), lead_at90=at(0.9), lead_at_progmax=float(L[int(np.argmax(P))]),
               adv_led10=adv10, adv_led20=adv20,
               cath_led_success=(bool(succ) and lead_final > lead_final_mm) if succ is not None else None,
               cath_led_episode=bool(adv20 >= adv_frac) if adv20 == adv20 else None)
    return out


# ============================================================ coil (slack)
def _slack_grade(gw: float, cath: float) -> str:
    """Slack-only severity proxy, max over the two devices (calibrated on the
    recorded episodes: P(closed loop | stored length) per step, gw 50-75 mm
    0.75, 75-100 0.99, > 100 1.0; cath 50-75 0.36, 75-100 0.82, > 100 1.0;
    <= 30 mm 0.0.  Eye-labelled single loops carry 30-140 mm, double loops
    230-260 mm, 'multiple' loops > 400 mm -> multi-loop above 200 mm)."""
    gw = gw if gw == gw else -1.0; cath = cath if cath == cath else -1.0
    if gw > MULTI_MM or cath > MULTI_MM: return "multi-loop"
    if gw > GW_LOOP_PROXY_MM or cath > CATH_LOOP_PROXY_MM: return "loop"
    if gw >= BOW_MM or cath >= BOW_MM: return "bow"
    return "none"


def coil_from_features(row, gw_mm: float = GW_COIL_MM, cath_mm: float = CATH_COIL_MM) -> Dict[str, Any]:
    """Slack-only coil / buckle indicators from a FEATURES row (log-derived
    episode; also fine for a record after ``core_features``).

    Uses ``slack_max``, ``slack_final`` (guidewire: ins_gw - proj_s of the
    frontier), ``cath_slack_max``, ``cath_slack_final`` (env5 catheter slack;
    NaN in logs without the field), ``prog_max``, ``success``.

    Returns
      coil              project rule: slack_max > 100 or cath_slack_max > 50
      coil_gw / coil_cath   which device trips the rule
      coil_loose        slack_max > 50 or cath_slack_max > 30 (recall-tuned on
                        the eye labels: 0.94 recall, 0.89 precision)
      loop_proxy        slack_max > 50 or cath_slack_max > 50 (polyline-
                        calibrated "a closed loop existed at some step";
                        differs from the project rule only for wire slack
                        50-100 mm, which is a loop 75-99 % of the time)
      coil_final        rule applied to the LAST step (what a snapshot shows)
      grade_max / grade_final   none | bow | loop | multi-loop (slack proxy)
      stored_max_mm     max(slack_max, cath_slack_max)
      near_target_buckle  failure that got to >= 75 % of the path and ends with
                        >= 20 mm stored in either device (the "buckle fight")
      near_target_buckle_grade  grade_final for those, else None
      has_cath_slack    whether the catheter slack field exists
    """
    sm, sf = _f(_get(row, "slack_max")), _f(_get(row, "slack_final"))
    cm, cf = _f(_get(row, "cath_slack_max")), _f(_get(row, "cath_slack_final"))
    has_c = cm == cm
    cmv = cm if has_c else -1.0; cfv = cf if cf == cf else -1.0
    coil_gw = bool(sm > gw_mm); coil_cath = bool(cmv > cath_mm)
    succ = _get(row, "success"); prog = _f(_get(row, "prog_max"), 0.0)
    gmax = _slack_grade(sm, cmv); gfin = _slack_grade(sf, cfv)
    ntb = (succ is not None and not bool(succ) and prog >= 0.75 and gfin != "none")
    return dict(coil=coil_gw or coil_cath, coil_gw=coil_gw, coil_cath=coil_cath,
                coil_loose=bool(sm > 50 or cmv > 30), loop_proxy=bool(sm > GW_LOOP_PROXY_MM or cmv > CATH_LOOP_PROXY_MM),
                coil_final=bool(sf > gw_mm or cfv > cath_mm), grade_max=gmax, grade_final=gfin,
                stored_max_mm=float(max(sm if sm == sm else -1.0, cmv)), near_target_buckle=bool(ntb),
                near_target_buckle_grade=gfin if ntb else None, has_cath_slack=bool(has_c))


# ========================================================= coil (polyline)
def resample_polyline(pts: np.ndarray, ds: float):
    """(K,3) -> uniform arclength spacing ds; returns (points, arclength)."""
    pts = np.asarray(pts, np.float64); pts = pts[np.isfinite(pts).all(1)]
    if len(pts) < 2: return pts, np.zeros(len(pts))
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1); s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] < 2 * ds: return pts, s
    t = np.linspace(0.0, s[-1], int(math.floor(s[-1] / ds)) + 1)
    return np.column_stack([np.interp(t, s, pts[:, d]) for d in range(3)]), t


def smooth_polyline(X: np.ndarray, ds: float, sigma: float) -> np.ndarray:
    """Gaussian smoothing along arclength (sigma in mm), reflecting about the ends."""
    if sigma <= 0 or len(X) < 5: return X
    h = int(math.ceil(3 * sigma / ds)); h = min(h, len(X) - 1)
    k = np.exp(-0.5 * (np.arange(-h, h + 1) * ds / sigma) ** 2); k /= k.sum()
    Xp = np.concatenate([2 * X[0] - X[1:h + 1][::-1], X, 2 * X[-1] - X[-h - 1:-1][::-1]])
    return np.column_stack([np.convolve(Xp[:, d], k, mode="valid") for d in range(3)])


def polyline_geometry(X: np.ndarray, ds: float = RESAMPLE_DS_MM, sigma: float = SMOOTH_SIGMA_MM,
                      prox_mm: float = LOOP_PROX_MM, sep_mm: float = LOOP_SEP_MM) -> Optional[Dict[str, Any]]:
    """Geometry of ONE device polyline (tip first).  Returns None if shorter
    than ~5 samples.  Keys: arclen, turn_deg (total turning of the smoothed
    curve), rmin_mm (min bend radius of the smoothed curve -- biased up by the
    smoothing, indicative only), closure_mm (min distance between nodes >=
    sep_mm apart in arclength), loop (closure_mm < prox_mm), n_loops
    (connected components of closure pairs in the node-index plane),
    s_loop_mm (arclength from the tip where the closest loop starts)."""
    from scipy.spatial.distance import cdist
    R, _ = resample_polyline(X, ds)
    if len(R) < 5: return None
    S = smooth_polyline(R, ds, sigma)
    d = np.diff(S, axis=0); seg = np.linalg.norm(d, axis=1); ok = seg > 1e-9
    t = d[ok] / seg[ok][:, None]
    th = np.arccos(np.clip((t[:-1] * t[1:]).sum(1), -1.0, 1.0))
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    kmax = float(th.max() / ds) if len(th) else 0.0
    D = cdist(S, S); A = np.abs(cum[:, None] - cum[None, :]); cand = A > sep_mm
    closure = float(D[cand].min()) if cand.any() else float("nan")
    loop = bool(closure == closure and closure < prox_mm)
    n_loops = 0; s_loop = float("nan")
    if loop:
        from scipy.ndimage import label
        M = np.triu((D < prox_mm) & cand, 1)
        _, n_loops = label(M, structure=np.ones((3, 3), int))
        ii, jj = np.nonzero(M); s_loop = float(cum[ii].min())
    return dict(arclen=float(cum[-1]), n_samples=int(len(S)), turn_deg=float(np.degrees(th.sum())),
                rmin_mm=float(1.0 / kmax) if kmax > 0 else float("inf"), closure_mm=closure, loop=loop,
                n_loops=int(n_loops), s_loop_mm=s_loop)


def _path_prefix(path: np.ndarray, s_end: float) -> Optional[np.ndarray]:
    if path is None or len(path) < 2 or not (s_end == s_end) or s_end <= 0: return None
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1); cum = np.concatenate([[0.0], np.cumsum(seg)])
    m = int(np.searchsorted(cum, min(s_end, cum[-1])))
    return path[:max(m + 1, 3)]


def coil_from_record(rec: Dict[str, Any], ds: float = RESAMPLE_DS_MM, sigma: float = SMOOTH_SIGMA_MM,
                     prox_mm: float = LOOP_PROX_MM, sep_mm: float = LOOP_SEP_MM, steps=None,
                     bow_mm: float = BOW_MM) -> Dict[str, Any]:
    """Polyline coil / buckle indicators for a recorded episode
    (``rec = traj_render.load_record(path)``: keys gw, cath (T,64,3), ins_gw,
    ins_cath, proj_s, cath_slack, path, path_len, success ...).

    Per step and per device the smoothed polyline is tested for a closed
    loop (self-closure) and counted (``n_loops``); the stored length is the
    slack (guidewire: ins_gw - proj_s, catheter: rec["cath_slack"]).  Grade per
    step = max over devices of
        multi-loop  n_loops >= 2
        loop        loop
        bow         no loop, stored length >= bow_mm
        none        otherwise
    ``steps``: iterable of step indices to evaluate (default: all).

    Returns
      grade_max, grade_final           episode-max and last-step grade
      loop_any_gw / loop_any_cath      a closed loop existed at some step
      loop_steps                       number of steps with a loop (either device)
      first_loop_step                  step index of the first loop, -1 if none
      slack_at_first_loop / cath_slack_at_first_loop   stored length when it closed
      n_loops_max                      max simultaneous loop count (either device)
      closure_min_gw / closure_min_cath   min self-closure distance (mm)
      turn_excess_max_deg              max over steps of (smoothed wire turning -
                                       smoothed planned-path turning up to the
                                       frontier projection); noise floor about
                                       -250..+50 deg on clean wires, +360 per loop
      path_closure_mm                  the planned path's own self-closure at
                                       sep_mm (anatomy control; > prox_mm means the
                                       loop test cannot fire on the anatomy alone)
      near_target_buckle               steps with prog >= 0.9 and stored >= bow_mm
      near_target_loop_frac            share of those steps that hold a closed loop
      per_step                         dict of arrays: grade, loop_gw, loop_cath,
                                       closure_gw, closure_cath, n_loops_gw,
                                       n_loops_cath, slack, cath_slack, prog
    """
    T = int(len(rec["proj_s"])); idx = list(range(T)) if steps is None else [int(s) for s in steps]
    G, C = np.asarray(rec["ins_gw"], float), np.asarray(rec["ins_cath"], float)
    P = np.asarray(rec["proj_s"], float); CS = np.asarray(rec.get("cath_slack", np.full(T, np.nan)), float)
    pl = _f(rec.get("path_len")); pl = pl if pl > 0 else max(float(np.nanmax(P)), 1.0)
    slack = G - P; prog = P / pl
    path = np.asarray(rec["path"], float) if rec.get("path") is not None and len(rec["path"]) else None
    n = len(idx)
    ps = dict(step=np.array(idx), grade=np.array(["none"] * n, dtype=object), loop_gw=np.zeros(n, bool), loop_cath=np.zeros(n, bool),
              closure_gw=np.full(n, np.nan), closure_cath=np.full(n, np.nan), n_loops_gw=np.zeros(n, int), n_loops_cath=np.zeros(n, int),
              turn_excess_gw=np.full(n, np.nan), slack=slack[idx], cath_slack=CS[idx], prog=prog[idx])
    for k, t in enumerate(idx):
        gg = polyline_geometry(rec["gw"][t], ds, sigma, prox_mm, sep_mm)
        cg = polyline_geometry(rec["cath"][t], ds, sigma, prox_mm, sep_mm)
        nl = 0; lp = False
        if gg:
            ps["loop_gw"][k] = gg["loop"]; ps["closure_gw"][k] = gg["closure_mm"]; ps["n_loops_gw"][k] = gg["n_loops"]
            nl = max(nl, gg["n_loops"]); lp = lp or gg["loop"]
            pp = _path_prefix(path, float(P[t]))
            if pp is not None:
                pg = polyline_geometry(pp, ds, sigma, prox_mm, sep_mm)
                if pg: ps["turn_excess_gw"][k] = gg["turn_deg"] - pg["turn_deg"]
        if cg:
            ps["loop_cath"][k] = cg["loop"]; ps["closure_cath"][k] = cg["closure_mm"]; ps["n_loops_cath"][k] = cg["n_loops"]
            nl = max(nl, cg["n_loops"]); lp = lp or cg["loop"]
        stored = max(_f(slack[t], -1.0), _f(CS[t], -1.0))
        ps["grade"][k] = "multi-loop" if nl >= 2 else "loop" if lp else "bow" if stored >= bow_mm else "none"
    order = {g: i for i, g in enumerate(GRADES)}
    gmax = GRADES[max(order[g] for g in ps["grade"])] if n else "none"
    anyloop = ps["loop_gw"] | ps["loop_cath"]
    first = int(np.nonzero(anyloop)[0][0]) if anyloop.any() else -1
    pc = float("nan")
    if path is not None:
        pg = polyline_geometry(path, ds, sigma, prox_mm, sep_mm)
        if pg: pc = pg["closure_mm"]
    nt = (ps["prog"] >= 0.9) & (np.maximum(np.nan_to_num(ps["slack"], nan=-1), np.nan_to_num(ps["cath_slack"], nan=-1)) >= bow_mm)
    return dict(grade_max=gmax, grade_final=str(ps["grade"][-1]) if n else "none",
                loop_any_gw=bool(ps["loop_gw"].any()), loop_any_cath=bool(ps["loop_cath"].any()), loop_steps=int(anyloop.sum()),
                first_loop_step=first, slack_at_first_loop=float(ps["slack"][first]) if first >= 0 else float("nan"),
                cath_slack_at_first_loop=float(ps["cath_slack"][first]) if first >= 0 else float("nan"),
                n_loops_max=int(max(ps["n_loops_gw"].max(), ps["n_loops_cath"].max())) if n else 0,
                closure_min_gw=float(np.nanmin(ps["closure_gw"])) if np.isfinite(ps["closure_gw"]).any() else float("nan"),
                closure_min_cath=float(np.nanmin(ps["closure_cath"])) if np.isfinite(ps["closure_cath"]).any() else float("nan"),
                turn_excess_max_deg=float(np.nanmax(ps["turn_excess_gw"])) if np.isfinite(ps["turn_excess_gw"]).any() else float("nan"),
                path_closure_mm=pc, near_target_buckle=int(nt.sum()),
                near_target_loop_frac=float(anyloop[nt].mean()) if nt.any() else float("nan"),
                slack_max=float(np.nanmax(slack)), cath_slack_max=float(np.nanmax(CS)) if np.isfinite(CS).any() else float("nan"),
                per_step=ps)


__all__ = ["catheter_led", "coil_from_features", "coil_from_record", "polyline_geometry", "resample_polyline", "smooth_polyline", "GRADES"]
