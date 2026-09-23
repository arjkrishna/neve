"""phase_taxonomy.py -- explicit, phase-based taxonomy of navigation episodes.

Second pass on the trajectory atlas (saved/traj): instead of 11 k-means types
fitted on a feature matrix that contains episode length and progress (which
are the outcome: every failure is a 600-step timeout), each episode is typed
by the *process* the stall detector saw:

    did the frontier ever stall?  ->  did every stall close?  ->  how many,
    how deep was the withdrawal, where was the LAST stall, and which physical
    regime (catheter shoved ahead / device coiled) was present.

Level 1 (`phase_type`) uses no step count and no progress measure.
Level 2 (`phase_type2`) appends a coarse severity from max progress.

Fields read (a features.jsonl row, dict or pandas Series):
    events (list of {k, r, close, onset, first, p0}), pl, ev_n, ev_unrec,
    ev_ret_max, cath_lead_max, slack_max, cath_slack_max, cmd_push_frac,
    prog_max (level 2 only).
Missing / NaN fields degrade gracefully (see _get).
"""
import math

# ---- thresholds (mm unless stated) -------------------------------------------
LEAD_CATHLED = 30.0      # catheter tip >30 mm ahead of the wire: catheter-led run
LEAD_SHOVE = 80.0        # catheter >80 mm ahead of the wire: shove regime
COIL_CATH_SLACK = 50.0   # project coil definition (buckle_clear_b_coil_v1):
COIL_GW_SLACK = 100.0    #   cath_slack > 50 OR guidewire slack > 100
HARD_MM = 8.0            # withdrawal inside a stall > 8 mm = "hard" (detector soft_max)
MULTI = 3                # >= 3 stall events = a fight
LOC_TARGET = 0.90        # last-stall arclength / path length bands
LOC_DISTAL = 0.66
LOC_MID = 0.33
IDLE_PUSH_FRAC = 0.10    # wire |cmd| > 10 on fewer than 10 % of steps (wire-led episodes only)
SEV_REACHED = 0.90       # level-2 severity bands on prog_max
SEV_FAR = 0.50

ORDER = ["free-run", "cath-led-run", "idle-creep", "light-recovery", "deep-recovery",
         "fight-escaped-near", "fight-escaped-far", "doorstep-stall", "shove-capped",
         "coiled-stuck", "stuck-midpath", "stuck-proximal"]
SEVERITIES = ["reached", "far", "short"]

_DEFS = {
    "free-run": "No stall event; wire-led (catheter never >30 mm ahead of the wire); the policy pushed.",
    "cath-led-run": "No stall event; the catheter ran >30 mm ahead of the wire at some point (procedural-era sprint).",
    "idle-creep": "No stall event, wire-led, and the policy hardly pushed the wire (|cmd|>10 on <10 % of steps): the detector needs a push to fire, so idling and gentle creeping are invisible to it.",
    "light-recovery": "1-2 stalls, all closed (frontier passed), max withdrawal inside a stall <= 8 mm (grind or soft).",
    "deep-recovery": "1-2 stalls, all closed, at least one withdrawal > 8 mm (hard recovery).",
    "fight-escaped-near": ">= 3 stalls, all closed; the last stall sat in the distal third or the target zone (p0/pl > 0.66).",
    "fight-escaped-far": ">= 3 stalls, all closed; the last stall sat at or before 0.66 of the path.",
    "doorstep-stall": "Last stall still open at the end, in the target zone (p0/pl > 0.9), no shove and no coil.",
    "shove-capped": "Last stall still open with the catheter shoved > 80 mm ahead of the wire (any location).",
    "coiled-stuck": "Last stall still open with a coil (cath_slack > 50 mm or wire slack > 100 mm), catheter lead <= 80 mm.",
    "stuck-midpath": "Last stall still open between 0.33 and 0.9 of the path, plain mechanism (no shove, no coil).",
    "stuck-proximal": "Last stall still open in the first third of the path, plain mechanism (no shove, no coil).",
}
_SEV_DEFS = {"reached": "prog_max >= 0.9 (frontier reached the target zone)",
             "far": "0.5 <= prog_max < 0.9", "short": "prog_max < 0.5"}


def _get(row, key, default=float("nan")):
    """Field access that works for dicts and pandas rows; None/NaN count as missing."""
    try:
        v = row[key]
    except (KeyError, IndexError, TypeError):
        return default
    if v is None:
        return default
    if isinstance(v, float) and math.isnan(v):
        return default
    return v


def _isnan(x):
    return isinstance(x, float) and math.isnan(x)


def regimes(row):
    """The axes the taxonomy is built from, as a dict (useful for tables)."""
    lead = _get(row, "cath_lead_max")
    slack = _get(row, "slack_max", 0.0)
    cslack = _get(row, "cath_slack_max")
    ev = _get(row, "events", None)
    n_ev = int(_get(row, "ev_n", len(ev) if isinstance(ev, list) else 0))
    n_open = int(_get(row, "ev_unrec", 0))
    ret_max = float(_get(row, "ev_ret_max", 0.0))
    pl = _get(row, "pl")
    last_loc = float("nan")
    if isinstance(ev, list) and ev:
        p0 = ev[-1].get("p0", float("nan"))
        if not _isnan(pl) and pl and pl > 0 and not _isnan(p0):
            last_loc = float(p0) / float(pl)
    if _isnan(last_loc) and n_ev > 0 and n_open > 0:
        # open last stall: its onset frontier equals the max progress within 1 mm
        last_loc = float(_get(row, "prog_max", float("nan")))
    shove = (not _isnan(lead)) and lead > LEAD_SHOVE
    cathled = (not _isnan(lead)) and lead > LEAD_CATHLED
    coil = ((not _isnan(cslack)) and cslack > COIL_CATH_SLACK) or (slack > COIL_GW_SLACK)
    push = _get(row, "cmd_push_frac", 1.0)
    idle = push < IDLE_PUSH_FRAC
    return dict(n_ev=n_ev, n_open=n_open, ret_max=ret_max, last_loc=last_loc, shove=shove,
                cathled=cathled, coil=coil, idle=idle)


def loc_band(x):
    if _isnan(x):
        return "na"
    if x > LOC_TARGET:
        return "target"
    if x > LOC_DISTAL:
        return "distal"
    if x > LOC_MID:
        return "mid"
    return "prox"


def phase_type(row):
    """Level-1 phase type (one of ORDER). Pure function of a features row."""
    r = regimes(row)
    if r["n_ev"] == 0:
        if r["cathled"]:                          # catheter carried the wire: a sprint, not idling
            return "cath-led-run"
        return "idle-creep" if r["idle"] else "free-run"
    if r["n_open"] == 0:                          # every stall closed
        if r["n_ev"] >= MULTI:
            loc = r["last_loc"]
            return "fight-escaped-near" if (not _isnan(loc) and loc > LOC_DISTAL) else "fight-escaped-far"
        return "deep-recovery" if r["ret_max"] > HARD_MM else "light-recovery"
    # last stall open at episode end
    if r["shove"]:
        return "shove-capped"
    if r["coil"]:
        return "coiled-stuck"
    band = loc_band(r["last_loc"])
    if band == "target":
        return "doorstep-stall"
    if band == "prox":
        return "stuck-proximal"
    return "stuck-midpath"                        # mid, distal, or unknown location


def severity(row):
    p = _get(row, "prog_max", float("nan"))
    if _isnan(p):
        return "na"
    if p >= SEV_REACHED:
        return "reached"
    if p >= SEV_FAR:
        return "far"
    return "short"


def phase_type2(row):
    """Level-2: '<phase>/<severity>'."""
    return phase_type(row) + "/" + severity(row)


def describe():
    """Category definitions, in ORDER, plus the severity bands under key '_severity'."""
    d = {k: _DEFS[k] for k in ORDER}
    d["_severity"] = dict(_SEV_DEFS)
    return d


if __name__ == "__main__":
    for k, v in describe().items():
        print("%-20s %s" % (k, v))
