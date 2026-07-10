"""
Shift-TYPE-aware router vs single mechanisms, on streams with temporal structure.

Goal: test whether a router that picks the mechanism by SHIFT TYPE beats either
single mechanism. The router does NOT cascade (flush -> then check coverage).
Instead it classifies the shift on the fly:

  * ABRUPT (clear change-point) -> Detector-flush (one-time hard reset).
  * GRADUAL (slow drift)        -> Adaptive-window (smooth shrink/grow).

Classification uses two RollingShiftDetectors: a FAST one (window=20) and a SLOW
one (window=100). On an abrupt jump the fast median rises before the slow one, so
the divergence (sig_fast - sig_slow) spikes -> abrupt. On a slow drift both rise
together -> divergence stays small -> gradual.

We evaluate on TWO purpose-built streams to look for a DOUBLE DISSOCIATION:
  Stream A (abrupt) : in-dist (calm, matches calibration) -> HARD switch to NuInsSeg.
                      Expect flush > adaptive in the transient right after the switch.
  Stream B (drift)  : in-dist -> mild -> severe -> drift (gradual ramp on PanNuke).
                      Expect adaptive > flush (flush mistimes / fires only once).

If each mechanism wins its own regime, the router earns its place. If adaptive
wins both, the honest conclusion is "adaptive-window alone suffices, no router".

CPU only, runs on cached pkls. No GPU / no Vast.
  python pathosam_router_shift.py
"""
from __future__ import annotations
import sys, pickle
from collections import defaultdict
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "kaggle" / "lib"))
from conformal import (PBAwareJointConformalOnline, RollingShiftDetector,  # noqa
                       empirical_quantile, pb_count, pb_variance)

ALPHA = 0.1
WINDOW = 300


# ---------------------------------------------------------------- load
def load():
    d = REPO / "data"
    with open(d / "pathosam_predictions.pkl", "rb") as f:
        dpan = pickle.load(f)
    with open(d / "pathosam_nuinsseg_preds.pkl", "rb") as f:
        dnu = pickle.load(f)
    settings = dpan["predictions_by_setting"]
    gtc = np.asarray(dpan["gt_counts"])
    gts = [np.array([float(g.sum())]) for g in gtc]

    def pick(*cands):
        for k in cands:
            if k in settings:
                return k
        raise KeyError(f"none of {cands} in {list(settings)}")

    keys = dict(
        in_=pick("in_dist", "in-dist", "indist"),
        mild=pick("mild", "mild_shift"),
        sev=pick("severe", "severe_shift"),
    )

    def mk(key):
        return [{"scores": np.asarray(p["scores"]),
                 "probs": np.ones((len(p["scores"]), 1)), "K": 1}
                for p in settings[key]]

    pan = {name: mk(k) for name, k in keys.items()}
    print(f"settings available: {list(settings.keys())}")
    print(f"using -> {keys}")
    return pan, gts, dnu["preds"], dnu["gts"]


def nonconf(p, gt):
    if len(p["scores"]) == 0:
        return float(abs(gt[0]))
    n = pb_count(p["scores"], p["probs"])[0]
    sg = np.sqrt(pb_variance(p["scores"], p["probs"])[0] + 1e-6)
    return abs(gt[0] - n) / sg


def interval(p, q):
    if len(p["scores"]) == 0:
        return 0.0, 0.0
    n = pb_count(p["scores"], p["probs"])[0]
    sg = np.sqrt(pb_variance(p["scores"], p["probs"])[0] + 1e-6)
    return max(0.0, n - q * sg), n + q * sg


PAN, PAN_GTS, NU_PREDS, NU_GTS = load()
PAN_IN = PAN["in_"]
# calibration = in-dist nonconformity scores (warmstart), same as the main pipeline
PAN_SCORES = np.array([nonconf(PAN_IN[i], PAN_GTS[i]) for i in range(len(PAN_IN))])
print(f"cal(in-dist) {len(PAN_SCORES)} | NuInsSeg {len(NU_PREDS)} | "
      f"q_cal={empirical_quantile(PAN_SCORES, ALPHA):.2f}\n")


# ---------------------------------------------------------------- streams
def _sample(rng, n, length):
    return rng.choice(length, n, replace=length < n)


def stream_abrupt(seed, n_pre=150, n_post=300):
    """in-dist (calm) -> HARD switch to NuInsSeg. Split post into early/late."""
    rng = np.random.RandomState(seed)
    items, labels = [], []
    for i in _sample(rng, n_pre, len(PAN_IN)):
        items.append((PAN_IN[i], PAN_GTS[i])); labels.append("pre(in-dist)")
    post = list(_sample(rng, n_post, len(NU_PREDS)))
    for j, i in enumerate(post):
        items.append((NU_PREDS[i], NU_GTS[i]))
        labels.append("post_early" if j < 50 else "post_late")
    return items, labels


def stream_drift(seed, per=150):
    """in-dist -> mild -> severe, gradual ramp on PanNuke (no drift setting in pkl)."""
    rng = np.random.RandomState(seed)
    items, labels = [], []
    for name, preds in [("in-dist", PAN["in_"]), ("mild", PAN["mild"]),
                        ("severe", PAN["sev"])]:
        for i in _sample(rng, per, len(preds)):
            items.append((preds[i], PAN_GTS[i])); labels.append(name)
    return items, labels


# ---------------------------------------------------------------- methods
# Each method returns a list of per-step records (n, sigma, q, gt) so we can
# compute BOTH raw coverage/width AND a calibrated "width needed to hit 90%".
def _ns(p):
    if len(p["scores"]) == 0:
        return 0.0, 0.0
    n = pb_count(p["scores"], p["probs"])[0]
    sg = np.sqrt(pb_variance(p["scores"], p["probs"])[0] + 1e-6)
    return n, sg


def _cov(n, sg, q, g):
    lo = max(0.0, n - q * sg); hi = n + q * sg
    return lo <= g <= hi


def base_pbo(items):
    m = PBAwareJointConformalOnline(alpha=ALPHA, window=WINDOW); m.warmstart(PAN_SCORES)
    rec = []
    for p, gt in items:
        q = m.get_quantile(); n, sg = _ns(p)
        rec.append((n, sg, q, gt[0]))
        m.update(nonconf(p, gt))
    return rec


def mech_flush(items, flush_thresh=0.5):
    m = PBAwareJointConformalOnline(alpha=ALPHA, window=WINDOW); m.warmstart(PAN_SCORES)
    det = RollingShiftDetector(window=100).fit_baseline(PAN_SCORES)
    tgt, flushed, rec = [], False, []
    for p, gt in items:
        q = m.get_quantile(); n, sg = _ns(p)
        rec.append((n, sg, q, gt[0]))
        s = nonconf(p, gt); tgt.append(s)
        if not flushed and det.step(s) >= flush_thresh:
            m.scores = list(tgt[-WINDOW:]); flushed = True
        else:
            m.update(s)
    return rec


def mech_adapt(items, target=0.9, cov_win=50, w_min=40, w_max=WINDOW):
    scores = list(PAN_SCORES[-w_max:]); eff = w_max; recent, rec = [], []
    for p, gt in items:
        q = empirical_quantile(np.asarray(scores[-eff:]), ALPHA) if scores else float("inf")
        n, sg = _ns(p); rec.append((n, sg, q, gt[0]))
        recent.append(_cov(n, sg, q, gt[0])); recent = recent[-cov_win:]
        rc = np.mean(recent)
        if rc < target:
            eff = max(w_min, int(eff * 0.9))
        elif rc > target + 0.03:
            eff = min(w_max, int(eff * 1.05))
        scores.append(nonconf(p, gt)); scores = scores[-w_max:]
    return rec


ROUTER_INFO = []


def mech_router(items, tau_div=0.15, tau_fast=0.25, tau_slow=0.15,
                target=0.9, cov_win=50, w_min=40):
    """Route by shift TYPE: fast/slow detector divergence -> abrupt (flush);
    slow detector high + coverage low -> gradual (adaptive shrink)."""
    scores = list(PAN_SCORES[-WINDOW:]); eff = WINDOW; tgt = []
    det_f = RollingShiftDetector(window=20).fit_baseline(PAN_SCORES)
    det_s = RollingShiftDetector(window=100).fit_baseline(PAN_SCORES)
    flushed, recent, rec = False, [], []
    n_abrupt = n_drift = 0
    for p, gt in items:
        q = empirical_quantile(np.asarray(scores[-eff:]), ALPHA) if scores else float("inf")
        n, sg = _ns(p); rec.append((n, sg, q, gt[0]))
        s = nonconf(p, gt); tgt.append(s)
        sf = det_f.step(s); ss = det_s.step(s); div = sf - ss
        recent.append(_cov(n, sg, q, gt[0])); recent = recent[-cov_win:]
        rc = np.mean(recent)
        if (not flushed) and div >= tau_div and sf >= tau_fast:
            scores = list(tgt[-WINDOW:]); eff = WINDOW; flushed = True; n_abrupt += 1
        elif ss >= tau_slow and rc < target:
            eff = max(w_min, int(eff * 0.9)); n_drift += 1
        elif rc > target + 0.03:
            eff = min(WINDOW, int(eff * 1.05))
        scores.append(s); scores = scores[-WINDOW:]
    ROUTER_INFO.append((n_abrupt, n_drift))
    return rec


# ---------------------------------------------------------------- metrics
def cov_width(rec, c=1.0):
    """coverage, mean width when every interval's half-width q*sigma is scaled by c."""
    cov, wid = [], []
    for n, sg, q, g in rec:
        if not np.isfinite(q):
            cov.append(True); wid.append(float("inf")); continue
        lo = max(0.0, n - c * q * sg); hi = n + c * q * sg
        cov.append(lo <= g <= hi); wid.append(hi - lo)
    return float(np.mean(cov)), float(np.mean(wid))


def width_at(rec, target=0.90):
    """Calibrated efficiency: scale all intervals by c so coverage = target,
    then return that width. c<1 => method was conservative (could shrink)."""
    if cov_width(rec, 40.0)[0] < target:          # cannot reach even at 40x
        return cov_width(rec, 40.0)[1], 40.0
    lo, hi = 0.02, 40.0
    for _ in range(45):
        mid = 0.5 * (lo + hi)
        if cov_width(rec, mid)[0] >= target:
            hi = mid
        else:
            lo = mid
    return cov_width(rec, hi)[1], hi


# ---------------------------------------------------------------- run
def run_stream(method_fn, build_stream, nseeds=5):
    raw_c, raw_w, cal_w, mult = [], [], [], []
    seg_c = defaultdict(list)
    for sd in range(nseeds):
        items, labels = build_stream(sd)
        rec = method_fn(items)
        rc, rw = cov_width(rec, 1.0)
        cw, cm = width_at(rec, 0.90)
        raw_c.append(rc); raw_w.append(rw); cal_w.append(cw); mult.append(cm)
        for lab in set(labels):
            idx = [i for i, l in enumerate(labels) if l == lab]
            seg_c[lab].append(np.mean([cov_width([rec[i]], 1.0)[0] for i in idx]))
    seg = {lab: np.mean(v) * 100 for lab, v in seg_c.items()}
    return (np.mean(raw_c) * 100, np.std(raw_c) * 100, np.mean(raw_w),
            np.mean(cal_w), np.std(cal_w), np.mean(mult), seg)


METHODS = [
    ("PB-JCI Online", base_pbo),
    ("Detector-flush", mech_flush),
    ("Adaptive-window", mech_adapt),
    ("Router (type-aware)", mech_router),
]


def report(title, build_stream, seg_order):
    print("=" * 100)
    print(title)
    print("=" * 100)
    hdr = (f"{'Method':22s} | {'raw cov':>12s} | {'raw W':>7s} | "
           f"{'W@90%':>8s} | {'c':>5s} | " + " ".join(f"{s:>16s}" for s in seg_order))
    print(hdr); print("-" * len(hdr))
    for name, fn in METHODS:
        ROUTER_INFO.clear()
        cm, cs, rw, cw, cwsd, mult, seg = run_stream(fn, build_stream)
        segstr = " ".join(f"{seg.get(s, float('nan')):15.1f}%" for s in seg_order)
        print(f"{name:22s} | {cm:5.1f}+/-{cs:3.1f}% | {rw:7.2f} | "
              f"{cw:8.2f} | {mult:5.2f} | {segstr}")
    print()


report("STREAM A - ABRUPT: in-dist(150) -> HARD switch NuInsSeg(300). target 90%",
       stream_abrupt, ["pre(in-dist)", "post_early", "post_late"])

report("STREAM B - DRIFT: in-dist -> mild -> severe (150 each). target 90%",
       stream_drift, ["in-dist", "mild", "severe"])


# ---------------------------------------------------------------- conditional W@90
def report_conditional(title, build_stream, seg_order, nseeds=5):
    """Force EACH segment to exactly 90% coverage, then read its width.
    Shows the width each method needs for 90% CONDITIONAL coverage per regime."""
    print("=" * 100)
    print("CONDITIONAL W@90% (each segment rescaled to 90% on its own) -- " + title)
    print("=" * 100)
    hdr = f"{'Method':22s} | " + " | ".join(f"{s:>20s}" for s in seg_order)
    print(hdr); print("-" * len(hdr))
    for name, fn in METHODS:
        pool = defaultdict(list)
        for sd in range(nseeds):
            items, labels = build_stream(sd)
            for r, lab in zip(fn(items), labels):
                pool[lab].append(r)
        cells = []
        for lab in seg_order:
            w, c = width_at(pool[lab], 0.90)
            cells.append(f"W={w:7.2f} (x{c:4.2f})")
        print(f"{name:22s} | " + " | ".join(f"{x:>20s}" for x in cells))
    print()


report_conditional("STREAM A", stream_abrupt,
                   ["pre(in-dist)", "post_early", "post_late"])
report_conditional("STREAM B", stream_drift,
                   ["in-dist", "mild", "severe"])


# ---------------------------------------------------------------- Winkler / interval score
def interval_score(rec, alpha=ALPHA):
    """Winkler score (lower = better): width + (2/alpha)*miss-distance.
    Single proper metric: penalises both wide intervals AND non-coverage."""
    s = []
    for n, sg, q, g in rec:
        if not np.isfinite(q):
            s.append(float("inf")); continue
        L = max(0.0, n - q * sg); U = n + q * sg
        v = (U - L)
        if g < L:
            v += (2.0 / alpha) * (L - g)
        elif g > U:
            v += (2.0 / alpha) * (g - U)
        s.append(v)
    return float(np.mean(s))


def report_winkler(title, build_stream, seg_order, nseeds=5):
    print("=" * 100)
    print("WINKLER / INTERVAL SCORE  (lower = better; penalises width AND miss) -- " + title)
    print("=" * 100)
    hdr = f"{'Method':22s} | {'overall':>9s} | " + " | ".join(f"{s:>14s}" for s in seg_order)
    print(hdr); print("-" * len(hdr))
    for name, fn in METHODS:
        ov, seg = [], defaultdict(list)
        for sd in range(nseeds):
            items, labels = build_stream(sd)
            rec = fn(items)
            ov.append(interval_score(rec))
            pool = defaultdict(list)
            for r, lab in zip(rec, labels):
                pool[lab].append(r)
            for lab in seg_order:
                seg[lab].append(interval_score(pool[lab]))
        segstr = " | ".join(f"{np.mean(seg[s]):14.2f}" for s in seg_order)
        print(f"{name:22s} | {np.mean(ov):9.2f} | {segstr}")
    print()


report_winkler("STREAM A", stream_abrupt,
               ["pre(in-dist)", "post_early", "post_late"])
report_winkler("STREAM B", stream_drift,
               ["in-dist", "mild", "severe"])

print("READ:")
print(" * raw cov/raw W = as-run. W@90% = width if each method is rescaled to EXACTLY")
print("   90% coverage -> the FAIR efficiency number (smaller = tighter at matched cov).")
print(" * c = scale needed to reach 90%. c<1 => method OVER-covers (conservative, could")
print("   be narrower); c>1 => method UNDER-covers (invalid as-run, must widen).")
print(" * Verdict: compare W@90% across methods. If Online/Adaptive/Router have similar")
print("   W@90%, the coverage gaps are just operating-point, not real efficiency gaps.")
