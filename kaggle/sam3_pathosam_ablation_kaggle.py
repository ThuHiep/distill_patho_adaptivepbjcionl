import pickle
from pathlib import Path
import numpy as np

ALPHA = 0.1
WINDOW = 300
Z90 = 1.6448536269514722  

def _find(name, roots=("/kaggle/input", "data", "work", ".")):
    for root in roots:
        base = Path(root)
        hits = list(base.rglob(name)) if base.exists() else []
        if hits:
            return hits[0]
    raise FileNotFoundError(f"{name} not found under {roots}; set the path manually.")

PAN_PKL = _find("pathosam_predictions.pkl")
NU_PKL = _find("pathosam_nuinsseg_preds.pkl")
print("PAN_PKL:", PAN_PKL, "\nNU_PKL :", NU_PKL)

def empirical_quantile(scores, alpha):
    n = len(scores)
    if n == 0:
        return float("inf")
    level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return float(np.quantile(scores, level, method="higher"))

def pb_count(scores, probs):
    return (scores[:, None] * probs).sum(axis=0)

def pb_variance(scores, probs):
    w = scores[:, None] * probs
    return (w * (1.0 - w)).sum(axis=0)

def _norm_pred(p):
    s = np.asarray(p["scores"], float)
    return {"scores": s, "probs": np.ones((len(s), 1)), "K": 1}

def load():
    with open(PAN_PKL, "rb") as f:
        dpan = pickle.load(f)
    with open(NU_PKL, "rb") as f:
        dnu = pickle.load(f)
    settings = dpan["predictions_by_setting"]
    key = "in_dist" if "in_dist" in settings else list(settings)[0]
    gtc = np.asarray(dpan["gt_counts"])
    pan = [_norm_pred(p) for p in settings[key]]
    pgt = [np.array([float(g.sum())]) for g in gtc]
    nu_preds = [_norm_pred(p) for p in dnu["preds"]]
    nu_gts = [np.array([float(np.asarray(g).sum())]) for g in dnu["gts"]]
    print("settings:", list(settings.keys()), "-> calibration uses", key)
    print(f"cal(PanNuke)={len(pan)}  test(NuInsSeg)={len(nu_preds)}")
    return pan, pgt, nu_preds, nu_gts

def nhat_sigma(p):
    n = pb_count(p["scores"], p["probs"])[0]
    sg = np.sqrt(pb_variance(p["scores"], p["probs"])[0] + 1e-6)
    return n, sg

def nonconf(p, gt, use_sigma=True):
    if len(p["scores"]) == 0:
        return float(abs(gt[0]))
    n, sg = nhat_sigma(p)
    r = abs(gt[0] - n)
    return r / sg if use_sigma else r

def interval(p, q, use_sigma=True):
    if len(p["scores"]) == 0:
        return 0.0, 0.0
    n, sg = nhat_sigma(p)
    half = q * sg if use_sigma else q
    return max(0.0, n - half), n + half

def winkler(lo, hi, y, alpha=ALPHA):
    s = hi - lo
    if y < lo:
        s += (2.0 / alpha) * (lo - y)
    elif y > hi:
        s += (2.0 / alpha) * (y - hi)
    return s

PAN_PREDS, PAN_GTS, NU_PREDS, NU_GTS = load()

def cal_scores(use_sigma):
    return np.array([nonconf(PAN_PREDS[i], PAN_GTS[i], use_sigma)
                     for i in range(len(PAN_PREDS))])

def v_naive(order):
    c, w, s = [], [], []
    for i in order:
        if len(NU_PREDS[i]["scores"]) == 0:
            lo, hi = 0.0, 0.0
        else:
            n, sg = nhat_sigma(NU_PREDS[i]); lo, hi = max(0.0, n - Z90 * sg), n + Z90 * sg
        y = NU_GTS[i][0]
        c.append(lo <= y <= hi); w.append(hi - lo); s.append(winkler(lo, hi, y))
    return c, w, s

def v_static(order, use_sigma=True):
    q = empirical_quantile(cal_scores(use_sigma), ALPHA)
    c, w, s = [], [], []
    for i in order:
        lo, hi = interval(NU_PREDS[i], q, use_sigma); y = NU_GTS[i][0]
        c.append(lo <= y <= hi); w.append(hi - lo); s.append(winkler(lo, hi, y))
    return c, w, s

def v_online_fixed(order, use_sigma=True):
    scores = list(cal_scores(use_sigma)[-WINDOW:])
    c, w, s = [], [], []
    for i in order:
        q = empirical_quantile(np.asarray(scores[-WINDOW:]), ALPHA) if scores else float("inf")
        lo, hi = interval(NU_PREDS[i], q, use_sigma); y = NU_GTS[i][0]
        c.append(lo <= y <= hi); w.append(hi - lo); s.append(winkler(lo, hi, y))
        scores.append(nonconf(NU_PREDS[i], NU_GTS[i], use_sigma)); scores = scores[-WINDOW:]
    return c, w, s

def v_full(order, use_sigma=True, target=0.9, cov_win=50, w_min=40, w_max=WINDOW):
    scores = list(cal_scores(use_sigma)[-w_max:]); eff = w_max; recent = []
    c, w, s = [], [], []
    for i in order:
        q = empirical_quantile(np.asarray(scores[-eff:]), ALPHA) if scores else float("inf")
        lo, hi = interval(NU_PREDS[i], q, use_sigma); y = NU_GTS[i][0]; cov = lo <= y <= hi
        c.append(cov); w.append(hi - lo); s.append(winkler(lo, hi, y))
        recent.append(cov); recent = recent[-cov_win:]; rc = np.mean(recent)
        if rc < target:
            eff = max(w_min, int(eff * 0.9))
        elif rc > target + 0.03:
            eff = min(w_max, int(eff * 1.05))
        scores.append(nonconf(NU_PREDS[i], NU_GTS[i], use_sigma)); scores = scores[-w_max:]
    return c, w, s

def agg(fn, seeds=5):
    cs, ws, ss = [], [], []
    for sd in range(seeds):
        order = np.random.RandomState(sd).permutation(len(NU_PREDS))
        c, w, s = fn(order)
        cs.append(np.mean(c) * 100); ws.append(np.mean(w)); ss.append(np.mean(s))
    return np.mean(cs), np.std(cs), np.mean(ws), np.std(ws), np.mean(ss), np.std(ss)

ROWS = [
    ("Naive PB (no conformal)",            lambda o: v_naive(o)),
    ("Full - PB-sigma (raw residual)",     lambda o: v_full(o, use_sigma=False)),
    ("Full - online (static split)",       lambda o: v_static(o, True)),
    ("Full - adaptive (fixed window)",     lambda o: v_online_fixed(o, True)),
    ("Full Adaptive PB-JCI Online",        lambda o: v_full(o, True)),
]

print("=" * 80)
print("ABLATION (leave-one-out) -- PathoSAM->NuInsSeg, cal=PanNuke, 5 seeds, target 90%")
print("=" * 80)
print(f"{'Variant':32s} | {'Coverage':>13s} | {'Avg. width':>15s} | {'Winkler':>15s}")
print("-" * 80)
res = {}
for name, fn in ROWS:
    cm, csd, wm, wsd, sm, ssd = agg(fn)
    res[name] = (cm, wm, sm)
    print(f"{name:32s} | {cm:5.1f}+/-{csd:4.1f}% | {wm:6.2f}+/-{wsd:5.2f} | {sm:6.2f}+/-{ssd:5.2f}")
print("-" * 80)

on = res["Full Adaptive PB-JCI Online"]
off = res["Full - PB-sigma (raw residual)"]
print("PB-sigma effect (same adaptive window, sigma ON vs OFF):")
print(f"  sigma ON  : cov {on[0]:5.1f}%  width {on[1]:7.2f}  winkler {on[2]:7.2f}")
print(f"  sigma OFF : cov {off[0]:5.1f}%  width {off[1]:7.2f}  winkler {off[2]:7.2f}")
print(f"  delta     : width {off[1]-on[1]:+7.2f}  winkler {off[2]-on[2]:+7.2f}  "
      f"(positive => sigma gives tighter/lower)")
verdict = "helps" if on[2] < off[2] else ("neutral" if abs(on[2]-off[2]) < 1e-6 else "hurts")
print(f"  => PB-sigma {verdict} on this benchmark (judge by Winkler at matched coverage).")
print("-" * 80)
print("JCI/max-statistic is degenerate here (K=1); see the paper for its multi-class ablation.")
