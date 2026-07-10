"""
Builder -> sam3_pathosam_robustness.ipynb

CPU-only Kaggle notebook bundling all PathoSAM conformal robustness experiments that
answer the reviewer's Risk-2 / Risk-3 points. Runs on cached prediction pkls (no GPU,
no model). Seconds to minutes.

  Risk 2  feedback robustness (delayed / sparse / noisy)
  Risk 3  extreme-shift mechanisms (detector-flush / adaptive-window / fallback / hybrid)
  Exp 1   detector threshold ablation (coverage, false-alarm, detection delay)
  Exp 2   detector-flush on the 4 augmentation settings (must not harm normal cases)
  Exp 4   head-to-head: PB-JCI Online vs +flush vs +adaptive-window

ATTACH a Kaggle dataset containing BOTH:
  - pathosam_predictions.pkl       (PanNuke clean-2228, in_dist/mild/severe + gt_counts)
  - pathosam_nuinsseg_preds.pkl    (NuInsSeg total-count preds)
(Create it by uploading data/pathosam_predictions.pkl + data/pathosam_nuinsseg_preds.pkl.)

Note: Exp 3 (SAM3 -> NuInsSeg / CoNSeP) needs the SAM3 pkls (phase_C_preds_seed42,
phase_E_nuinsseg_preds, consep) — attach those too if you want it (cell at the end is a stub).
"""
from __future__ import annotations
import json
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / "sam3_pathosam_robustness.ipynb"
CONFORMAL = "%%writefile conformal.py\n" + (HERE / "lib" / "conformal.py").read_text(encoding="utf-8")


def md(*lines):
    src = [l if l.endswith("\n") else l + "\n" for l in lines]
    if src:
        src[-1] = src[-1].rstrip("\n")
    return {"cell_type": "markdown", "metadata": {}, "source": src}


def code(body):
    lines = body.lstrip("\n").splitlines(keepends=True)
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": lines}


cells = []
cells.append(md(
    "# PathoSAM conformal robustness — Risk 2 / Risk 3 / validation (CPU)",
    "",
    "Answers reviewer points without any GPU — runs on cached prediction pkls.",
    "",
    "**Attach a dataset with BOTH** `pathosam_predictions.pkl` + `pathosam_nuinsseg_preds.pkl`.",
))

cells.append(md("## 00 — conformal.py (baked) + load pkls"))
cells.append(code(CONFORMAL))
cells.append(code('''
import glob, pickle, sys
import numpy as np
if "." not in sys.path: sys.path.insert(0, ".")
from conformal import (PBAwareJointConformalOnline, AdaptiveConformalInference,
                       RollingShiftDetector, empirical_quantile, pb_count, pb_variance)

def find(name):
    h = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    assert h, f"{name} not found — attach the dataset holding it."
    return h[0]

with open(find("pathosam_predictions.pkl"), "rb") as f: DP = pickle.load(f)
with open(find("pathosam_nuinsseg_preds.pkl"), "rb") as f: DN = pickle.load(f)
PBS, GT = DP["predictions_by_setting"], np.asarray(DP["gt_counts"])
N = len(GT); NU_PREDS, NU_GTS = DN["preds"], DN["gts"]
ALPHA, WINDOW, K = 0.1, 300, 5
print(f"PanNuke clean {N} | NuInsSeg {len(NU_PREDS)}")
'''))

cells.append(code('''
# ---- helpers: K=1 (total, NuInsSeg cross) and K=5 (joint, augmentation) ----
def t_nonconf(p, gt):
    if len(p["scores"])==0: return float(abs(gt[0]))
    n=pb_count(p["scores"],p["probs"])[0]; sg=np.sqrt(pb_variance(p["scores"],p["probs"])[0]+1e-6)
    return abs(gt[0]-n)/sg
def t_interval(p,q):
    if len(p["scores"])==0: return 0.0,0.0
    n=pb_count(p["scores"],p["probs"])[0]; sg=np.sqrt(pb_variance(p["scores"],p["probs"])[0]+1e-6)
    return max(0.0,n-q*sg), n+q*sg
def j_score(p,gt):
    if len(p["scores"])==0: return float(np.abs(gt).max())
    n=pb_count(p["scores"],p["probs"]); sg=np.sqrt(pb_variance(p["scores"],p["probs"])+1e-6)
    return max(abs(gt[k]-n[k])/sg[k] for k in range(K))
def j_interval(p,q):
    if len(p["scores"])==0: return np.zeros(K),np.zeros(K)
    n=pb_count(p["scores"],p["probs"]); sg=np.sqrt(pb_variance(p["scores"],p["probs"])+1e-6)
    return np.maximum(0,n-q*sg), n+q*sg

PAN_T=[{"scores":np.asarray(p["scores"]),"probs":np.ones((len(p["scores"]),1)),"K":1} for p in PBS["in_dist"]]
PAN_T_GT=[np.array([float(g.sum())]) for g in GT]
PAN_T_SCORES=np.array([t_nonconf(PAN_T[i],PAN_T_GT[i]) for i in range(N)])
def split(seed):
    idx=np.random.RandomState(seed).permutation(N); return idx[:N//2], idx[N//2:]
def warm_j(cal): return np.array([j_score(PBS["in_dist"][i],GT[i]) for i in cal])
print("helpers ready")
'''))

# ---- Risk 2 ----
cells.append(md("## Risk 2 — feedback robustness (PB-JCI Online, severe shift, K=5)"))
cells.append(code('''
def stream_eval(test, cal, feedback="full", d=0, p=1.0, sigma=0.0, seed=0):
    m=PBAwareJointConformalOnline(alpha=ALPHA,window=WINDOW); m.warmstart(warm_j(cal))
    rng=np.random.RandomState(1000+seed); pending=[]; cov,wid=[],[]
    for t,i in enumerate(test):
        pr=PBS["severe_shift"][i]; q=m.get_quantile(); lo,hi=j_interval(pr,q)
        cov.append(bool(((GT[i]>=lo)&(GT[i]<=hi)).all())); wid.append(float((hi-lo).mean()))
        g=GT[i].astype(float)
        if sigma>0: g=np.maximum(0,g+rng.normal(0,sigma,size=K))
        s=j_score(pr,g)
        if feedback=="sparse" and rng.random()>p: pass
        elif feedback=="delayed":
            pending.append((t+d,s))
            for ap,sc in [x for x in pending if x[0]<=t]: m.update(sc)
            pending=[x for x in pending if x[0]>t]
        else: m.update(s)
    return np.mean(cov)*100, np.mean(wid)
def agg2(**kw):
    cs,ws=[],[]
    for sd in range(5):
        cal,test=split(sd); c,w=stream_eval(test,cal,seed=sd,**kw); cs.append(c); ws.append(w)
    return np.mean(cs),np.std(cs),np.mean(ws)
print(f"{'Feedback':28s} | {'Coverage':>13s} | {'Width':>8s}")
print("-"*56)
for name,kw in [("Full (baseline)",dict(feedback="full")),
    ("Delayed lag=10",dict(feedback="delayed",d=10)),("Delayed lag=50",dict(feedback="delayed",d=50)),
    ("Delayed lag=100",dict(feedback="delayed",d=100)),
    ("Sparse 50%",dict(feedback="sparse",p=.5)),("Sparse 25%",dict(feedback="sparse",p=.25)),
    ("Sparse 10%",dict(feedback="sparse",p=.1)),
    ("Noisy sigma=1",dict(feedback="noisy",sigma=1.)),("Noisy sigma=2",dict(feedback="noisy",sigma=2.)),
    ("Noisy sigma=3",dict(feedback="noisy",sigma=3.))]:
    cm,cs,wm=agg2(**kw); print(f"{name:28s} | {cm:5.1f}+/-{cs:3.1f}% | {wm:7.2f}")
'''))

# ---- Risk 3 mechanisms ----
cells.append(md("## Risk 3 — extreme-shift mechanisms (PathoSAM -> NuInsSeg, K=1)"))
cells.append(code('''
def base_pbo(order):
    m=PBAwareJointConformalOnline(alpha=ALPHA,window=WINDOW); m.warmstart(PAN_T_SCORES); c,w=[],[]
    for i in order:
        q=m.get_quantile(); lo,hi=t_interval(NU_PREDS[i],q); c.append(lo<=NU_GTS[i][0]<=hi); w.append(hi-lo)
        m.update(t_nonconf(NU_PREDS[i],NU_GTS[i]))
    return c,w
def base_aci(order):
    m=AdaptiveConformalInference(alpha_target=ALPHA,gamma=0.05); m.reset(); m.history_scores=list(PAN_T_SCORES); c,w=[],[]
    for i in order:
        q=m.get_quantile(); lo,hi=t_interval(NU_PREDS[i],q); cov=lo<=NU_GTS[i][0]<=hi; c.append(cov); w.append(hi-lo)
        m.update(t_nonconf(NU_PREDS[i],NU_GTS[i]),cov)
    return c,w
def mech_flush(order,thresh=0.5):
    m=PBAwareJointConformalOnline(alpha=ALPHA,window=WINDOW); m.warmstart(PAN_T_SCORES)
    det=RollingShiftDetector(window=100).fit_baseline(PAN_T_SCORES); tgt=[]; flushed=False; c,w=[],[]
    for i in order:
        q=m.get_quantile(); lo,hi=t_interval(NU_PREDS[i],q); c.append(lo<=NU_GTS[i][0]<=hi); w.append(hi-lo)
        s=t_nonconf(NU_PREDS[i],NU_GTS[i]); tgt.append(s)
        if not flushed and det.step(s)>=thresh: m.scores=list(tgt[-WINDOW:]); flushed=True
        else: m.update(s)
    return c,w
def mech_aw(order,target=0.9,cov_win=50,w_min=40):
    scores=list(PAN_T_SCORES[-WINDOW:]); eff=WINDOW; rec=[]; c,w=[],[]
    for i in order:
        q=empirical_quantile(np.asarray(scores[-eff:]),ALPHA) if scores else float("inf")
        lo,hi=t_interval(NU_PREDS[i],q); cov=lo<=NU_GTS[i][0]<=hi; c.append(cov); w.append(hi-lo)
        rec.append(cov); rec=rec[-cov_win:]; rc=np.mean(rec)
        if rc<target: eff=max(w_min,int(eff*0.9))
        elif rc>target+0.03: eff=min(WINDOW,int(eff*1.05))
        scores.append(t_nonconf(NU_PREDS[i],NU_GTS[i])); scores=scores[-WINDOW:]
    return c,w
def mech_fb(order,target=0.9,eta=0.03,cov_win=50):
    m=PBAwareJointConformalOnline(alpha=ALPHA,window=WINDOW); m.warmstart(PAN_T_SCORES); mult=1.0; rec=[]; c,w=[],[]
    for i in order:
        q=m.get_quantile()*mult; lo,hi=t_interval(NU_PREDS[i],q); cov=lo<=NU_GTS[i][0]<=hi; c.append(cov); w.append(hi-lo)
        rec.append(cov); rec=rec[-cov_win:]; rc=np.mean(rec)
        if rc<target: mult*=(1+eta)
        elif rc>target+0.03: mult=max(1.0,mult*(1-eta))
        mult=min(mult,6.0); m.update(t_nonconf(NU_PREDS[i],NU_GTS[i]))
    return c,w
def mech_hyb(order):
    pb=PBAwareJointConformalOnline(alpha=ALPHA,window=WINDOW); pb.warmstart(PAN_T_SCORES)
    aci=AdaptiveConformalInference(alpha_target=ALPHA,gamma=0.05); aci.reset(); aci.history_scores=list(PAN_T_SCORES); c,w=[],[]
    for i in order:
        l1,h1=t_interval(NU_PREDS[i],pb.get_quantile()); l2,h2=t_interval(NU_PREDS[i],aci.get_quantile())
        lo,hi=min(l1,l2),max(h1,h2); c.append(lo<=NU_GTS[i][0]<=hi); w.append(hi-lo)
        s=t_nonconf(NU_PREDS[i],NU_GTS[i]); pb.update(s); aci.update(s,l2<=NU_GTS[i][0]<=h2)
    return c,w
def run(fn,seeds=5):
    cs,ws=[],[]
    for sd in range(seeds):
        o=np.random.RandomState(sd).permutation(len(NU_PREDS)); c,w=fn(o); cs.append(np.mean(c)); ws.append(np.mean(w))
    return np.mean(cs)*100,np.std(cs)*100,np.mean(ws)
print(f"{'Method':26s} | {'Coverage':>13s} | {'Width':>8s}")
print("-"*54)
for nm,fn in [("Baseline PB-JCI Online",base_pbo),("Baseline ACI",base_aci),
    ("A. Detector-flush",mech_flush),("B. Adaptive-window",mech_aw),
    ("C. Fallback-multiplier",mech_fb),("D. Hybrid max(PB,ACI)",mech_hyb)]:
    cm,cs,wm=run(fn); print(f"{nm:26s} | {cm:5.1f}+/-{cs:3.1f}% | {wm:7.2f}")
'''))

# ---- Exp 1 ----
cells.append(md("## Exp 1 — Detector threshold ablation (coverage / false-alarm / delay)"))
cells.append(code('''
def flush_first(order,thresh):
    m=PBAwareJointConformalOnline(alpha=ALPHA,window=WINDOW); m.warmstart(PAN_T_SCORES)
    det=RollingShiftDetector(window=100).fit_baseline(PAN_T_SCORES); tgt=[]; flushed=False; first=-1; c,w=[],[]
    for t,i in enumerate(order):
        q=m.get_quantile(); lo,hi=t_interval(NU_PREDS[i],q); c.append(lo<=NU_GTS[i][0]<=hi); w.append(hi-lo)
        s=t_nonconf(NU_PREDS[i],NU_GTS[i]); tgt.append(s)
        if not flushed and det.step(s)>=thresh: m.scores=list(tgt[-WINDOW:]); flushed=True; first=t
        else: m.update(s)
    return np.mean(c)*100,np.mean(w),first
def false_alarm(thresh,seeds=5):
    rs=[]
    for sd in range(seeds):
        idx=np.random.RandomState(sd).permutation(N); cal,test=idx[:N//2],idx[N//2:]
        det=RollingShiftDetector(window=100).fit_baseline(PAN_T_SCORES[cal]); fires=0
        for i in test:
            if det.step(PAN_T_SCORES[i])>=thresh: fires+=1
        rs.append(fires/len(test))
    return np.mean(rs)*100
print(f"{'thresh':>6s} | {'Cov':>8s} | {'Width':>7s} | {'flush@pos':>9s} | {'falseAlarm':>10s}")
print("-"*54)
for th in [0.2,0.35,0.5,0.7,1.0]:
    cs,ws,fs=[],[],[]
    for sd in range(5):
        o=np.random.RandomState(sd).permutation(len(NU_PREDS)); cm,wm,fst=flush_first(o,th)
        cs.append(cm); ws.append(wm); fs.append(fst if fst>=0 else np.nan)
    fp=np.nanmean(fs); fp=f"{fp:.0f}" if not np.isnan(fp) else "never"
    print(f"{th:6.2f} | {np.mean(cs):6.1f}% | {np.mean(ws):7.2f} | {fp:>9s} | {false_alarm(th):9.1f}%")
'''))

# ---- Exp 2 ----
cells.append(md("## Exp 2 — Detector-flush on 4 augmentation settings (must NOT harm)"))
cells.append(code('''
def make_stream(setting,test):
    if setting!="drift": return [(PBS[setting][i],GT[i]) for i in test]
    t=len(test)//3
    return ([(PBS["in_dist"][i],GT[i]) for i in test[:t]]+
            [(PBS["mild_shift"][i],GT[i]) for i in test[t:2*t]]+
            [(PBS["severe_shift"][i],GT[i]) for i in test[2*t:]])
def eval_j(stream,cal,variant,thresh=0.5):
    m=PBAwareJointConformalOnline(alpha=ALPHA,window=WINDOW); m.warmstart(warm_j(cal))
    det=RollingShiftDetector(window=100).fit_baseline(warm_j(cal)); tgt=[]; flushed=False
    eff=WINDOW; rec=[]; aw=list(warm_j(cal)[-WINDOW:]); c,w=[],[]
    for pr,gt in stream:
        q=(empirical_quantile(np.asarray(aw[-eff:]),ALPHA) if variant=="aw" and aw else m.get_quantile())
        lo,hi=j_interval(pr,q); cov=bool(((gt>=lo)&(gt<=hi)).all()); c.append(cov); w.append(float((hi-lo).mean()))
        s=j_score(pr,gt)
        if variant=="base": m.update(s)
        elif variant=="flush":
            tgt.append(s)
            if not flushed and det.step(s)>=thresh: m.scores=list(tgt[-WINDOW:]); flushed=True
            else: m.update(s)
        elif variant=="aw":
            rec.append(cov); rec=rec[-50:]; rc=np.mean(rec)
            if rc<0.9: eff=max(40,int(eff*0.9))
            elif rc>0.93: eff=min(WINDOW,int(eff*1.05))
            aw.append(s); aw=aw[-WINDOW:]
    return np.mean(c)*100,np.std([np.mean(c)]),np.mean(w)
def agg_j(setting,variant,seeds=5):
    cs,ws=[],[]
    for sd in range(seeds):
        cal,test=split(sd); cm,_,wm=eval_j(make_stream(setting,test),cal,variant); cs.append(cm); ws.append(wm)
    return np.mean(cs),np.std(cs),np.mean(ws)
print(f"{'Setting':12s} | {'PB-JCI Online':>20s} | {'+Detector-flush':>20s}")
print("-"*60)
for s in ["in_dist","mild_shift","severe_shift","drift"]:
    b=agg_j(s,"base"); f=agg_j(s,"flush")
    print(f"{s:12s} | {b[0]:5.1f}+/-{b[1]:3.1f}% w{b[2]:6.2f} | {f[0]:5.1f}+/-{f[1]:3.1f}% w{f[2]:6.2f}")
'''))

# ---- Exp 4 ----
cells.append(md("## Exp 4 — Head-to-head: PB-JCI Online vs +flush vs +adaptive-window"))
cells.append(code('''
def nu_variant(variant,thresh=0.5,seeds=5):
    cs,ws=[],[]
    for sd in range(seeds):
        o=np.random.RandomState(sd).permutation(len(NU_PREDS))
        if variant=="flush": cm,wm,_=flush_first(o,thresh); cs.append(cm); ws.append(wm); continue
        if variant=="aw":
            c,w=mech_aw(o); cs.append(np.mean(c)*100); ws.append(np.mean(w)); continue
        c,w=base_pbo(o); cs.append(np.mean(c)*100); ws.append(np.mean(w))
    return np.mean(cs),np.mean(ws)
print(f"{'Scenario':22s} | {'PB-JCI Online':>15s} | {'+Detector-flush':>15s} | {'+Adaptive-win':>15s}")
print("-"*78)
# NuInsSeg cross
out={v:nu_variant(v) for v in ["base","flush","aw"]}
print(f"{'NuInsSeg cross (K=1)':22s} | {out['base'][0]:5.1f}% w{out['base'][1]:6.2f} | "
      f"{out['flush'][0]:5.1f}% w{out['flush'][1]:6.2f} | {out['aw'][0]:5.1f}% w{out['aw'][1]:6.2f}")
# augmentation severe + drift
for s in ["severe_shift","drift"]:
    b=agg_j(s,"base"); f=agg_j(s,"flush"); a=agg_j(s,"aw")
    print(f"{('Augment '+s+' (K=5)'):22s} | {b[0]:5.1f}% w{b[2]:6.2f} | {f[0]:5.1f}% w{f[2]:6.2f} | {a[0]:5.1f}% w{a[2]:6.2f}")
'''))

cells.append(md(
    "## Exp 3 + SAM3 — mechanism is BACKBONE-AGNOSTIC (needs SAM3 pkls)",
    "",
    "Attach: `phase-c-predictions` (SAM3 PanNuke), `phase-e-nuinsseg-preds`, `consep-preds`.",
    "Goal: SAM3 cross-dataset shift is MILD → Detector-flush must **not over-widen** (stays",
    "inert when no shift), and the feedback robustness must replicate on SAM3 too.",
))
cells.append(code('''
def gload(name):
    h = glob.glob(f"/kaggle/input/**/{name}", recursive=True); return h[0] if h else None
PC = gload("phase_C_predictions.pkl"); PE = gload("phase_E_nuinsseg_preds.pkl"); CS = gload("consep_preds.pkl")
print("phase_C:", PC); print("phase_E nuinsseg:", PE); print("consep:", CS)

def norm_t(preds, gts):
    P = []
    for p in preds:
        s = np.asarray(p["scores"] if isinstance(p, dict) else p)
        P.append({"scores": s, "probs": np.ones((len(s), 1)), "K": 1})
    G = [float(g[0]) if hasattr(g, "__len__") else float(g) for g in gts]
    return P, G

def cross_compare(cal_scores, tp, tg, seeds=5, thresh=0.5):
    def nc(p, gt):
        if len(p["scores"]) == 0: return float(abs(gt))
        n = pb_count(p["scores"], p["probs"])[0]; sg = np.sqrt(pb_variance(p["scores"], p["probs"])[0]+1e-6); return abs(gt-n)/sg
    def iv(p, q):
        if len(p["scores"]) == 0: return 0., 0.
        n = pb_count(p["scores"], p["probs"])[0]; sg = np.sqrt(pb_variance(p["scores"], p["probs"])[0]+1e-6); return max(0., n-q*sg), n+q*sg
    out = {}
    for variant in ["base", "aci", "flush", "aw"]:
        cc, ww = [], []
        for sd in range(seeds):
            order = np.random.RandomState(sd).permutation(len(tp))
            if variant == "aci":
                m = AdaptiveConformalInference(alpha_target=ALPHA, gamma=0.05); m.reset(); m.history_scores = list(cal_scores)
            else:
                m = PBAwareJointConformalOnline(alpha=ALPHA, window=WINDOW); m.warmstart(cal_scores)
            det = RollingShiftDetector(window=100).fit_baseline(cal_scores)
            tgt, flushed, eff, rec, aw = [], False, WINDOW, [], list(cal_scores[-WINDOW:])
            c, w = [], []
            for i in order:
                q = (empirical_quantile(np.asarray(aw[-eff:]), ALPHA) if variant == "aw" and aw else m.get_quantile())
                lo, hi = iv(tp[i], q); cov = lo <= tg[i] <= hi; c.append(cov); w.append(hi-lo); s = nc(tp[i], tg[i])
                if variant == "base": m.update(s)
                elif variant == "aci": m.update(s, cov)
                elif variant == "flush":
                    tgt.append(s)
                    if not flushed and det.step(s) >= thresh: m.scores = list(tgt[-WINDOW:]); flushed = True
                    else: m.update(s)
                elif variant == "aw":
                    rec.append(cov); rec = rec[-50:]; rc = np.mean(rec)
                    if rc < 0.9: eff = max(40, int(eff*0.9))
                    elif rc > 0.93: eff = min(WINDOW, int(eff*1.05))
                    aw.append(s); aw = aw[-WINDOW:]
            cc.append(np.mean(c)*100); ww.append(np.mean(w))
        out[variant] = (np.mean(cc), np.mean(ww))
    return out

if PC and (PE or CS):
    dpc = pickle.load(open(PC, "rb"))
    PBS_S, GT_S = dpc["predictions_by_setting"], np.asarray(dpc["gt_counts"])
    pan_t = [{"scores": np.asarray(p["scores"]), "probs": np.ones((len(p["scores"]), 1)), "K": 1} for p in PBS_S["in_dist"]]
    pan_g = [float(g.sum()) for g in GT_S]
    def t_nc1(p, gt):
        if len(p["scores"]) == 0: return float(abs(gt))
        n = pb_count(p["scores"], p["probs"])[0]; sg = np.sqrt(pb_variance(p["scores"], p["probs"])[0]+1e-6); return abs(gt-n)/sg
    cal_S = np.array([t_nc1(pan_t[i], pan_g[i]) for i in range(len(pan_t))])
    print(f"\\nSAM3 PanNuke cal scores: {len(cal_S)} | q={empirical_quantile(cal_S, ALPHA):.2f}")
    print("\\n" + "="*86)
    print("EXP 3 — SAM3 cross-dataset + mechanisms (MILD shift: flush must NOT over-widen)")
    print("="*86)
    print(f"{'Target':16s} | {'PB-JCI Online':>15s} | {'ACI':>15s} | {'+flush':>15s} | {'+adapt-win':>15s}")
    print("-"*86)
    for label, path in [("SAM3->NuInsSeg", PE), ("SAM3->CoNSeP", CS)]:
        if not path: continue
        d = pickle.load(open(path, "rb")); tp, tg = norm_t(d["preds"], d["gts"])
        r = cross_compare(cal_S, tp, tg)
        print(f"{label:16s} | {r['base'][0]:5.1f}% w{r['base'][1]:6.2f} | {r['aci'][0]:5.1f}% w{r['aci'][1]:6.2f} | "
              f"{r['flush'][0]:5.1f}% w{r['flush'][1]:6.2f} | {r['aw'][0]:5.1f}% w{r['aw'][1]:6.2f}")
    print("-"*86)
    print("Expect: SAM3 shift mild -> base already ~89-90%; flush ~= base (detector stays inert), NO over-width.")
else:
    print("SAM3 pkls not all attached -> skipping Exp 3. Attach phase-c-predictions + phase-e-nuinsseg-preds + consep-preds.")
'''))
cells.append(md("### SAM3 Risk 2 — feedback robustness replicates on SAM3 (severe augmentation)"))
cells.append(code('''
if PC:
    NS = len(GT_S)
    def js_s(p, gt):
        if len(p["scores"]) == 0: return float(np.abs(gt).max())
        n = pb_count(p["scores"], p["probs"]); sg = np.sqrt(pb_variance(p["scores"], p["probs"])+1e-6); return max(abs(gt[k]-n[k])/sg[k] for k in range(K))
    def ji_s(p, q):
        if len(p["scores"]) == 0: return np.zeros(K), np.zeros(K)
        n = pb_count(p["scores"], p["probs"]); sg = np.sqrt(pb_variance(p["scores"], p["probs"])+1e-6); return np.maximum(0, n-q*sg), n+q*sg
    def sp(seed): idx = np.random.RandomState(seed).permutation(NS); return idx[:NS//2], idx[NS//2:]
    def wj(cal): return np.array([js_s(PBS_S["in_dist"][i], GT_S[i]) for i in cal])
    def se(test, cal, feedback="full", d=0, p=1.0, sigma=0.0, seed=0):
        m = PBAwareJointConformalOnline(alpha=ALPHA, window=WINDOW); m.warmstart(wj(cal))
        rng = np.random.RandomState(1000+seed); pend = []; cov, wid = [], []
        for t, i in enumerate(test):
            pr = PBS_S["severe_shift"][i]; q = m.get_quantile(); lo, hi = ji_s(pr, q)
            cov.append(bool(((GT_S[i] >= lo) & (GT_S[i] <= hi)).all())); wid.append(float((hi-lo).mean()))
            g = GT_S[i].astype(float)
            if sigma > 0: g = np.maximum(0, g+rng.normal(0, sigma, size=K))
            s = js_s(pr, g)
            if feedback == "sparse" and rng.random() > p: pass
            elif feedback == "delayed":
                pend.append((t+d, s))
                for ap, sc in [x for x in pend if x[0] <= t]: m.update(sc)
                pend = [x for x in pend if x[0] > t]
            else: m.update(s)
        return np.mean(cov)*100, np.mean(wid)
    def a2(**kw):
        cs, ws = [], []
        for sd in range(5):
            cal, test = sp(sd); c, w = se(test, cal, seed=sd, **kw); cs.append(c); ws.append(w)
        return np.mean(cs), np.std(cs), np.mean(ws)
    print(f"{'Feedback (SAM3)':24s} | {'Coverage':>13s} | {'Width':>8s}")
    print("-"*50)
    for name, kw in [("Full",dict(feedback="full")),("Delayed lag=50",dict(feedback="delayed",d=50)),
        ("Sparse 50%",dict(feedback="sparse",p=.5)),("Sparse 25%",dict(feedback="sparse",p=.25)),
        ("Noisy sigma=2",dict(feedback="noisy",sigma=2.))]:
        cm, cs, wm = a2(**kw); print(f"{name:24s} | {cm:5.1f}+/-{cs:3.1f}% | {wm:7.2f}")
else:
    print("phase-c-predictions not attached -> skip SAM3 feedback.")
'''))

# ---- Priority 2: modern baselines + Priority 3: ablations ----
cells.append(md("## Bảng 9 — Modern baselines (Weighted/NexCP/Naive) + ablations"))
cells.append(code('''
from sklearn.linear_model import LogisticRegression
Z = 1.645
def feats(p):
    s = np.asarray(p["scores"])
    return [0,0,0,0] if len(s)==0 else [float(s.sum()),float(len(s)),float(s.mean()),float(s.std())]
def wquantile(scores, weights, level):
    o=np.argsort(scores); s=np.asarray(scores)[o]; w=np.asarray(weights)[o]; cw=np.cumsum(w)/w.sum()
    return s[min(np.searchsorted(cw,level),len(s)-1)]
def run_cross(method, seeds=5):
    cs,ws=[],[]
    for sd in range(seeds):
        o=np.random.RandomState(sd).permutation(len(NU_PREDS)); c,w=method(o); cs.append(np.mean(c)*100); ws.append(np.mean(w))
    return np.mean(cs),np.std(cs),np.mean(ws)
def b_pbo(order):
    m=PBAwareJointConformalOnline(alpha=ALPHA,window=WINDOW); m.warmstart(PAN_T_SCORES); c,w=[],[]
    for i in order:
        q=m.get_quantile(); lo,hi=t_interval(NU_PREDS[i],q); c.append(lo<=NU_GTS[i][0]<=hi); w.append(hi-lo); m.update(t_nonconf(NU_PREDS[i],NU_GTS[i]))
    return c,w
def b_naive(order):
    c,w=[],[]
    for i in order:
        p=NU_PREDS[i]
        if len(p["scores"])==0: lo,hi=0,0
        else:
            n=pb_count(p["scores"],p["probs"])[0]; sg=np.sqrt(pb_variance(p["scores"],p["probs"])[0]+1e-6); lo,hi=max(0,n-Z*sg),n+Z*sg
        c.append(lo<=NU_GTS[i][0]<=hi); w.append(hi-lo)
    return c,w
def b_weighted(order):
    Xs=np.array([feats({"scores":np.asarray(p["scores"])}) for p in PBS["in_dist"]])
    Xt=np.array([feats(NU_PREDS[i]) for i in range(len(NU_PREDS))])
    clf=LogisticRegression(max_iter=1000).fit(np.vstack([Xs,Xt]), np.r_[np.zeros(len(Xs)),np.ones(len(Xt))])
    pt=clf.predict_proba(Xs)[:,1]; wts=pt/np.clip(1-pt,1e-6,1)
    q=wquantile(PAN_T_SCORES,wts,1-ALPHA); c,w=[],[]
    for i in order:
        lo,hi=t_interval(NU_PREDS[i],q); c.append(lo<=NU_GTS[i][0]<=hi); w.append(hi-lo)
    return c,w
def b_nexcp(order, rho=0.99):
    hist=list(PAN_T_SCORES); c,w=[],[]
    for i in order:
        wts=np.array([rho**(len(hist)-1-k) for k in range(len(hist))]); q=wquantile(hist,wts,1-ALPHA)
        lo,hi=t_interval(NU_PREDS[i],q); c.append(lo<=NU_GTS[i][0]<=hi); w.append(hi-lo); hist.append(t_nonconf(NU_PREDS[i],NU_GTS[i]))
    return c,w
# COP — Conformal Optimistic Prediction (ICLR 2026, arXiv:2512.07770). Same score as PB-JCI Online;
#   q_hat += eta*(1[s>q]-alpha) ; F_hat = windowed empirical CDF(w=100) ; q = q_hat - lam*(F_hat(q_hat)-(1-alpha))
class COP:
    def __init__(self, eta, lam=1.0, w=100, warm=None):
        self.eta,self.lam,self.w=eta,lam,w
        self.qhat=empirical_quantile(np.asarray(warm),ALPHA); self.q=self.qhat; self.win=list(np.asarray(warm)[-w:])
    def get_q(self): return max(0.0,self.q)
    def update(self,s,q_used):
        self.qhat=max(0.0,self.qhat+self.eta*((1.0 if s>q_used else 0.0)-ALPHA))
        self.win.append(float(s)); self.win=self.win[-self.w:]
        Fhat=float(np.mean(np.asarray(self.win)<=self.qhat)); self.q=max(0.0,self.qhat-self.lam*(Fhat-(1-ALPHA)))
def b_cop(order, eta=5.0):
    m=COP(eta,warm=PAN_T_SCORES); c,w=[],[]
    for i in order:
        q=m.get_q(); lo,hi=t_interval(NU_PREDS[i],q); c.append(lo<=NU_GTS[i][0]<=hi); w.append(hi-lo); m.update(t_nonconf(NU_PREDS[i],NU_GTS[i]),q)
    return c,w
print("Bảng 9a — baselines (PathoSAM->NuInsSeg):")
print(f"{'Method':34s} | {'Coverage':>13s} | {'Width':>7s}")
for nm,fn in [("Naive PB (no conformal)",b_naive),("Weighted Conformal (covar-shift)",b_weighted),
              ("NexCP decayed-online (Barber23)",b_nexcp),("COP best-eta (ICLR2026)",b_cop),("PB-JCI Online (ours)",b_pbo)]:
    cm,cs,wm=run_cross(fn); print(f"{nm:34s} | {cm:5.1f}+/-{cs:3.1f}% | {wm:6.2f}")
print("  COP eta-sweep (best shot for the 2026 SOTA baseline):")
for eta in [0.1,0.5,1.0,2.0,5.0]:
    cm,cs,wm=run_cross(lambda o,eta=eta: b_cop(o,eta)); print(f"    eta={eta:4.1f}: {cm:5.1f}+/-{cs:3.1f}% | w {wm:6.2f}")
print("\\nBảng 9c — window size sensitivity (PB-JCI Online):")
for Wn in [100,200,300,500]:
    def mw(order, Wn=Wn):
        m=PBAwareJointConformalOnline(alpha=ALPHA,window=Wn); m.warmstart(PAN_T_SCORES); c,w=[],[]
        for i in order:
            q=m.get_quantile(); lo,hi=t_interval(NU_PREDS[i],q); c.append(lo<=NU_GTS[i][0]<=hi); w.append(hi-lo); m.update(t_nonconf(NU_PREDS[i],NU_GTS[i]))
        return c,w
    cm,cs,wm=run_cross(mw); print(f"  window={Wn:4d}: {cm:5.1f}+/-{cs:3.1f}% | w {wm:6.2f}")
'''))
cells.append(code('''
# Bảng 9b — PB-sigma ablation (joint K=5, augmentation)
def js_ab(p,gt,use):
    if len(p["scores"])==0: return float(np.abs(gt).max())
    n=pb_count(p["scores"],p["probs"])
    if use: sg=np.sqrt(pb_variance(p["scores"],p["probs"])+1e-6); return max(abs(gt[k]-n[k])/sg[k] for k in range(K))
    return max(abs(gt[k]-n[k]) for k in range(K))
def ji_ab(p,q,use):
    if len(p["scores"])==0: return np.zeros(K),np.zeros(K)
    n=pb_count(p["scores"],p["probs"])
    if use: sg=np.sqrt(pb_variance(p["scores"],p["probs"])+1e-6); return np.maximum(0,n-q*sg),n+q*sg
    return np.maximum(0,n-q),n+q
def w_ab(cal,use): return np.array([js_ab(PBS["in_dist"][i],GT[i],use) for i in cal])
def ev_ab(setting,use,seeds=5):
    cs,ws=[],[]
    for sd in range(seeds):
        cal,test=split(sd); m=PBAwareJointConformalOnline(alpha=ALPHA,window=WINDOW); m.warmstart(w_ab(cal,use)); c,w=[],[]
        for i in test:
            pr=PBS[setting][i]; q=m.get_quantile(); lo,hi=ji_ab(pr,q,use)
            c.append(bool(((GT[i]>=lo)&(GT[i]<=hi)).all())); w.append(float((hi-lo).mean())); m.update(js_ab(pr,GT[i],use))
        cs.append(np.mean(c)*100); ws.append(np.mean(w))
    return np.mean(cs),np.std(cs),np.mean(ws)
print("Bảng 9b — ablation PB-sigma vs raw error (PB-JCI Online):")
print(f"{'Setting':12s} | {'WITH PB-sigma':>20s} | {'WITHOUT (raw)':>20s}")
for s in ["in_dist","mild_shift","severe_shift"]:
    a=ev_ab(s,True); b=ev_ab(s,False)
    print(f"{s:12s} | {a[0]:5.1f}+/-{a[1]:3.1f}% w{a[2]:6.2f} | {b[0]:5.1f}+/-{b[1]:3.1f}% w{b[2]:6.2f}")
'''))

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.10"}},
      "nbformat": 4, "nbformat_minor": 5}
OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Wrote {OUT.name}: {len(cells)} cells")
