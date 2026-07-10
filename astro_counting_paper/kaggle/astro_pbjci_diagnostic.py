# =====================================================================================
# astro_pbjci_diagnostic.py  —  DIAGNOSTIC-FIRST cho domain THIEN VAN (dem sao / thien ha)
# -------------------------------------------------------------------------------------
# MUC TIEU: kiem tra domain thien van co thoa PROFILE THANG cua Bai 1 KHONG, TRUOC khi dot
# compute chay full crosstable (bai hoc tu trees: chay full roi moi biet hong).
# Do 3 dieu kien (theo mechanism_win_criteria):
#   (1) BIAS thap  : mean(gt - softcount) per-class ~ 0  (khong under/over-dem he thong)
#   (2) SIGMA-GAIN : rho(sigma,|err|) - rho(sqrt(count),|err|) > 0  (PB-sigma tien doan sai
#                    so VUOT count -> khong thua nhu trees)
#   (3) SHIFT      : static-CP (q co dinh tu CAL) under-cover ro tren TEST (co dat recovery)
# PASS ca 3 -> chay crosstable joint K=2 (Static/ACI/Rolling-Origin/PB-Fixed/Adaptive).
#
# DATA schema — moi "field" (mot vung troi):
#   dict(mag[N], is_star[N] bool, p_detect[N], p_star[N])   (candidate = nguon that)
#   CAL = survey SAU (mlim lon, residual nho) -> TEST = survey NONG (mlim nho, residual lon)
#        = HARDENING dung chieu (song anh PanNuke->NuInsSeg).
# USE_SDSS=True : SEMI-SYNTHETIC co mo neo THAT (mo ta trung thuc, KHONG "fully real").
#   THAT   : SDSS DR17 PhotoObj — vi tri, magnitude, flux, SNR, phan loai probPSF (pipeline).
#   MODELED: p_detect = completeness 5-sigma tren SNR THAT; shift = INDUCED depth (SNR/DEPTH_FACTOR).
#   Frame khi viet: "completeness model on real SNR + controlled/induced depth shift".
# USE_SDSS=False: MO PHONG (test harness offline, khong backbone). KHONG rig de pass.
# =====================================================================================
import numpy as np

ALPHA = 0.1
COVERAGE = 1 - ALPHA
USE_SDSS = False           # True: dung SDSS that (can internet + astroquery)
N_FIELDS = 400
RNG_SEED = 0

# ================================================================== DATASET + BACKBONE
# RAW field = dict(mag, is_star, snr_deep, probPSF) — dai luong DEEP (goc). observe(depth) tao
# quan sat o do sau bat ky: DEEP (CAL) va SHALLOW=deep/depth (TEST) -> cho phep sweep do sau.
# BACKBONE (USE_SDSS) = SDSS Photo pipeline: probPSF (P sao) + psfFlux/ivar (SNR that).
# SHIFT = forward-model do sau: SNR_shallow = SNR_deep / DEPTH_FACTOR (semi-real chuan nganh).
SDSS_REGION = dict(ra0=150.0, ra1=152.0, dec0=0.0, dec1=2.0)  # ~4 deg^2 trong footprint SDSS
DEPTH_FACTOR = 3.0            # SHALLOW: SNR = deep SNR / DEPTH_FACTOR (nong hon ~1.2 mag)
SNR0, SNRW = 5.0, 1.5        # completeness 5-sigma: p_detect = logistic((SNR - SNR0)/SNRW)
SIM_SNR_ZP = 23.5            # sim: SNR_deep = 10^(0.4*(ZP - mag))

def _sep(snr):  return 1.0 / (1.0 + np.exp(-(np.asarray(snr, float) - 3.0)))
def _detect_prob(snr):  return 1.0 / (1.0 + np.exp(-(np.asarray(snr, float) - SNR0) / SNRW))

# ---- SEMI-SYNTHETIC (co mo neo THAT) — mo ta trung thuc, KHONG noi "fully real" ----
# THAT tu pipeline SDSS: vi tri, magnitude, flux, SNR (psfFlux*sqrt(ivar)), phan loai probPSF.
# MODELED: p_detect = completeness model CHUAN (5-sigma logistic) tren SNR THAT — forward-model qua
#   trinh detect (catalog chi chua nguon DA detect nen khong co p_detect san). Shift = INDUCED depth
#   (SNR/DEPTH_FACTOR) + phan loai mo di. Limitation: fully-observational (Stripe82 co-add vs
#   single-epoch) = future work. Sigma khong thoai hoa vi p_detect(SNR) lien tuc (khac Muc 2).
def _obs_deep(f):                                          # DEEP: p_detect(SNR that) + probPSF that
    return dict(mag=f["mag"], is_star=f["is_star"],
                p_detect=_detect_prob(f["snr_deep"]), p_star=np.clip(f["probPSF"], 0, 1))
def _obs_shallow(f, depth):                                # SHALLOW: SNR/depth -> completeness thap + phan loai mo
    ss = f["snr_deep"] / depth
    return dict(mag=f["mag"], is_star=f["is_star"], p_detect=_detect_prob(ss),
                p_star=0.5 + (np.clip(f["probPSF"], 0, 1) - 0.5) * _sep(ss))

def observe(raw_cal, raw_test, depth):
    return [_obs_deep(f) for f in raw_cal], [_obs_shallow(f, depth) for f in raw_test]

def simulate_raw(n=N_FIELDS, seed=RNG_SEED):
    rng = np.random.default_rng(seed); out = [[], []]
    for half in (0, 1):
        for _ in range(n):
            N = rng.poisson(60)
            mag = 19.5 + rng.exponential(1.6, N)             # luminosity function
            is_star = rng.random(N) < 0.40
            snr = 10 ** (0.4 * (SIM_SNR_ZP - mag))           # deep SNR tu mag
            pp = np.clip(np.where(is_star, 0.5 + 0.5 * _sep(snr), 0.5 - 0.5 * _sep(snr)), 0, 1)
            out[half].append(dict(mag=mag, is_star=is_star, snr_deep=snr, probPSF=pp))
    return out[0], out[1]

def load_sdss_raw(n=N_FIELDS):
    try:
        from astroquery.sdss import SDSS
    except ImportError:
        raise ImportError("Kaggle: !pip install astroquery ; va bat Internet=On.")
    r = SDSS_REGION
    sql = (f"SELECT p.ra, p.dec, p.type, p.probPSF_r, p.psfFlux_r, p.psfFluxIvar_r "
           f"FROM PhotoObjAll p "
           f"WHERE p.ra BETWEEN {r['ra0']} AND {r['ra1']} AND p.dec BETWEEN {r['dec0']} AND {r['dec1']} "
           f"AND p.mode=1 AND p.clean=1 AND p.type IN (3,6) "
           f"AND p.psfFluxIvar_r > 0 AND p.psfFlux_r > 0")
    tab = SDSS.query_sql(sql, data_release=17)
    if tab is None or len(tab) == 0:
        raise RuntimeError("SDSS tra ve rong — doi SDSS_REGION hoac data_release.")
    ra = np.asarray(tab["ra"], float); dec = np.asarray(tab["dec"], float)
    is_star = np.asarray(tab["type"]) == 6                    # nhan THAT (pipeline)
    probPSF = np.clip(np.asarray(tab["probPSF_r"], float), 0, 1)
    flux = np.asarray(tab["psfFlux_r"], float); ivar = np.asarray(tab["psfFluxIvar_r"], float)
    snr_deep = flux * np.sqrt(ivar)                          # backbone: SNR THAT
    mag = 22.5 - 2.5 * np.log10(np.clip(flux, 1e-3, None))
    print(f"[SDSS] {len(ra)} nguon that | sao={int(is_star.sum())} thien ha={int((~is_star).sum())} "
          f"| median SNR={np.median(snr_deep):.1f}")
    G = int(np.ceil(np.sqrt(2 * n)))
    gx = np.clip(((ra - r["ra0"]) / (r["ra1"] - r["ra0"]) * G).astype(int), 0, G - 1)
    gy = np.clip(((dec - r["dec0"]) / (r["dec1"] - r["dec0"]) * G).astype(int), 0, G - 1)
    cell = gx * G + gy; raw_cal, raw_test = [], []
    for c in range(G * G):
        idx = np.where(cell == c)[0]
        if len(idx) < 5:
            continue
        fld = dict(mag=mag[idx], is_star=is_star[idx], snr_deep=snr_deep[idx], probPSF=probPSF[idx])
        (raw_cal if c % 2 == 0 else raw_test).append(fld)
    if not raw_cal or not raw_test:
        raise RuntimeError("Khong du o luoi — giam n hoac mo rong SDSS_REGION.")
    return raw_cal, raw_test

# ------------------------------------------------------------------ PB soft-count + cut
def field_stats(f, mag_cut):
    """Tra ve n(2,), sigma(2,), gt(2,) sau completeness cut mag<mag_cut. Lop 0=sao, 1=thien ha."""
    m = f["mag"] < mag_cut
    pd = f["p_detect"][m]; ps = f["p_star"][m]
    w0 = pd * ps; w1 = pd * (1.0 - ps)                        # soft weight sao / thien ha
    n = np.array([w0.sum(), w1.sum()])
    sigma = np.sqrt(np.array([(w0 * (1 - w0)).sum(), (w1 * (1 - w1)).sum()]) + 1e-6)
    star = f["is_star"][m]
    gt = np.array([float(star.sum()), float((~star).sum())])
    return n, sigma, gt

def build(fields, mag_cut):
    return [field_stats(f, mag_cut) for f in fields]

def empirical_quantile(scores, alpha):
    s = np.asarray(scores); n = len(s)
    if n == 0: return float("inf")
    lv = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return float(np.quantile(s, lv, method="higher"))

def maxstat(rec):                                            # joint nonconformity = max qua lop
    n, sigma, gt = rec
    return float(np.max(np.abs(gt - n) / sigma))

def joint_cover(rec, q):
    n, sigma, gt = rec
    lo = np.maximum(0, n - q * sigma); hi = n + q * sigma
    return bool(np.all((gt >= lo) & (gt <= hi)))

def joint_winkler(rec, q, alpha=ALPHA):
    n, sigma, gt = rec; lo = np.maximum(0, n - q * sigma); hi = n + q * sigma
    tot = 0.0
    for k in range(len(gt)):
        s = hi[k] - lo[k]
        if gt[k] < lo[k]: s += (2 / alpha) * (lo[k] - gt[k])
        elif gt[k] > hi[k]: s += (2 / alpha) * (gt[k] - hi[k])
        tot += s
    return tot

def joint_width(rec, q):                                    # do rong macro = mean(hi-lo) qua K lop
    n, sigma, gt = rec; lo = np.maximum(0, n - q * sigma); hi = n + q * sigma
    return float(np.mean(hi - lo))

def per_class_cover(rec, q):                                # coverage RIENG tung lop [sao, thien ha]
    n, sigma, gt = rec; lo = np.maximum(0, n - q * sigma); hi = n + q * sigma
    return (gt >= lo) & (gt <= hi)

def evalrec(rec, q):                                        # (joint_cov, width, winkler, per-class-cov)
    return (joint_cover(rec, q), joint_width(rec, q), joint_winkler(rec, q), per_class_cover(rec, q))

def local_stats(cov_seq, w=50):                             # conditional validity: min rolling cov + max miss-run
    c = np.asarray(cov_seq, float); n = len(c)
    mlc = float(np.convolve(c, np.ones(w) / w, mode="valid").min()) if n >= w else float(c.mean())
    run = mx = 0
    for v in cov_seq:
        run = 0 if v else run + 1; mx = max(mx, run)
    return mlc, mx

# ================================================================== LOAD
RAW_CAL, RAW_TEST = (load_sdss_raw() if USE_SDSS else simulate_raw())
CAL_F, TEST_F = observe(RAW_CAL, RAW_TEST, DEPTH_FACTOR)
print(f"[data] {'SDSS' if USE_SDSS else 'SIM'} | CAL fields={len(CAL_F)} TEST fields={len(TEST_F)}")

# ================================================================== DIAGNOSTIC-FIRST (sweep mag_cut)
# Chon mag_cut la danh-doi: sang qua -> shift bien mat; mo qua -> bias (incomplete) thong tri.
def diagnostics(cal, test):
    cn = np.array([r[0] for r in cal]); csig = np.array([r[1] for r in cal]); cgt = np.array([r[2] for r in cal])
    tn = np.array([r[0] for r in test]); tsig = np.array([r[1] for r in test]); tgt = np.array([r[2] for r in test])
    bias = (tgt - tn).mean(0)                                # per-class bias tren TEST
    err = np.abs(tgt - tn).ravel(); sig = tsig.ravel(); sqn = np.sqrt(np.maximum(tn, 1e-6)).ravel()
    def rho(a, b):
        a = a - a.mean(); b = b - b.mean()
        d = np.sqrt((a * a).sum() * (b * b).sum())
        return float((a * b).sum() / d) if d > 0 else 0.0
    sgain = rho(sig, err) - rho(sqn, err)
    q0 = empirical_quantile([maxstat(r) for r in cal], ALPHA)        # static q tu CAL
    stat_cov_cal  = np.mean([joint_cover(r, q0) for r in cal]) * 100
    stat_cov_test = np.mean([joint_cover(r, q0) for r in test]) * 100
    return dict(bias=bias, sgain=sgain, q0=q0, cov_cal=stat_cov_cal, cov_test=stat_cov_test,
                rho_sig=rho(sig, err), rho_sqn=rho(sqn, err))

# GATE dung: dong gop = COVERAGE RECOVERY duoi shift (KHONG phai PB-sigma/width).
# -> chi yeu cau SHIFT du manh (static under-cover) + calibration lanh. bias/sig-gain chi
#    la thong tin phu (cho cau chuyen WIDTH/conditional), KHONG chan crosstable.
print("\n" + "=" * 78 + "\nDIAGNOSTIC (sweep mag_cut) - gate = SHIFT manh, khong gate theo sig-gain\n" + "=" * 78)
print(f"{'mag_cut':>7s} | {'bias(star,gal)':>18s} | {'sig-gain(width)':>15s} | {'static cov% CAL/TEST':>22s}")
print("-" * 78)
cand = []
for mag_cut in [20.0, 20.5, 21.0, 21.5, 22.0]:
    cal = build(CAL_F, mag_cut); test = build(TEST_F, mag_cut); d = diagnostics(cal, test)
    ok_shift = d["cov_test"] < COVERAGE * 100 - 3          # static under-cover >=3pp
    ok_healthy = 85.0 <= d["cov_cal"] <= 95.0              # loai cut degenerate (qua it nguon)
    tag = " <- shift manh" if (ok_shift and ok_healthy) else ""
    if ok_shift and ok_healthy: cand.append((d["cov_test"], mag_cut, cal, test, d))
    print(f"{mag_cut:7.1f} | ({d['bias'][0]:6.1f},{d['bias'][1]:6.1f}) | {d['sgain']:+15.3f} | "
          f"{d['cov_cal']:8.1f} / {d['cov_test']:8.1f}{tag}")
print("-" * 78)
print("Gate crosstable: co it nhat 1 cut voi static-cov(TEST)<87% & 85<=cov(CAL)<=95 (shift du de test recovery).")

if not cand:
    print("\n>>> KHONG co shift du manh. Chinh sim/SDSS. DUNG chay crosstable.")
else:
    cand.sort()                                           # shift manh nhat (cov_test nho nhat)
    _, MAG_CUT, CAL, TEST, D = cand[0]
    print(f"\n>>> Chon mag_cut={MAG_CUT} (shift manh nhat): static-CP sap {D['cov_cal']:.1f}%(CAL)->{D['cov_test']:.1f}%(TEST). "
          f"[phu: bias~({D['bias'][0]:.1f},{D['bias'][1]:.1f}), sig-gain={D['sgain']:+.3f}] -> chay crosstable recovery.")

    # ============================================================== CROSSTABLE joint K=2 (chi khi PASS)
    CAL_SCORES = np.array([maxstat(r) for r in CAL]); q0 = D["q0"]

    def m_static(order):
        return [evalrec(TEST[i], q0) for i in order]
    def m_aci(order, g=0.05):
        a = ALPHA; hist = list(CAL_SCORES); out = []
        for i in order:
            q = empirical_quantile(hist, a); e = evalrec(TEST[i], q); cov = e[0]
            out.append(e)
            a = min(0.5, max(1e-3, a + g * (ALPHA - (0.0 if cov else 1.0)))); hist.append(maxstat(TEST[i]))
        return out
    def m_rolling(order, ms=200):
        win = list(CAL_SCORES[-ms:]); out = []
        for i in order:
            q = empirical_quantile(win, ALPHA); out.append(evalrec(TEST[i], q))
            win.append(maxstat(TEST[i])); win = win[-ms:]
        return out
    def m_pbfixed(order, w=300):
        sc = list(CAL_SCORES[-w:]); out = []
        for i in order:
            q = empirical_quantile(sc[-w:], ALPHA); out.append(evalrec(TEST[i], q))
            sc.append(maxstat(TEST[i])); sc = sc[-w:]
        return out
    def m_adapt(order, w_max=300, w_min=40, cov_win=50, rs=0.9, rg=1.05, beta=0.03):
        sc = list(CAL_SCORES[-w_max:]); eff = w_max; recent = []; out = []
        for i in order:
            q = empirical_quantile(sc[-eff:], ALPHA) if sc else float("inf")
            e = evalrec(TEST[i], q); cov = e[0]; out.append(e)
            recent.append(1.0 if cov else 0.0); recent = recent[-cov_win:]; rc = np.mean(recent)
            if rc < COVERAGE: eff = max(w_min, int(eff * rs))
            elif rc > COVERAGE + beta: eff = min(w_max, int(eff * rg))
            sc.append(maxstat(TEST[i])); sc = sc[-w_max:]
        return out

    # ---- 4 baseline hien dai con lai (VERBATIM tu Bai 1, adapt sang joint max-statistic) ----
    import math
    from scipy.special import logsumexp
    def _wq(sc, wt, lv):
        o = np.argsort(sc); s = np.asarray(sc)[o]; w = np.asarray(wt)[o]; cw = np.cumsum(w) / w.sum()
        return s[min(np.searchsorted(cw, lv), len(s) - 1)]
    def oc_quantile(a, q):
        a = np.asarray(a, float); return float("inf") if len(a) == 0 else float(np.quantile(a, q, method="inverted_cdf"))
    def pinball_loss(y, yh, q): return np.maximum(q * (y - yh), (1 - q) * (yh - y))
    def pinball_loss_grad(y, yh, q): return -q * (y > yh) + (1 - q) * (y < yh)

    def m_nexcp(order, rho=0.99):                             # Barber 2023 (weighted quantile decay)
        hist = list(CAL_SCORES); out = []
        for i in order:
            wts = rho ** (len(hist) - 1 - np.arange(len(hist))); q = _wq(hist, wts, 1 - ALPHA)
            out.append(evalrec(TEST[i], q)); hist.append(maxstat(TEST[i]))
        return out

    class FACICore:                                          # Gibbs-Candes 2024
        def __init__(self, calib, coverage=COVERAGE):
            self.coverage = coverage; self.gammas = np.asarray([0.001 * 2 ** j for j in range(8)]); self.k = len(self.gammas)
            self.alphas = np.full(self.k, 1 - coverage); self.log_w = np.zeros(self.k); self.I = 100; self.sigma = 1.0 / (2 * self.I)
            a = 1 - coverage; denom = ((1 - a) ** 2 * a ** 3 + a ** 2 * (1 - a) ** 3) / 3
            self.eta = np.sqrt(3 / self.I) * np.sqrt((np.log(self.I * self.k) + 2) / denom); self.residuals = [float(r) for r in calib]
        def predict(self):
            lw = self.log_w; alpha = np.dot(np.exp(lw - logsumexp(lw)), self.alphas)
            return float(oc_quantile(np.abs(np.asarray(self.residuals)), 1 - alpha))
        def update(self, s):
            res = self.residuals
            if len(res) > math.floor(1 / (1 - self.coverage)):
                beta = float(np.mean(np.asarray(res) >= s)); losses = pinball_loss(beta, self.alphas, 1 - self.coverage)
                wbar = self.log_w - self.eta * losses
                self.log_w = logsumexp([wbar, np.full(self.k, logsumexp(wbar))], b=[[1 - self.sigma], [self.sigma / self.k]], axis=0)
                self.log_w = self.log_w - logsumexp(self.log_w); err = self.alphas > beta
                self.alphas = np.clip(self.alphas + self.gammas * ((1 - self.coverage) - err), 0, 1)
            res.append(float(s))
    def m_faci(order):
        m = FACICore(CAL_SCORES); out = []
        for i in order:
            q = m.predict(); out.append(evalrec(TEST[i], q)); m.update(maxstat(TEST[i]))
        return out

    class _OGD:                                              # SAOCP expert (Bhatnagar 2023)
        def __init__(self, t, scale, alpha, yhat_0, g=8):
            self.scale = scale; self.base_lr = scale / np.sqrt(3); self.alpha = alpha; self.yhat = yhat_0; self.grad_norm = 0; u = 0
            while t % 2 == 0: t /= 2; u += 1
            self.lifetime = g * 2 ** u; self.z = 0; self.wz = 0; self.s_t = 0
        @property
        def expired(self): return self.s_t > self.lifetime
        def loss(self, y): return pinball_loss(y, self.yhat, 1 - self.alpha)
        @property
        def w(self): return 0 if self.s_t == 0 else self.z / self.s_t * (1 + self.wz)
        def update(self, y, meta_loss):
            w = self.w; g = np.clip((meta_loss - self.loss(y)) / self.scale / max(self.alpha, 1 - self.alpha), -1 * (w > 0), 1)
            self.z += g; self.wz += g * w; self.s_t += 1; grad = pinball_loss_grad(y, self.yhat, 1 - self.alpha); self.grad_norm += grad ** 2
            if self.grad_norm != 0: self.yhat = max(0, self.yhat - self.base_lr / np.sqrt(self.grad_norm) * grad)
    class SAOCPCore:
        def __init__(self, warm, coverage=COVERAGE, lifetime=8):
            self.coverage = coverage; self.lifetime = lifetime; self.t = 1; self.experts = {}
            r = np.abs(np.asarray(warm, float)); self.scale = 1.0 if len(r) == 0 else float(np.max(r) * np.sqrt(3))
            for s in r: self._step(float(s))
        def get_p(self):
            e = self.experts; prior = {t: 1 / (t ** 2 * (1 + np.floor(np.log2(t)))) for t in e}; z = sum(prior.values())
            if z == 0: return {}
            prior = {t: v / z for t, v in prior.items()}; p = {t: prior[t] * max(0, x.w) for t, x in e.items()}; sp = sum(p.values())
            return {t: v / sp for t, v in p.items()} if sp > 0 else prior
        def predict(self):
            p = self.get_p(); return sum(p[t] * self.experts[t].yhat for t in p)
        def _step(self, s):
            sh = self.predict()
            for t in [k for k, v in self.experts.items() if v.expired]: self.experts.pop(t)
            self.experts[self.t] = _OGD(self.t, self.scale, 1 - self.coverage, yhat_0=sh, g=self.lifetime)
            ml = pinball_loss(s, self.predict(), self.coverage)
            for e in self.experts.values(): e.update(s, ml)
            self.t += 1
        def quantile(self): return self.predict()
        def update(self, s): self._step(float(s))
    def m_saocp(order):
        m = SAOCPCore(CAL_SCORES, COVERAGE, 8); out = []
        for i in order:
            q = m.quantile(); out.append(evalrec(TEST[i], q)); m.update(maxstat(TEST[i]))
        return out

    class COP:                                               # Hu 2026
        def __init__(self, eta, lam=1.0, w=100, warm=None):
            self.eta = eta; self.lam = lam; self.w = w; self.qhat = empirical_quantile(np.asarray(warm), ALPHA)
            self.q = self.qhat; self.win = list(np.asarray(warm)[-w:])
        def get_q(self): return max(0.0, self.q)
        def update(self, s, qu):
            self.qhat = max(0.0, self.qhat + self.eta * ((1.0 if s > qu else 0.0) - ALPHA)); self.win.append(float(s)); self.win = self.win[-self.w:]
            F = float(np.mean(np.asarray(self.win) <= self.qhat)); self.q = max(0.0, self.qhat - self.lam * (F - (1 - ALPHA)))
    def m_cop(order, eta=None):
        eta = 0.15 * q0 if eta is None else eta            # scale-aware (score astro nho hon trees)
        m = COP(eta, warm=CAL_SCORES); out = []
        for i in order:
            q = m.get_q(); out.append(evalrec(TEST[i], q)); m.update(maxstat(TEST[i]), q)
        return out

    ROWS = [("Static split-CP (no update)", m_static), ("ACI (Gibbs-Candes 2021)", m_aci),
            ("NexCP (Barber 2023)", m_nexcp), ("FACI (Gibbs-Candes 2024)", m_faci),
            ("SAOCP (Bhatnagar 2023)", m_saocp), ("COP (Hu 2026)", m_cop),
            ("Rolling-Origin CP (2026)", m_rolling), ("PB-JCI Online-Fixed (ours)", m_pbfixed),
            ("Adaptive PB-JCI Online (ours)", m_adapt)]
    T = len(TEST)
    def agg(fn, seeds=5):
        cs, ks, allw = [], [], []
        pc = [[], []]; mlcs, mruns = [], []                  # per-class cover ; local stats
        for sd in range(seeds):
            o = np.random.RandomState(sd).permutation(T); r = fn(o)
            cseq = [x[0] for x in r]
            cs.append(np.mean(cseq) * 100)
            ks.append(np.mean([x[2] for x in r])); allw += [x[1] for x in r]
            for k in (0, 1): pc[k] += [bool(x[3][k]) for x in r]
            mlc, mr = local_stats(cseq); mlcs.append(mlc); mruns.append(mr)
        return dict(cov=np.mean(cs), cov_std=np.std(cs),
                    cov_star=np.mean(pc[0]) * 100, cov_gal=np.mean(pc[1]) * 100,
                    width=float(np.mean(allw)), wink=np.mean(ks), wink_std=np.std(ks),
                    min_local=float(np.mean(mlcs)) * 100, max_miss=int(np.max(mruns)))

    print("\n" + "=" * 100 + f"\nCROSSTABLE joint K=2 (cal=deep -> test=shallow), 5 seeds, target {COVERAGE*100:.0f}%\n" + "=" * 100)
    print(f"{'Method':30s} | {'Joint%':>7s} | {'Sao%':>5s} | {'ThHa%':>5s} | {'AvgW':>6s} | "
          f"{'Winkler':>8s} | {'minLoc%':>7s} | {'miss':>4s}")
    print("-" * 100); tab = {}
    for name, fn in ROWS:
        d = agg(fn); tab[name] = d
        print(f"{name:30s} | {d['cov']:6.1f} | {d['cov_star']:5.1f} | {d['cov_gal']:5.1f} | {d['width']:6.2f} | "
              f"{d['wink']:8.2f} | {d['min_local']:7.1f} | {d['max_miss']:4d}")
    print("-" * 100)
    print("Sao%/ThHa% = coverage RIENG tung lop (marginal). AvgW/Winkler = mean (nhu Bai 1). "
          "minLoc% = min rolling coverage (w=50), miss = chuoi miss dai nhat (conditional validity).")
    # Dong gop = COVERAGE RECOVERY: Adaptive co hoi phuc ~nominal trong khi baseline under-cover?
    ad = tab['Adaptive PB-JCI Online (ours)']
    others = {n: tab[n] for n in tab if 'Static' not in n and 'Adaptive' not in n}
    bo = max(others, key=lambda n: others[n]['cov'])           # baseline coverage cao nhat
    nonstat = {n: tab[n] for n in tab if 'Static' not in n}
    wrank = sorted(nonstat, key=lambda n: nonstat[n]['wink']).index('Adaptive PB-JCI Online (ours)') + 1
    print(f"Static-CP={tab['Static split-CP (no update)']['cov']:.1f}% (sap) | Adaptive={ad['cov']:.1f}% | "
          f"baseline cao nhat khac = {bo.split(' (')[0]} {others[bo]['cov']:.1f}% | "
          f"Adaptive Winkler hang {wrank}/{len(nonstat)}")
    ad_valid = ad['cov'] >= COVERAGE * 100 - 0.5                     # ~ dat nominal
    ad_best_wink = (wrank == 1)                                      # Winkler = headline metric
    uniq_nominal = ad_valid and all(others[x]['cov'] < COVERAGE * 100 for x in others)
    if ad_best_wink and ad_valid:
        print(f">>> WIN (headline): Adaptive Winkler #1 ({ad['wink']:.2f}) + coverage {ad['cov']:.1f}% >= nominal"
              + (" — DUY NHAT dat >=90%." if uniq_nominal else "."))
    elif uniq_nominal:
        print(">>> COVERAGE-RECOVERY: Adaptive DUY NHAT dat nominal (Winkler chua #1).")
    else:
        print(">>> Adaptive chua tach ro khoi baseline - xem ky (nhu trees).")

    import os
    # ---------------------------------------------------------- DEPTH sweep: recovery vs do sau
    print("\n" + "=" * 78 + f"\nDEPTH sweep (mag_cut={MAG_CUT} co dinh) - shift cang manh, Adaptive cang tach\n" + "=" * 78)
    swrows = ROWS[1:]                                        # bo Static (da co cot staticTEST%)
    hdr = [n.split(' (')[0][:8] for n, _ in swrows]
    print(f"{'depth':>5s} | {'statTEST%':>9s} | " + " | ".join(f"{h:>8s}" for h in hdr))
    print("-" * 92); depth_tab = {}
    for depth in [2.0, 3.0, 4.0, 5.0]:
        _, tf = observe(RAW_CAL, RAW_TEST, depth)
        TEST = build(tf, MAG_CUT); T = len(TEST)
        st = np.mean([joint_cover(r, q0) for r in TEST]) * 100
        covs = [agg(fn)['cov'] for _, fn in swrows]
        depth_tab[str(depth)] = dict(static=st, covs=dict(zip([n for n, _ in swrows], covs)))
        print(f"{depth:5.1f} | {st:9.1f} | " + " | ".join(f"{c:8.1f}" for c in covs))
    print("-" * 92)
    print("Ky vong: depth tang -> static + fixed-rate sap nhanh, Adaptive giu ~nominal lau nhat.")

    # ---------------------------------------------------------- SENSITIVITY do doc completeness (kappa = SNRW)
    # Reviewer hoi: ket luan co vung theo tham so mo hinh hoa khong? -> quet do doc completeness.
    print("\n" + "=" * 78 + f"\nSENSITIVITY kappa (SNRW), mag_cut={MAG_CUT} + depth={DEPTH_FACTOR} co dinh\n" + "=" * 78)
    print(f"{'kappa':>6s} | {'static cov%':>11s} | {'Adapt cov%':>10s} | {'Adapt Wink':>10s}")
    print("-" * 78)
    def _run_adapt(scores, recs, w_max=300, w_min=40, cov_win=50, rs=0.9, rg=1.05, beta=0.03):
        sc = list(scores[-w_max:]); eff = w_max; recent = []; cov = []; wk = []
        for r in recs:
            q = empirical_quantile(sc[-eff:], ALPHA) if sc else float("inf")
            c = joint_cover(r, q); cov.append(c); wk.append(joint_winkler(r, q))
            recent.append(1.0 if c else 0.0); recent = recent[-cov_win:]; rc = np.mean(recent)
            if rc < COVERAGE: eff = max(w_min, int(eff * rs))
            elif rc > COVERAGE + beta: eff = min(w_max, int(eff * rg))
            sc.append(maxstat(r)); sc = sc[-w_max:]
        return np.mean(cov) * 100, float(np.mean(wk))
    _snrw0 = SNRW; sens = {}
    for kap in [1.0, 1.5, 2.0, 3.0]:
        SNRW = kap                                            # doi do doc completeness (global)
        cf, tf = observe(RAW_CAL, RAW_TEST, DEPTH_FACTOR)
        calk = build(cf, MAG_CUT); testk = build(tf, MAG_CUT)
        sck = np.array([maxstat(r) for r in calk]); q0k = empirical_quantile(sck, ALPHA)
        st = np.mean([joint_cover(r, q0k) for r in testk]) * 100
        ac, aw = _run_adapt(sck, testk); sens[str(kap)] = dict(static=st, adapt_cov=ac, adapt_wink=aw)
        print(f"{kap:6.1f} | {st:11.1f} | {ac:10.1f} | {aw:10.2f}")
    SNRW = _snrw0                                             # KHOI PHUC de PART 2 dung
    print("-" * 78)
    print("Ky vong: Adaptive giu cov ~muc tieu + Winkler thap qua moi kappa -> ket luan vung.")

    # ---------------------------------------------------------- PART 2: abrupt stream recovery
    pre = build([_obs_deep(f) for f in RAW_TEST], MAG_CUT)                    # deep (pre-shift)
    post = build([_obs_shallow(f, DEPTH_FACTOR) for f in RAW_TEST], MAG_CUT)  # shallow (post-shift)
    TEST = pre + post; T = len(TEST); CP = len(pre); order = np.arange(T)
    def rollc(c, w=50):
        c = np.asarray(c, float)
        return np.array([c[max(0, t - w + 1):t + 1].mean() for t in range(len(c))])
    print("\n" + "=" * 78 + f"\nPART 2 - abrupt stream deep->shallow (changepoint {CP}), recovery\n" + "=" * 78)
    print(f"{'Method':32s} | {'pre cov':>8s} | {'post-30 cov':>11s} | {'overall':>8s}")
    print("-" * 78); curves = {}
    for name, fn in ROWS:
        r = fn(order); c = np.array([x[0] for x in r], float); curves[name] = rollc(c).tolist()
        print(f"{name:32s} | {c[:CP].mean():7.1%} | {c[CP:CP + 30].mean():10.1%} | {c.mean():7.1%}")
    print("-" * 78)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 3.6)); ax.axvspan(CP, T, color="#D55E00", alpha=0.05)
        # BO Static split-CP ra khoi hinh -> so cac baseline HIEN DAI voi Adaptive PB-JCI Online
        col = {"ACI (Gibbs-Candes 2021)": "#0072B2", "NexCP (Barber 2023)": "#56B4E9",
               "FACI (Gibbs-Candes 2024)": "#009E73", "SAOCP (Bhatnagar 2023)": "#E69F00",
               "COP (Hu 2026)": "#CC79A7", "Rolling-Origin CP (2026)": "#8C564B",
               "PB-JCI Online-Fixed (ours)": "#999999", "Adaptive PB-JCI Online (ours)": "#D55E00"}
        for name in col:                                        # Adaptive: day + noi bat, baseline: mo
            is_ad = "Adaptive" in name
            ax.plot(curves[name], color=col[name], lw=2.4 if is_ad else 1.0,
                    label=name.split(" (")[0], zorder=6 if is_ad else 3, alpha=1.0 if is_ad else 0.7)
        ax.axhline(COVERAGE, color="0.2", ls="--", lw=1, label=f"target {COVERAGE:.0%}")
        ax.axvline(CP, color="0.5", ls=":", lw=0.9)
        ax.set_xlabel("stream step (khoi cam = shallow survey)"); ax.set_ylabel("rolling coverage (w=50)")
        ax.set_ylim(0.4, 1.04); ax.legend(ncol=3, fontsize=7, loc="lower left")
        ax.set_title("Astro: baseline hien dai vs Adaptive PB-JCI Online (deep->shallow shift)", fontsize=9.5)
        fig.tight_layout()
        fig_dir = "figures" if os.path.isdir("figures") else "."
        fig.savefig(os.path.join(fig_dir, "astro_recovery.png"), dpi=200, bbox_inches="tight")
        fig.savefig(os.path.join(fig_dir, "astro_recovery.pdf"), bbox_inches="tight")
        print(f"WROTE {os.path.join(fig_dir, 'astro_recovery.png')}")
    except Exception as e:
        print(f"(bo qua hinh: {e})")

    import json
    out_dir = "results" if os.path.isdir("results") else "."
    out_path = os.path.join(out_dir, "astro_diagnostic_results.json")
    json.dump({"mag_cut": MAG_CUT, "depth_factor": DEPTH_FACTOR,
               "diagnostic": {k: (v.tolist() if hasattr(v, 'tolist') else v) for k, v in D.items()},
               "crosstable": tab, "depth_sweep": depth_tab, "sensitivity_kappa": sens,
               "part2": {"cp": CP, "curves": curves}},
              open(out_path, "w"), indent=1)
    print(f"Da luu {out_path}")
