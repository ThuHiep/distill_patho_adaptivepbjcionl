"""
eval_cpcp.py — Baseline 2026 CPCP (Colorful Pinball Conformal Prediction, Chen & Li, Tsinghua,
ICML 2026, arXiv 2512.24139). Dùng ĐÚNG code official (github.com/Cqyiiii/Colorful-Pinball-
Conformal-Prediction-CPCP, MIT) — import verbatim model + trainer của họ; chỉ thay bước cuối
`get_metrics_nd(...)` bằng TRẢ interval/ảnh để ta chấm worst-org (organ_conditional_stats). Thủ tục
lõi (mean-net → 3-head density-weighted pinball → split conformal trên rectified score) GIỮ NGUYÊN
y hệt `methods.run_rcp_density_improved`.

Vì sao hợp: CPCP tối thiểu hoá Mean Squared Conditional Error (MSCE) → đấu TRỰC TIẾP trục
conditional coverage / worst-org của R2. Method 2026 (ICML), code official, deps sạch
(torch/numpy/sklearn — KHÔNG auto_LiRPA/rpy2). Áp lên đặc trưng sâu student (--dump_feat), leak-free.

CPCP train mạng riêng → cần 3 phần: train (mean+quantile) / cal (conformal) / test. Ta chia
tr/cal/te theo seed (mặc định 0.4/0.2/0.4). MAE = |y − mu_net|. Cùng organ_conditional_stats/Winkler.

Cài: git clone https://github.com/Cqyiiii/Colorful-Pinball-Conformal-Prediction-CPCP.git CPCP
     (deps: torch numpy scikit-learn pandas scipy — đã có)
Chạy (pkl phải có 'feat'):
  python eval_cpcp.py --preds work/student_r2_pannuke_f3_nocolon_poisson_feat.pkl \
      --cpcp_dir ./CPCP --seeds 5 --alpha 0.1 --min_organ_imgs 10
"""
from __future__ import annotations
import argparse, os, pickle, sys, warnings
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_coverage_transfer import winkler_score, organ_conditional_stats  # noqa: E402

warnings.filterwarnings("ignore")


def load_feat(path):
    obj = pickle.load(open(path, "rb"))
    preds = obj["preds"]
    if "feat" not in preds[0]:
        raise SystemExit("pkl KHÔNG có 'feat' — chạy distill_student_r2.py --dump_feat.")
    gt = np.array([float(np.asarray(g).reshape(-1)[0]) for g in obj["gts"]])
    organs = list(obj.get("organs", ["_all_"] * len(preds)))
    if len(organs) != len(preds):
        organs = ["_all_"] * len(preds)
    feat = np.stack([np.asarray(p["feat"], np.float32) for p in preds])
    return feat, gt, organs


def cpcp_intervals(X_tr, Y_tr, X_cal, Y_cal, X_te, alpha, epsilon, mode, clip_max, mix_ratio):
    """Y HỆT methods.run_rcp_density_improved (CPCP official) nhưng TRẢ (mu_te, y_lo, y_hi) thay vì
    get_metrics_nd. Mọi thành phần (Net, MonotonicThreeHeadNet, train_mean, train_three_head_base,
    finetune_main_head_improved) import verbatim từ repo CPCP."""
    import torch
    from models import Net, MonotonicThreeHeadNet
    from trainers import train_mean, train_three_head_base, finetune_main_head_improved
    from utils import to_tensor, to_numpy, DEVICE
    D = Y_tr.shape[1]
    mu_net = Net(X_tr.shape[1], D).to(DEVICE)
    train_mean(mu_net, to_tensor(X_tr), to_tensor(Y_tr)); mu_net.eval()
    with torch.no_grad():
        mu_cal = to_numpy(mu_net(to_tensor(X_cal)))
        S_cal = np.max(np.abs(Y_cal - mu_cal), axis=1)
        mu_te = to_numpy(mu_net(to_tensor(X_te)))
    n = len(X_cal); idx1 = int(0.4 * n); idx2 = int(0.8 * n)
    perm = np.random.permutation(n)
    X_cal, S_cal = X_cal[perm], S_cal[perm]
    X_est1, S_est1 = X_cal[:idx1], S_cal[:idx1]
    X_est2, S_est2 = X_cal[idx1:idx2], S_cal[idx1:idx2]
    X_score, S_score = X_cal[idx2:], S_cal[idx2:]
    target_q = 1 - alpha
    taus_list = [max(0.01, target_q - epsilon), target_q, min(0.99, target_q + epsilon)]
    r_net = MonotonicThreeHeadNet(X_tr.shape[1]).to(DEVICE)
    train_three_head_base(r_net, to_tensor(X_est1), to_tensor(S_est1.reshape(-1, 1)), taus_list, epochs=200)
    finetune_main_head_improved(r_net, to_tensor(X_est2), to_tensor(S_est2.reshape(-1, 1)),
                                target_tau=target_q, epsilon=epsilon, epochs=200,
                                mode=mode, clip_max=clip_max, mix_ratio=mix_ratio, save_weights_path=None)
    r_net.eval()
    with torch.no_grad():
        tau_conf = to_numpy(r_net(to_tensor(X_score)))[:, 1].flatten()
        tau_te = to_numpy(r_net(to_tensor(X_te)))[:, 1].flatten()
        tau_conf, tau_te = np.maximum(tau_conf, 1e-4), np.maximum(tau_te, 1e-4)
        scores = S_score - tau_conf
        q = np.quantile(scores, np.ceil((1 - alpha) * (len(scores) + 1)) / len(scores))
        width = (tau_te + q)
        mu_te = mu_te.reshape(-1)
        y_lo = np.maximum(0.0, mu_te - width); y_hi = mu_te + width
    return mu_te, y_lo, y_hi


def eval_cpcp(feat, gt, organs, alpha, seeds, min_organ_imgs, cpcp_dir,
              epsilon, mode, clip_max, mix_ratio, tr_ratio, cal_ratio):
    sys.path.insert(0, os.path.abspath(cpcp_dir))
    N = len(gt); target = 1 - alpha
    organs = np.asarray(organs, dtype=object)
    per_seed, pooled = [], []
    for s in range(seeds):
        rng = np.random.RandomState(1000 + s)
        perm = rng.permutation(N)
        n_tr = int(tr_ratio * N); n_cal = int(cal_ratio * N)
        tr, cal, te = perm[:n_tr], perm[n_tr:n_tr + n_cal], perm[n_tr + n_cal:]
        mu_te, lo, hi = cpcp_intervals(feat[tr], gt[tr].reshape(-1, 1).astype(np.float32),
                                       feat[cal], gt[cal].reshape(-1, 1).astype(np.float32),
                                       feat[te], alpha, epsilon, mode, clip_max, mix_ratio)
        yte = gt[te]
        cov = (yte >= lo) & (yte <= hi)
        wink = np.array([winkler_score(lo[j], hi[j], yte[j], alpha) for j in range(len(te))])
        ae = np.abs(yte - mu_te)
        per_seed.append({"coverage": float(cov.mean()), "width": float((hi - lo).mean()),
                         "winkler": float(wink.mean()), "mae": float(ae.mean())})
        for j, ti in enumerate(te):
            pooled.append((organs[ti], int(ti), bool(cov[j]), float(cov[j])))
        print(f"  seed {s+1}/{seeds}: cov={cov.mean():.3f} width={(hi-lo).mean():.2f} winkler={wink.mean():.2f}")
    def ms(k): v = np.array([d[k] for d in per_seed]); return {"mean": float(v.mean()), "std": float(v.std())}
    cond = organ_conditional_stats(pooled, target, min_organ_imgs)
    return {"coverage": ms("coverage"), "width": ms("width"), "winkler": ms("winkler"),
            "mae": ms("mae"), "conditional": cond, "winkler_seeds": [d["winkler"] for d in per_seed]}


def pretty(d, cfg):
    print("\n" + "=" * 92)
    print(f"CPCP — Colorful Pinball CP (Chen & Li, ICML 2026) | alpha={cfg['alpha']} "
          f"target={1-cfg['alpha']:.3f} seeds={cfg['seeds']} mode={cfg['mode']} (X=deep feat student)")
    print("=" * 92)
    cd = d["conditional"]; wo = cd["worst_organ_coverage"]; gap = cd["organ_coverage_gap"]
    wo_s = f"{wo:.3f}" if wo is not None else "n/a"; gap_s = f"{gap:.3f}" if gap is not None else "n/a"
    print(f"marg.cov={d['coverage']['mean']:.3f}  width={d['width']['mean']:.2f}  "
          f"Winkler={d['winkler']['mean']:.2f}±{d['winkler']['std']:.2f}  MAE={d['mae']['mean']:.2f}")
    print(f"worst-org={wo_s}  org-gap={gap_s}  #under={cd['n_organs_undercovered']}/{cd['n_organs_eval']}")
    print("[GHI CHÚ] CPCP tối thiểu hoá MSCE (conditional) → so worst-org với R2/CondConf/PCP (mục 8).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True, help="pkl R2 CÓ 'feat' (dump_feat)")
    ap.add_argument("--cpcp_dir", default="./CPCP", help="repo CPCP (chứa methods.py/models.py)")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--min_organ_imgs", type=int, default=10)
    ap.add_argument("--epsilon", type=float, default=0.02)
    ap.add_argument("--mode", default="clip", choices=["vanilla", "clip", "mix"])
    ap.add_argument("--clip_max", type=float, default=5.0)
    ap.add_argument("--mix_ratio", type=float, default=0.5)
    ap.add_argument("--tr_ratio", type=float, default=0.4)
    ap.add_argument("--cal_ratio", type=float, default=0.2)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    feat, gt, organs = load_feat(args.preds)
    print(f"loaded {args.preds} N={len(gt)} feat_dim={feat.shape[1]} organs={len(set(organs))}")
    d = eval_cpcp(feat, gt, organs, args.alpha, args.seeds, args.min_organ_imgs, args.cpcp_dir,
                  args.epsilon, args.mode, args.clip_max, args.mix_ratio, args.tr_ratio, args.cal_ratio)
    cfg = {"alpha": args.alpha, "seeds": args.seeds, "mode": args.mode}
    pretty(d, cfg)
    if args.out:
        import json
        json.dump({"config": cfg, "CPCP": d}, open(args.out, "w"), indent=2, ensure_ascii=False)
        print(f"\nĐã lưu {args.out}")


if __name__ == "__main__":
    main()
