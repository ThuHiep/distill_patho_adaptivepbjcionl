"""
test_pbud_losses.py — De-risk phần TOÁN của loss TRƯỚC khi lên vast.

Chạy LOCAL (chỉ cần torch + numpy, không cần PathoSAM/GPU):
    python test_pbud_losses.py

Kiểm mọi rủi ro của loss mà không phụ thuộc dữ liệu thật:
  1. pb_moments khớp ĐÚNG numpy conformal.py (pb_count/pb_variance) — không lệch định nghĩa.
  2. pbud_loss: forward hữu hạn, backward ra gradient KHÁC 0 trên student (khả vi thật).
  3. Số hạng VARIANCE (gamma) THỰC SỰ tạo gradient (nếu không, PBUD vô nghĩa).
  4. ccad_loss: forward hữu hạn, backward khác 0, và GIẢM khi các nhóm cân bằng hơn (sanity đúng hướng).
  5. Edge cases: N=0 (không instance), K=1 (NuInsSeg) và K=4 (MoNuSAC) đều không nổ.
"""
import sys, os
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "kaggle", "lib"))
from pbud_losses import pb_moments, pbud_loss, ccad_loss, standardized_residual
import conformal as C  # numpy reference của repo

PASS, FAIL = "\033[92mPASS\033[0m", "\033[91mFAIL\033[0m"
results = []


def check(name, cond, extra=""):
    results.append(cond)
    print(f"  [{PASS if cond else FAIL}] {name}" + (f"  {extra}" if extra else ""))


def test_1_matches_numpy():
    print("Test 1: pb_moments khớp numpy conformal.py")
    rng = np.random.RandomState(0)
    for K in (1, 4, 5):
        N = rng.randint(1, 30)
        s = rng.uniform(0, 1, N).astype(np.float64)
        p = rng.uniform(0, 1, (N, K)); p = p / p.sum(1, keepdims=True)
        mu_np = C.pb_count(s, p); var_np = C.pb_variance(s, p)
        mu_t, var_t = pb_moments(torch.tensor(s), torch.tensor(p))
        check(f"K={K} mean khớp", np.allclose(mu_t.numpy(), mu_np, atol=1e-9),
              f"max|Δ|={np.abs(mu_t.numpy()-mu_np).max():.2e}")
        check(f"K={K} var khớp", np.allclose(var_t.numpy(), var_np, atol=1e-9),
              f"max|Δ|={np.abs(var_t.numpy()-var_np).max():.2e}")


def test_2_pbud_differentiable():
    print("Test 2: pbud_loss khả vi (gradient tới student khác 0)")
    torch.manual_seed(0)
    N, K = 12, 4
    s_S = torch.rand(N, requires_grad=True)
    p_logit = torch.randn(N, K, requires_grad=True)
    p_S = torch.softmax(p_logit, dim=1)
    s_T = torch.rand(N); p_T = torch.softmax(torch.randn(N, K), dim=1)
    gt = torch.rand(K) * 10
    out = pbud_loss(s_S, p_S, s_T, p_T, gt)
    check("loss hữu hạn", torch.isfinite(out["loss"]).item(), f"loss={out['loss'].item():.4f}")
    out["loss"].backward()
    g_s = s_S.grad.abs().sum().item(); g_p = p_logit.grad.abs().sum().item()
    check("grad tới s_S khác 0", g_s > 0, f"Σ|grad|={g_s:.4f}")
    check("grad tới p_logit khác 0", g_p > 0, f"Σ|grad|={g_p:.4f}")


def test_3_variance_term_contributes():
    print("Test 3: số hạng VARIANCE (gamma) tạo gradient riêng")
    torch.manual_seed(1)
    N, K = 15, 1
    def grad_s(gamma):
        s_S = torch.rand(N, requires_grad=True)
        p_S = torch.ones(N, K)
        s_T = torch.rand(N); p_T = torch.ones(N, K); gt = torch.rand(K) * 10
        out = pbud_loss(s_S, p_S, s_T, p_T, gt, alpha=0.0, beta=0.0, gamma=1.0)
        out["loss"].backward()
        return s_S.grad.abs().sum().item(), out["loss"].item()
    g, l = grad_s(1.0)
    check("gamma=1 (chỉ var): loss hữu hạn & grad khác 0", np.isfinite(l) and g > 0,
          f"loss={l:.4f} Σ|grad|={g:.4f}")


def test_4_ccad_sanity():
    print("Test 4: ccad_loss — cân bằng nhóm thì loss thấp hơn")
    torch.manual_seed(2)
    # dựng batch: mọi ảnh cùng phân phối => nhóm cân bằng => loss ~ thấp
    B, K = 12, 1
    def make_batch(imbalance):
        mus, vars_, gts, groups = [], [], [], []
        for i in range(B):
            g = i % 3
            # nhóm 2 lệch mạnh nếu imbalance
            bias = (imbalance * 8.0) if g == 2 else 0.0
            mu = torch.tensor([5.0 + bias], requires_grad=True)
            var = torch.tensor([4.0])
            gt = torch.tensor([5.0])
            mus.append(mu); vars_.append(var); gts.append(gt); groups.append(g)
        return mus, vars_, gts, groups
    mus_b, vars_b, gts_b, grp = make_batch(imbalance=1.0)
    out_imb = ccad_loss(mus_b, vars_b, gts_b, grp)
    mus_ok, vars_ok, gts_ok, grp2 = make_batch(imbalance=0.0)
    out_bal = ccad_loss(mus_ok, vars_ok, gts_ok, grp2)
    check("loss hữu hạn", torch.isfinite(out_imb["loss"]).item(),
          f"imb={out_imb['loss'].item():.4f} bal={out_bal['loss'].item():.4f}")
    check("nhóm lệch -> loss CAO hơn nhóm cân bằng",
          out_imb["loss"].item() > out_bal["loss"].item())
    # khả vi
    mus_b2, vars_b2, gts_b2, grp3 = make_batch(imbalance=1.0)
    o = ccad_loss(mus_b2, vars_b2, gts_b2, grp3)
    o["loss"].backward()
    gsum = sum((m.grad.abs().sum().item() if m.grad is not None else 0.0) for m in mus_b2)
    check("ccad khả vi (grad khác 0)", gsum > 0, f"Σ|grad|={gsum:.4f}")


def test_5_edge_cases():
    print("Test 5: edge cases N=0, K=1, K=4")
    # N=0
    mu, var = pb_moments(torch.zeros(0), torch.zeros(0, 4))
    check("N=0 -> mu,var = 0, shape (K,)", mu.shape == (4,) and float(mu.sum()) == 0.0)
    # pbud với student rỗng (ảnh không phát hiện instance nào)
    gt = torch.tensor([3.0, 0.0, 1.0, 2.0])
    out = pbud_loss(torch.zeros(0), torch.zeros(0, 4), torch.zeros(0), torch.zeros(0, 4), gt)
    check("pbud N=0 loss hữu hạn", torch.isfinite(out["loss"]).item(),
          f"loss={out['loss'].item():.4f}")
    # K=1
    s = torch.rand(8, requires_grad=True)
    out1 = pbud_loss(s, torch.ones(8, 1), torch.rand(8), torch.ones(8, 1), torch.tensor([4.0]))
    out1["loss"].backward()
    check("K=1 khả vi", s.grad.abs().sum().item() > 0)


if __name__ == "__main__":
    print("=" * 66)
    print("DE-RISK LOSS (local, không cần PathoSAM/GPU) — torch", torch.__version__)
    print("=" * 66)
    for t in (test_1_matches_numpy, test_2_pbud_differentiable,
              test_3_variance_term_contributes, test_4_ccad_sanity, test_5_edge_cases):
        t()
    n_pass = sum(results); n = len(results)
    print("=" * 66)
    print(f"KẾT QUẢ: {n_pass}/{n} pass")
    print("=" * 66)
    sys.exit(0 if n_pass == n else 1)
