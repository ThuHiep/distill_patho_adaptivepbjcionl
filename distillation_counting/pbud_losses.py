"""
pbud_losses.py — Loss cho Paper 2: PBUD + CCAD (torch, khả vi).

Định nghĩa Poisson-Binomial (khớp ĐÚNG conformal.py: pb_count / pb_variance):
  Cho per-instance existence s (N,) và class prob p (N,K):
    w      = s[:,None] * p              # (N,K), đóng góp Bernoulli của instance i vào lớp k
    mu_k   = Σ_i w_{i,k}                # kỳ vọng số đếm lớp k
    var_k  = Σ_i w_{i,k}(1 - w_{i,k})   # phương sai PB lớp k

Ba loss:
  - pb_moments        : (mu, var) khả vi từ (s, p).
  - pbud_loss         : α·task + β·(mean KD) + γ·(VARIANCE KD)  ← số hạng γ là điểm mới (PBUD).
  - ccad_loss         : ép phân phối score chuẩn hoá đồng đều giữa các NHÓM (organ) → cân bằng
                        conditional coverage. Đây là surrogate KHẢ VI cho conditional coverage
                        (ghi rõ là surrogate, không giấu). Target = cái conformal KHÔNG bảo đảm.

LƯU Ý TRUNG THỰC: ccad_loss là một surrogate (match mean+std của score chuẩn hoá theo nhóm về global).
Nó nhắm ĐÚNG đại lượng (phân phối score theo subgroup) nhưng không phải conditional coverage đo trực
tiếp (coverage không khả vi). Có thể thay bằng MMD/quantile-matching giữa các nhóm — để mở.
"""
from __future__ import annotations
from typing import Dict, Optional
import torch

EPS = 1e-6


def pb_moments(s: torch.Tensor, p: torch.Tensor):
    """s:(N,), p:(N,K) -> mu:(K,), var:(K,). Khả vi. N=0 -> zeros(K)."""
    K = p.shape[1] if p.dim() == 2 else 1
    if s.numel() == 0:
        z = s.new_zeros(K)
        return z, z
    w = s.unsqueeze(1) * p                      # (N,K)
    mu = w.sum(dim=0)                            # (K,)
    var = (w * (1.0 - w)).sum(dim=0)            # (K,)
    return mu, var


def pbud_loss(s_S: torch.Tensor, p_S: torch.Tensor,
              s_T: torch.Tensor, p_T: torch.Tensor,
              gt: torch.Tensor,
              alpha: float = 0.4, beta: float = 0.3, gamma: float = 0.3) -> Dict[str, torch.Tensor]:
    """PBUD cho MỘT ảnh. Trả dict gồm 'loss' + các thành phần (để log/ablation).
    gt:(K,) nhãn đếm thật. gamma>0 là điểm mới (distill VARIANCE)."""
    mu_S, var_S = pb_moments(s_S, p_S)
    mu_T, var_T = pb_moments(s_T, p_T)
    sig_S = torch.sqrt(var_S + EPS)
    sig_T = torch.sqrt(var_T + EPS)

    L_task = ((mu_S - gt) ** 2).mean()          # giám sát nhãn thật
    L_mean = ((mu_S - mu_T) ** 2).mean()        # distill MEAN (KD chuẩn cũng có)
    L_var = ((sig_S - sig_T) ** 2).mean()       # ★ distill VARIANCE (PBUD)
    loss = alpha * L_task + beta * L_mean + gamma * L_var
    return {"loss": loss, "L_task": L_task.detach(),
            "L_mean": L_mean.detach(), "L_var": L_var.detach()}


def soft_winkler_loss(mu: torch.Tensor, var: torch.Tensor, gt: torch.Tensor,
                      alpha: float = 0.1, k: float = 1.6449) -> torch.Tensor:
    """Interval score (Winkler) KHẢ VI cho khoảng [mu-k·sigma, mu+k·sigma] ở mức cố định k.
    Phạt CẢ width LẪN miscoverage → student KHÔNG thể 'thắng' bằng cách phồng sigma (width nổ
    thì Winkler tăng). Đây là cách chặn nghiệm suy biến của CCAD v1. Trả scalar (mean over K)."""
    sigma = torch.sqrt(var + EPS)
    lo = mu - k * sigma
    hi = mu + k * sigma
    width = hi - lo
    penalty = (2.0 / alpha) * (torch.relu(lo - gt) + torch.relu(gt - hi))
    return (width + penalty).mean()


def standardized_residual(mu_S: torch.Tensor, var_S: torch.Tensor,
                          gt: torch.Tensor) -> torch.Tensor:
    """Score chuẩn hoá kiểu conformal joint: max_k |gt_k - mu_k| / sigma_k. Khả vi. Scalar."""
    sig = torch.sqrt(var_S + EPS)
    r_k = torch.abs(gt - mu_S) / sig
    return r_k.max()                            # joint (max over classes), khớp PBAwareJointConformal


def ccad_loss(mu_S_batch, var_S_batch, gt_batch, group_ids,
              min_group: int = 2) -> Dict[str, torch.Tensor]:
    """CCAD trên MỘT batch. Ép phân phối score chuẩn hoá của mỗi NHÓM (organ) ~ global.

    mu_S_batch/var_S_batch: list độ dài B, mỗi phần tử (K,). gt_batch: list (K,). group_ids: list[int/str].
    L = mean_g [ (mean_g(r) - mean_all(r))^2 + (std_g(r) - std_all(r))^2 ] trên các nhóm >= min_group mẫu.
    Khả vi qua r (r phụ thuộc mu_S, var_S của student)."""
    r = torch.stack([standardized_residual(mu_S_batch[i], var_S_batch[i], gt_batch[i])
                     for i in range(len(gt_batch))])   # (B,)
    if r.numel() < 2:
        return {"loss": r.new_zeros(()), "n_groups": 0}
    mean_all = r.mean()
    std_all = r.std(unbiased=False) + EPS
    # gom theo nhóm
    from collections import defaultdict
    idx_by_g = defaultdict(list)
    for i, g in enumerate(group_ids):
        idx_by_g[g].append(i)
    terms = []
    for g, idxs in idx_by_g.items():
        if len(idxs) < min_group:
            continue
        rg = r[torch.tensor(idxs, device=r.device)]
        mean_g = rg.mean()
        std_g = rg.std(unbiased=False)
        terms.append((mean_g - mean_all) ** 2 + (std_g - std_all) ** 2)
    if not terms:
        return {"loss": r.new_zeros(()), "n_groups": 0}
    loss = torch.stack(terms).mean()
    return {"loss": loss, "n_groups": len(terms)}
