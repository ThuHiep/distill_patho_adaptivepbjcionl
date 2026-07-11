"""
r2_losses.py — Loss cho Paper 2 / thiết kế R2: Distributional Count Distillation.

Ý tưởng (xem PHAN_TICH_LOI_va_THIET_KE_LAI.md mục 6): student xuất TRỰC TIẾP (mu, log_sigma)
cho MỖI ảnh; train/eval CÙNG đo (mu, sigma) -> aligned. Bỏ chuỗi "instance -> PB -> conformal
sinh sigma" (nguồn misalignment ở 3 vòng PBUD/CCAD trước).

mu   = sum(density_map_student)                 # count suy luận (density-sum, bền hơn threshold+CC)
sigma= exp(log_sigma_head)                        # độ bất định đếm/ảnh, HỌC ĐƯỢC, heteroscedastic

Ba số hạng (mọi thứ ALIGNED với cái eval chấm):
  L_density : MSE(density_S, density_T)           # KD: học ĐẾM từ teacher (mean), density-map chuẩn
  L_count   : |sum(density_S) - GT|               # ép count suy luận đúng (trực tiếp, per-image)
  L_nll     : Gaussian NLL(GT | mu, sigma)        # ★ học sigma CALIBRATED (lớn ở organ khó)

CƠ SỞ (không heuristic):
  - Density-map counting: Lempitsky & Zisserman, NeurIPS 2010 (sum density = count).
  - Heteroscedastic aleatoric NLL: Kendall & Gal, NeurIPS 2017.
  - beta-NLL (chống lỗi gradient của Gaussian NLL khi sigma nhỏ): Seitzer et al., ICLR 2022
    "On the Pitfalls of Heteroscedastic Uncertainty Estimation". beta=0.5 khuyến nghị.
  - Poisson count NLL (tùy chọn, equidispersion): so sánh ablation.

TẤT CẢ khả vi. Test local: test_r2_losses.py.
"""
from __future__ import annotations
from typing import Dict
import torch
import torch.nn.functional as F

EPS = 1e-6


def count_from_density(density: torch.Tensor) -> torch.Tensor:
    """density:(B,1,H,W) hoặc (B,H,W) -> count mu:(B,) = tổng theo không gian."""
    if density.dim() == 4:
        return density.sum(dim=(1, 2, 3))
    if density.dim() == 3:
        return density.sum(dim=(1, 2))
    raise ValueError(f"density dim {density.dim()} không hỗ trợ")


def density_kd_loss(density_S: torch.Tensor, density_T: torch.Tensor) -> torch.Tensor:
    """KD mean: MSE per-pixel giữa density student và teacher. Cả hai (B,1,H,W)."""
    return F.mse_loss(density_S, density_T)


def count_loss(mu: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """|mu - GT| trung bình batch. mu,gt:(B,)."""
    return (mu - gt).abs().mean()


def gaussian_nll(mu: torch.Tensor, log_sigma: torch.Tensor, gt: torch.Tensor,
                 beta: float = 0.5, detach_mu: bool = False) -> torch.Tensor:
    """Gaussian NLL heteroscedastic cho count, có beta-weighting (Seitzer 2022).

    NLL_i = 0.5*[ (gt-mu)^2 / sigma^2 + 2*log_sigma ]   (bỏ hằng 0.5*log(2pi))
    beta-NLL: nhân mỗi số hạng với detach(sigma^(2*beta)) để gradient của mu KHÔNG bị sigma^2
    đè (lỗi kinh điển của Gaussian NLL). beta=0 -> NLL thường; beta=1 -> ~MSE có trọng số.

    detach_mu=True: tách mu khỏi NLL (dùng mu.detach() trong số hạng lỗi) -> NLL CHỈ dạy log_sigma,
    mu do count/density loss sở hữu. Ablation cho thấy NLL-coupling làm hỏng MAE; tách ra để σ học
    khớp |GT-mu| mà không kéo lệch mu (chuẩn khi tách mean/variance head).
    mu,log_sigma,gt:(B,). Trả scalar."""
    mu_err = mu.detach() if detach_mu else mu
    var = torch.exp(2.0 * log_sigma)
    nll = 0.5 * ((gt - mu_err) ** 2 / (var + EPS) + 2.0 * log_sigma)
    if beta > 0.0:
        w = var.detach() ** beta
        nll = nll * w
    return nll.mean()


def poisson_nll(mu: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Poisson NLL (equidispersion: var=mu). Dùng cho ABLATION so với Gaussian heteroscedastic.
    lambda=mu>0. NLL = mu - gt*log(mu) (bỏ log(gt!)). mu,gt:(B,). Trả scalar."""
    mu = torch.clamp(mu, min=EPS)
    return (mu - gt * torch.log(mu)).mean()


def r2_loss(density_S: torch.Tensor, density_T: torch.Tensor,
            log_sigma: torch.Tensor, gt: torch.Tensor,
            w_density: float = 1.0, w_count: float = 1.0, w_nll: float = 1.0,
            beta: float = 0.5, detach_mu: bool = False) -> Dict[str, torch.Tensor]:
    """Loss R2 tổng cho MỘT batch.
    density_S/density_T:(B,1,H,W); log_sigma,gt:(B,).
    detach_mu: tách mu khỏi NLL (NLL chỉ dạy sigma) — sửa NLL-coupling làm hỏng MAE (ablation).
    Trả dict: 'loss' + các thành phần (detach) để log/ablation."""
    mu = count_from_density(density_S)                 # (B,)
    L_density = density_kd_loss(density_S, density_T)
    L_count = count_loss(mu, gt)
    L_nll = gaussian_nll(mu, log_sigma, gt, beta=beta, detach_mu=detach_mu)
    loss = w_density * L_density + w_count * L_count + w_nll * L_nll
    return {"loss": loss,
            "L_density": L_density.detach(), "L_count": L_count.detach(),
            "L_nll": L_nll.detach(),
            "mu": mu.detach(), "sigma": torch.exp(log_sigma).detach()}
