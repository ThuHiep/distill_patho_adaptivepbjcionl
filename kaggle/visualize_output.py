"""
Visualize đầu ra của Adaptive PB-JCI Online cho MỘT ảnh — kiểu figure paper.

Hình 2 panel:
  TRÁI : ảnh mô bệnh học + chấm tâm mỗi nhân tế bào, TÔ MÀU THEO LOẠI.
  PHẢI : mỗi loại một dòng — điểm = số đoán E[N_k], thanh = khoảng [l_k,u_k],
         kim cương = số thật; xanh = phủ đúng, đỏ = trượt.
  Nhãn loại ở panel phải tô cùng màu với chấm panel trái → liên kết hai bên.

Cách dùng:
  (1) `python visualize_output.py`               → demo tổng hợp (không cần dữ liệu).
  (2) Trên KAGGLE (có PanNuke + model):
        from visualize_output import demo_pannuke
        demo_pannuke(fold=3, idx=0, pred=pred_dict, q_hat=q)   # pred thật
      hoặc bỏ pred → chỉ overlay + số thật (chưa vẽ khoảng).
"""
from __future__ import annotations
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")  # lưu ảnh, không cần màn hình
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ---- conformal helpers: lib nếu có, không thì inline (chạy độc lập) ----
try:
    from lib.conformal import pb_count, pb_variance, empirical_quantile
except Exception:
    def empirical_quantile(scores, alpha):
        n = len(scores)
        if n == 0:
            return float("inf")
        level = np.ceil((n + 1) * (1 - alpha)) / n
        return float(np.quantile(scores, min(level, 1.0), method="higher"))

    def pb_count(scores, probs):
        return (scores[:, None] * probs).sum(axis=0)

    def pb_variance(scores, probs):
        w = scores[:, None] * probs
        return (w * (1.0 - w)).sum(axis=0)

# Khớp pannuke_loader.CELL_TYPES (thứ tự kênh 0..4)
PANNUKE_CLASSES = ["Tân sinh (u)", "Viêm", "Mô liên kết", "Chết", "Biểu mô"]
# Màu theo loại — dùng CHUNG cho chấm overlay (trái) và nhãn dòng (phải)
CLASS_COLORS = ["#e41a1c", "#4daf4a", "#377eb8", "#000000", "#ff7f00"]


# ============================ TÍNH KHOẢNG ============================
def counts_to_interval(pred, q_hat):
    """pred={'scores':[n],'probs':[n,K]} → (E, sigma, lower, upper)."""
    s, p = np.asarray(pred["scores"]), np.asarray(pred["probs"])
    if len(s) == 0:
        K = p.shape[1] if p.ndim == 2 else len(PANNUKE_CLASSES)
        z = np.zeros(K)
        return z, z, z, z
    E = pb_count(s, p)
    sigma = np.sqrt(pb_variance(s, p) + 1e-6)
    lower = np.maximum(0, E - q_hat * sigma)
    upper = E + q_hat * sigma
    return E, sigma, lower, upper


# ====================== PANEL TRÁI: OVERLAY NHÂN ======================
def _instance_centroids(channel):
    """channel: instance-map 2D (int). Trả về list (row, col) tâm mỗi nhân."""
    ids = np.unique(channel)
    ids = ids[ids > 0]
    cents = []
    for i in ids:
        ys, xs = np.where(channel == i)
        cents.append((ys.mean(), xs.mean()))
    return cents


def overlay_nuclei(image, masks_per_type, ax, class_names=None, draw_legend=True):
    """
    image          : HxWx3
    masks_per_type : (5, H, W) instance-map mỗi loại (như pannuke_loader trả về)
    Vẽ ảnh + chấm tâm mỗi nhân, màu theo loại.
    """
    names = class_names or PANNUKE_CLASSES[:len(masks_per_type)]
    ax.imshow(image)
    for k, ch in enumerate(masks_per_type):
        cents = _instance_centroids(ch)
        if cents:
            ys, xs = zip(*cents)
            ax.scatter(xs, ys, s=42, c=CLASS_COLORS[k], edgecolors="white",
                       linewidths=0.8, label=f"{names[k]} ({len(cents)})", zorder=3)
    ax.set_title("Ảnh + tâm nhân (màu theo loại)")
    ax.axis("off")
    if draw_legend:
        # đặt chú thích DƯỚI ảnh (ngang), không che góc ảnh
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02),
                  ncol=3, fontsize=8, framealpha=0.9, title="Loại (số đếm thật)")


# ====================== PANEL PHẢI: KHOẢNG ======================
def _interval_panel(ax, E, lower, upper, gt, names, title):
    K = len(E)
    covered = (gt >= lower) & (gt <= upper)
    y = np.arange(K)[::-1]
    umax = max(float(np.max(upper)), 1.0)
    for i, yi in zip(range(K), y):
        c = "#2ca02c" if covered[i] else "#d62728"
        ax.plot([lower[i], upper[i]], [yi, yi], color=c, lw=7, alpha=0.35,
                solid_capstyle="round")
        ax.plot(E[i], yi, "o", color=c, ms=9)
        ax.plot(gt[i], yi, "D", color="black", ms=8, mfc="white", mew=1.6)
        ax.text(upper[i] + 0.012 * umax, yi, f"[{lower[i]:.0f}, {upper[i]:.0f}]",
                va="center", fontsize=9, color=c)
    ax.set_xlim(-0.02 * umax, umax * 1.42)   # chua chat de text [..] khong tran ra ngoai
    ax.set_yticks(y)
    labels = ax.set_yticklabels(names)
    for i, lab in zip(range(K), labels):          # nhãn tô màu theo loại
        lab.set_color(CLASS_COLORS[i])
        lab.set_fontweight("bold")
    ax.set_xlabel("Số tế bào")
    cov_all = "ĐÚNG cả lớp" if covered.all() else "có lớp TRƯỢT"
    ax.set_title(f"{title}\n(joint coverage: {cov_all})")
    ax.legend(handles=[
        Line2D([0], [0], marker="o", color="gray", ls="", ms=9, label="số đoán E[N]"),
        Line2D([0], [0], marker="D", color="black", ls="", ms=8, mfc="white",
               label="thật"),
    ], loc="lower right", fontsize=9)
    ax.grid(axis="x", alpha=0.25)
    ax.margins(y=0.15)
    return covered


# ====================== HÀM CHÍNH ======================
def visualize_prediction(pred, gt, q_hat, class_names=None, image=None,
                         masks_per_type=None, title="Đầu ra cho 1 ảnh",
                         save_path=None):
    """
    pred           : {'scores':[n],'probs':[n,K]}  (None → chỉ vẽ số thật)
    gt             : [K] số đếm thật
    image          : (tuỳ chọn) ảnh để vẽ panel trái
    masks_per_type : (tuỳ chọn) (5,H,W) để overlay chấm nhân
    """
    gt = np.asarray(gt, dtype=float)
    K = len(gt)
    names = class_names or PANNUKE_CLASSES[:K]

    if pred is not None:
        E, sigma, lower, upper = counts_to_interval(pred, q_hat)
    else:  # không có pred → khoảng = điểm (chỉ minh hoạ số thật)
        E = lower = upper = gt.copy()

    has_left = image is not None
    fig, axes = plt.subplots(1, 2 if has_left else 1,
                             figsize=(13 if has_left else 7, 5.2))
    axes = np.atleast_1d(axes)
    if has_left:
        if masks_per_type is not None:
            overlay_nuclei(image, masks_per_type, axes[0], names)
        else:
            axes[0].imshow(image); axes[0].set_title("Ảnh"); axes[0].axis("off")

    covered = _interval_panel(axes[-1], E, lower, upper, gt, names, title)

    fig.text(0.5, -0.02,
             f"Tổng (cộng lại): đoán {E.sum():.1f}  |  thật {gt.sum():.0f}"
             f"   — đầu ra chính là {K} khoảng theo loại; tổng chỉ là dòng phụ",
             ha="center", fontsize=9, style="italic")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=140, bbox_inches="tight")
        print(f"Đã lưu: {save_path}")
    plt.close(fig)
    return E, lower, upper, covered


# ====================== DEMO KAGGLE (dữ liệu thật) ======================
def demo_pannuke(fold=3, idx=0, pred=None, q_hat=1.8, root=None,
                 save_path="pannuke_fig.png"):
    """
    Chạy trên Kaggle: load 1 ảnh PanNuke thật (image + masks + GT counts),
    overlay nhân, và vẽ khoảng nếu có `pred` (từ model). Không pred → chỉ overlay+GT.
    """
    sys.path.insert(0, "lib")
    from pannuke_loader import PanNukeFold, DEFAULT_ROOT, CELL_TYPES
    fold_obj = PanNukeFold(root or DEFAULT_ROOT, fold)
    s = fold_obj[idx]
    print(f"[Fold {fold} idx {idx}] tissue={s['tissue']}  GT counts={s['counts']}")
    return visualize_prediction(
        pred=pred, gt=s["counts"], q_hat=q_hat,
        image=s["image"], masks_per_type=s["masks"],
        title=f"PanNuke Fold {fold} #{idx} — Adaptive PB-JCI Online",
        save_path=save_path,
    )


# ====================== DEMO TỔNG HỢP (không cần dữ liệu) ======================
def _demo():
    rng = np.random.RandomState(7)
    n = 35
    scores = rng.uniform(0.55, 0.98, size=n)
    logits = rng.randn(n, 5) * 1.3
    logits[:, 0] += 1.0
    probs = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    pred = {"scores": scores, "probs": probs}
    q_hat = 1.8
    E = pb_count(scores, probs)
    gt = np.clip(np.round(E + rng.randn(5) * 1.5), 0, None).astype(int)
    gt[1] = int(E[1] + 6)  # ép 1 lớp trượt
    print("E[N] (đoán):", np.round(E, 1), " GT:", gt)
    visualize_prediction(pred, gt, q_hat,
                         title="DEMO (tổng hợp) — Adaptive PB-JCI Online",
                         save_path="../figures/demo_output.png")


if __name__ == "__main__":
    _demo()
