#!/usr/bin/env python3
"""PROBE (concept-gate) cho ý "stain-perturbation uncertainty distillation".

Câu hỏi rẻ TRƯỚC khi cam kết vòng đầy đủ (chạy PathoSAM perturb + retrain):
  "Với một density-counter, phương sai count dưới xáo-trộn-nhuộm CÓ tương quan với lỗi không,
   và có mạnh hơn learned-σ sẵn có không?"
Nếu KHÔNG (corr≈0 hoặc variance mờ) -> teacher (cũng density) nhiều khả năng cũng vô dụng -> DỪNG.
Nếu CÓ -> motivate vòng teacher-uncertainty-distillation đầy đủ.

Chạy: Kaggle GPU (không cần micro_sam) — train student từ teacher_density_nuinsseg cache,
stain-TTA (HED jitter, Tellez 2019) N lần trên test -> σ_stain, so corr với lỗi + learned-σ.
"""
import os, sys, glob, pickle
import numpy as np
import torch
from skimage.color import rgb2hed, hed2rgb

REPO = os.environ.get("REPO", "/kaggle/working/repo")
sys.path.insert(0, os.path.join(REPO, "distillation_counting"))
from distill_student_r2 import DensitySigmaUNet, train, predict_r2  # noqa: E402

device = "cuda" if torch.cuda.is_available() else "cpu"

# --- teacher_density cache (train data, KHÔNG cần PathoSAM/raw) ---
cache = glob.glob("/kaggle/input/**/teacher_density_nuinsseg.pkl", recursive=True)
cache = cache[0] if cache else f"{REPO}/work/teacher_density_nuinsseg.pkl"
data = pickle.load(open(cache, "rb"))
print(f"[cache] {cache} | N={len(data)}")

# --- split train/test (probe: 1 split, 1 seed) ---
np.random.seed(0)
perm = np.random.permutation(len(data))
n_test = len(data) // 5
test_idx, train_idx = perm[:n_test].tolist(), perm[n_test:].tolist()

# --- train student (config chốt: poisson + detach_mu) ---
model = train(data, device, epochs=60, ch=32, lr=1e-3, train_idx=train_idx,
              w_density=1.0, w_count=0.01, w_nll=0.01, beta=0.5, bs=16,
              detach_mu=True, sigma_mode="poisson")


def stain_jitter(img_u8, sigma=0.08, rng=None):
    """HED-space jitter (Tellez 2019): perturb H,E channels multiplicative+additive."""
    rng = rng or np.random
    hed = rgb2hed(img_u8.astype(np.float32) / 255.0)
    a = rng.uniform(1 - sigma, 1 + sigma, 3)
    b = rng.uniform(-sigma, sigma, 3)
    for c in (0, 1):                       # Haematoxylin, Eosin
        hed[..., c] = hed[..., c] * a[c] + b[c]
    return (np.clip(hed2rgb(hed), 0, 1) * 255).astype(np.uint8)


# --- clean prediction + learned σ ---
test_data = [data[i] for i in test_idx]
clean = predict_r2(model, test_data, device)
mu_clean = np.array([p["mu"] for p in clean["preds"]])
sig_learned = np.array([p["sigma"] for p in clean["preds"]])
gt = np.array([d["gt"] for d in test_data])
err = np.abs(mu_clean - gt)

# --- stain-TTA variance ---
N = 8
mus = np.zeros((len(test_data), N))
rng = np.random.RandomState(1)
for k in range(N):
    pert = [{"img": stain_jitter(d["img"], 0.08, rng), "gt": d["gt"], "organ": d["organ"]}
            for d in test_data]
    mus[:, k] = [p["mu"] for p in predict_r2(model, pert, device)["preds"]]
sig_stain = mus.std(axis=1)


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1])


c_stain = corr(sig_stain, err)
c_learn = corr(sig_learned, err)
mag = 100 * sig_stain.mean() / max(gt.mean(), 1e-6)
print("\n===== PROBE: stain-perturbation uncertainty (student concept-gate) =====")
print(f"test N={len(test_data)} | gt mean {gt.mean():.1f} | MAE {err.mean():.2f}")
print(f"σ_stain: mean {sig_stain.mean():.2f} ({mag:.1f}% gt) | max {sig_stain.max():.1f}")
print(f"corr(σ_stain,   |err|) = {c_stain:+.3f}   <- KEY")
print(f"corr(σ_learned, |err|) = {c_learn:+.3f}   <- mốc so")
print(f"corr(σ_stain, σ_learned) = {corr(sig_stain, sig_learned):+.3f}   (thấp = bổ sung, cao = trùng)")
verdict = ("ĐÁNG đi tiếp (teacher version)" if (c_stain >= 0.30 and mag >= 5)
           else "YẾU -> nhiều khả năng KILL (ghi honest, khỏi retrain teacher)")
print(f"VERDICT probe: {verdict}")
