import os, sys, time, json
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from config_vast import (
    REPO_DIR, WORK, DATA_ROOT, CHECKPOINT_PATH, verify_env
)
print("[Vast.ai] Phase B Multi-seed Shift Calibration")
print("=" * 70)
verify_env()

from pannuke_loader import PanNukeFold, CELL_TYPES, DEFAULT_ROOT
from shift_detector import (
    extract_sam3_features, compute_mmd, compute_wasserstein, compute_energy,
    apply_hed_shift, apply_blur_shift, apply_hsv_jitter,
)
from sam3_train import make_transform, encode_image_frozen
from sam3.model_builder import build_sam3_image_model

SEEDS = [0, 42, 100, 200, 300]   
REF_SIZE = 200
TEST_SIZE = 200

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}\n")

print("Build SAM3 (frozen)...")
model = build_sam3_image_model(
    device=device, eval_mode=True,
    checkpoint_path=CHECKPOINT_PATH, load_from_HF=False,
)
model.eval()
for p in model.parameters():
    p.requires_grad = False
transform = make_transform(resolution=1008)

@torch.no_grad()
def extract_embeddings(samples_list, augment=None):
    embeddings = []
    for sample in tqdm(samples_list, desc="extract"):
        img_np = sample["image"]
        if augment is not None:
            aug_type, severity = augment
            if aug_type == "hed":
                img_np = apply_hed_shift(img_np, severity)
            elif aug_type == "blur":
                img_np = apply_blur_shift(img_np, severity)
            elif aug_type == "hsv":
                img_np = apply_hsv_jitter(img_np, severity)
        pil = Image.fromarray(img_np).convert("RGB")
        state = encode_image_frozen(model, transform, pil, device=device)
        
        feat = state.get("vision_features")
        if feat is None and "backbone_fpn" in state:
            feat = state["backbone_fpn"][-1]
        if feat is None:
            for k, v in state.items():
                if isinstance(v, torch.Tensor) and v.dim() == 4:
                    feat = v; break
        
        if feat.dim() == 4:
            feat = feat.mean(dim=[2, 3])  
        elif feat.dim() == 3:
            feat = feat.mean(dim=[1, 2])
        embeddings.append(feat.flatten().cpu().numpy())
    return np.array(embeddings)

fold1 = PanNukeFold(DATA_ROOT, 1)
fold2 = PanNukeFold(DATA_ROOT, 2)
fold3 = PanNukeFold(DATA_ROOT, 3)

results_by_seed = {}

for seed in SEEDS:
    print("\n" + "=" * 70)
    print(f"SEED {seed}")
    print("=" * 70)
    np.random.seed(seed)
    torch.manual_seed(seed)

    
    ref_idx = np.random.choice(len(fold1), REF_SIZE, replace=False)
    ref_samples = [fold1[int(i)] for i in ref_idx]
    print(f"[ref] Fold 1 N={REF_SIZE}")
    ref_emb = extract_embeddings(ref_samples)

    
    test_conditions = {
        "fold2_natural": (fold2, None),
        "fold3_natural": (fold3, None),
        "hed_mild":      (fold1, ("hed", "mild")),
        "hed_moderate":  (fold1, ("hed", "moderate")),
        "hed_severe":    (fold1, ("hed", "severe")),
        "blur_mild":     (fold1, ("blur", "mild")),
        "blur_moderate": (fold1, ("blur", "moderate")),
        "blur_severe":   (fold1, ("blur", "severe")),
        "hsv_mild":      (fold1, ("hsv", "mild")),
        "hsv_moderate":  (fold1, ("hsv", "moderate")),
        "hsv_severe":    (fold1, ("hsv", "severe")),
    }

    seed_results = {}
    for name, (fold, aug) in test_conditions.items():
        print(f"[test] {name}")
        test_idx = np.random.choice(len(fold), TEST_SIZE, replace=False)
        test_samples = [fold[int(i)] for i in test_idx]
        test_emb = extract_embeddings(test_samples, augment=aug)

        mmd_val   = compute_mmd(ref_emb, test_emb)
        wass_val  = compute_wasserstein(ref_emb, test_emb)
        energy_val= compute_energy(ref_emb, test_emb)

        seed_results[name] = {
            "mmd": float(mmd_val),
            "wasserstein": float(wass_val),
            "energy": float(energy_val),
        }
        print(f"   MMD={mmd_val:.4f}  Wass={wass_val:.4f}  Energy={energy_val:.4f}")

    results_by_seed[seed] = seed_results

print("\n" + "=" * 70)
print("AGGREGATE (mean ± std across seeds)")
print("=" * 70)

all_conditions = list(results_by_seed[SEEDS[0]].keys())
aggregate = {}
for cond in all_conditions:
    mmd_vals  = [results_by_seed[s][cond]["mmd"] for s in SEEDS]
    wass_vals = [results_by_seed[s][cond]["wasserstein"] for s in SEEDS]
    en_vals   = [results_by_seed[s][cond]["energy"] for s in SEEDS]
    aggregate[cond] = {
        "mmd_mean": float(np.mean(mmd_vals)),
        "mmd_std":  float(np.std(mmd_vals)),
        "wass_mean": float(np.mean(wass_vals)),
        "wass_std":  float(np.std(wass_vals)),
        "energy_mean": float(np.mean(en_vals)),
        "energy_std":  float(np.std(en_vals)),
    }
    print(f"  {cond:18s}: MMD={aggregate[cond]['mmd_mean']:.4f}±{aggregate[cond]['mmd_std']:.4f}")

delta_natural_mean = (aggregate["fold2_natural"]["mmd_mean"] +
                      aggregate["fold3_natural"]["mmd_mean"]) / 2
delta_severe_mean = (aggregate["blur_severe"]["mmd_mean"] +
                     aggregate["hed_severe"]["mmd_mean"]) / 2

if delta_severe_mean > 0:
    lambda_est_per_seed = []
    for s in SEEDS:
        d = (results_by_seed[s]["blur_severe"]["mmd"] +
             results_by_seed[s]["hed_severe"]["mmd"]) / 2
        if d > 0:
            lambda_est_per_seed.append(4.0 / d)
    lambda_mean = float(np.mean(lambda_est_per_seed))
    lambda_std = float(np.std(lambda_est_per_seed))
else:
    lambda_mean, lambda_std = 25.0, 0.0

print(f"\n  λ (calibrated): {lambda_mean:.2f} ± {lambda_std:.2f}")

out_path = f"{WORK}/phase_B_multiseed_results.json"
with open(out_path, "w") as f:
    json.dump({
        "config": {"seeds": SEEDS, "ref_size": REF_SIZE, "test_size": TEST_SIZE},
        "per_seed": results_by_seed,
        "aggregate": aggregate,
        "lambda_calibrated_mean": lambda_mean,
        "lambda_calibrated_std": lambda_std,
    }, f, indent=2)
print(f"\nSaved: {out_path}")
print("=" * 70)
print(f"Phase B DONE. λ = {lambda_mean:.2f} ± {lambda_std:.2f}")
