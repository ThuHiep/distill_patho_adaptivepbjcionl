from __future__ import annotations
from pathlib import Path

OUT = Path(__file__).parent / "sam3_pannuke_phaseC_jcc.ipynb"
LIB_DIR = Path(__file__).parent / "lib"

def _read(name: str) -> str:
    return (LIB_DIR / name).read_text(encoding="utf-8")

PANNUKE_LOADER = "%%writefile pannuke_loader.py\n" + _read("pannuke_loader.py")
METRICS        = "%%writefile metrics.py\n"        + _read("metrics.py")
LORA_SAM3      = "%%writefile lora_sam3.py\n"      + _read("lora_sam3.py")
SAM3_TRAIN     = "%%writefile sam3_train.py\n"     + _read("sam3_train.py")
TYPE_HEAD      = "%%writefile type_head.py\n"      + _read("type_head.py")
SHIFT_DETECTOR = "%%writefile shift_detector.py\n" + _read("shift_detector.py")
CONFORMAL      = "%%writefile conformal.py\n"      + _read("conformal.py")

def md(*lines: str) -> dict:
    src = [ln if ln.endswith("\n") else ln + "\n" for ln in lines]
    if src:
        src[-1] = src[-1].rstrip("\n")
    return {"cell_type": "markdown", "metadata": {}, "source": src}

def code(body: str) -> dict:
    lines = body.lstrip("\n").splitlines(keepends=True)
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines,
    }

cells: list[dict] = []

cells.append(md(
    "# Phase C: Joint Conformal Counting (JCC) Main Table",
    "",
    "**Goal:** Benchmark 5 conformal methods × 3 shift settings cho per-class counting.",
    "",
    "**5 Methods:**",
    "1. **Marginal split conformal** (baseline)",
    "2. **Adaptive Conformal Inference** (ACI, Gibbs-Candès 2021)",
    "3. **Shift-Aware ACI** (SA-ACI, ours novel)",
    "4. **PB-Aware Joint Conformal** (PB-JCI, ours novel)",
    "5. **Class-stratified conformal** (Bonferroni baseline)",
    "",
    "**3 Settings:**",
    "- In-distribution (Fold 3 reference, no augmentation)",
    "- Mild shift (HSV moderate)",
    "- Severe shift (blur severe)",
    "",
    "**Metrics:**",
    "- Marginal coverage per class (target 90%)",
    "- Joint coverage (PB-JCI vs others)",
    "- Average interval width per class",
    "",
    "**Prerequisites Kaggle:**",
    "- `hipinhththu/pannuke`",
    "- `hipinhththu/sam3-native-pt`",
    "- `phase-a2-lora-weights`",
    "- LoRA + TypeHead weights (Phase A3 output)",
    "",
    "**Compute budget:** ~5-6h Kaggle T4.",
))

cells.append(md("## 00 — Setup"))

cells.append(code('''
import subprocess, sys, os, platform, time, json
import torch
print("Python  :", sys.version.split()[0])
print("Torch   :", torch.__version__, "| CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU     :", torch.cuda.get_device_name(0))
'''))

cells.append(code('''
WORK = "/kaggle/working"
REPO_DIR = f"{WORK}/sam3_research"
SAM3_DIR = f"{REPO_DIR}/sam3"
CHECKPOINT_PATH = "/kaggle/input/datasets/hipinhththu/sam3-native-pt/sam3.pt"
DATA_ROOT = "/kaggle/input/datasets/hipinhththu/pannuke"

LORA_CKPT_CANDIDATES = [
    "/kaggle/input/datasets/hipinhththu/phase-a2-lora-weights/sam3_lora_rank16_final.pt",
]
TYPEHEAD_CANDIDATES = [
    "/kaggle/input/datasets/hipinhththu/phase-a3-typehead-weights/type_head_final.pt",
    "/kaggle/input/phase-a3-typehead-weights/type_head_final.pt",
    f"{WORK}/type_head_final.pt",
]
LORA_PATH = next((p for p in LORA_CKPT_CANDIDATES if os.path.exists(p)), None)
TH_PATH = next((p for p in TYPEHEAD_CANDIDATES if os.path.exists(p)), None)
assert LORA_PATH, "Khong tim thay LoRA. Da check: " + str(LORA_CKPT_CANDIDATES)
assert TH_PATH, "Khong tim thay TypeHead. Da check: " + str(TYPEHEAD_CANDIDATES)
print(f"LoRA    : {LORA_PATH}")
print(f"TypeHead: {TH_PATH}")

if not os.path.exists(REPO_DIR):
    subprocess.run(["git", "clone",
                    "https://github.com/duonguwu/sam3_research.git", REPO_DIR],
                   check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", SAM3_DIR], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "scikit-image", "scikit-learn", "opencv-python",
                "pycocotools", "einops", "tqdm"], check=True)
print("OK setup")
'''))

cells.append(md("## Helper modules"))
cells.append(code(PANNUKE_LOADER))
cells.append(code(METRICS))
cells.append(code(LORA_SAM3))
cells.append(code(SAM3_TRAIN))
cells.append(code(TYPE_HEAD))
cells.append(code(SHIFT_DETECTOR))
cells.append(code(CONFORMAL))

cells.append(code('''
import sys
for p in [".", WORK, SAM3_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from pannuke_loader import PanNukeFold, CELL_TYPES, DEFAULT_ROOT
from lora_sam3 import (inject_lora, freeze_non_lora, load_lora_state, DEFAULT_LORA_TARGETS)
from sam3_train import (make_transform, encode_image_frozen, encode_text,
                        forward_decoder_with_grad, inference_to_binary)
from type_head import (TypeHead, roi_pool_feature, extract_gt_instances,
                       per_class_counts, per_class_variance)
from shift_detector import (apply_hed_shift, apply_blur_shift, apply_hsv_jitter)
from conformal import (
    MarginalSplitConformal, AdaptiveConformalInference, ShiftAwareACI,
    PBAwareJointConformal, PBAwareJointConformalOnline, ClassStratifiedConformal,
    RollingShiftDetector, local_coverage_stats,
    coverage_per_class, joint_coverage, avg_width_per_class, macro_width,
    split_calibration_test, pb_count, pb_variance,
)
print("Helpers loaded.")
'''))

cells.append(md("## 01 — Build full pipeline (SAM3 + LoRA + TypeHead)"))

cells.append(code('''
from sam3.model_builder import build_sam3_image_model
from sam3.model.data_misc import FindStage
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Build SAM3...")
model = build_sam3_image_model(
    device=device, eval_mode=True,
    checkpoint_path=CHECKPOINT_PATH, load_from_HF=False,
)
model.eval()

LORA_R, LORA_ALPHA = 16, 32
replaced, n_lora = inject_lora(model, target_module_names=DEFAULT_LORA_TARGETS,
                                r=LORA_R, alpha=LORA_ALPHA, dropout=0.0)
load_lora_state(model, LORA_PATH)
freeze_non_lora(model)
print(f"LoRA: {len(replaced)} modules loaded.")

type_head = TypeHead(in_dim=256, hidden_dim=128, num_classes=5).to(device)
type_head.load_state_dict(torch.load(TH_PATH, map_location=device))
type_head.eval()
print(f"TypeHead loaded.")

transform = make_transform(resolution=1008)
find_stage = FindStage(
    img_ids=torch.tensor([0], device=device, dtype=torch.long),
    text_ids=torch.tensor([0], device=device, dtype=torch.long),
    input_boxes=None, input_boxes_mask=None, input_boxes_label=None,
    input_points=None, input_points_mask=None,
)
INFER_PROMPT = "cell"
SCORE_THRESH = 0.3
'''))

cells.append(md("## 02 — Inference: extract per-image (scores, probs) for conformal"))

cells.append(code('''
@torch.no_grad()
def predict_image(pil_img):
    """Return dict: {scores: (N,), probs: (N, 5), K: 5} for conformal.

    Pipeline: SAM3+LoRA -> N detections -> ROI pool -> TypeHead -> p_ik.
    """
    backbone_out = encode_image_frozen(model, transform, pil_img, device=device)

    feat = None
    if "vision_features" in backbone_out:
        feat = backbone_out["vision_features"]
    elif "backbone_fpn" in backbone_out:
        feat = backbone_out["backbone_fpn"][-1]
    else:
        for k, v in backbone_out.items():
            if isinstance(v, torch.Tensor) and v.dim() == 4:
                feat = v
                break
    if feat.dim() == 4:
        feat = feat[0]

    text_out = encode_text(model, INFER_PROMPT, device=device)
    backbone_out.update(text_out)
    geom = model._get_dummy_prompt()
    outputs = forward_decoder_with_grad(model, backbone_out, find_stage, geom)

    pred_logits = outputs["pred_logits"].float()
    pred_masks  = outputs["pred_masks"].float()
    pres_logit  = outputs["presence_logit_dec"].float()
    cls_prob = pred_logits.sigmoid()
    pres = pres_logit.sigmoid().unsqueeze(1)
    prob = (cls_prob * pres).squeeze(-1).squeeze(0)

    masks_up = F.interpolate(pred_masks, size=(256, 256), mode="bilinear",
                              align_corners=False).sigmoid().squeeze(0)
    masks_bin = (masks_up > 0.5)

    keep = prob > SCORE_THRESH
    if keep.sum() == 0:
        return {"scores": np.zeros(0), "probs": np.zeros((0, 5)), "K": 5}

    pred_masks_kept = masks_bin[keep]
    scores_kept = prob[keep].cpu().numpy()

    features = torch.zeros(len(pred_masks_kept), 256, device=device)
    for i in range(len(pred_masks_kept)):
        features[i] = roi_pool_feature(feat, pred_masks_kept[i].float())
    type_logits = type_head(features)
    type_probs = type_logits.softmax(dim=-1).cpu().numpy()

    return {
        "scores": scores_kept,
        "probs": type_probs,
        "K": 5,
    }

@torch.no_grad()
def get_gt_counts(sample):
    """Per-class GT count (5,)."""
    _, gt_classes = extract_gt_instances(sample, CELL_TYPES)
    counts = np.zeros(5, dtype=np.float32)
    for ci in range(5):
        counts[ci] = sum(1 for c in gt_classes if c == ci)
    return counts

print("Inference helpers ready.")
'''))

cells.append(md("## 03 — Prepare Fold 3 reference + 2 shift settings"))

cells.append(code('''
import numpy as np
from PIL import Image
from tqdm import tqdm

np.random.seed(42)
fold3 = PanNukeFold(DEFAULT_ROOT, 3)
print(f"Fold 3: {len(fold3)} patches")

N_SAMPLES = len(fold3)
indices = np.random.permutation(len(fold3))
print(f"N_SAMPLES = {N_SAMPLES} (full Fold 3, paper-grade)")
print(f"  -> Cal = ~{N_SAMPLES//2}, Test = ~{N_SAMPLES//2} per setting")

SETTINGS = {
    "in_dist":     {"augment": None,                       "label": "In-distribution"},
    "mild_shift":  {"augment": ("hsv", "moderate"),        "label": "Mild shift (HSV moderate)"},
    "severe_shift":{"augment": ("blur", "severe"),         "label": "Severe shift (blur severe)"},
}

def get_image_for_setting(sample, setting):
    """Apply augmentation if specified."""
    img_np = sample["image"]
    aug = SETTINGS[setting]["augment"]
    if aug is None:
        return Image.fromarray(img_np).convert("RGB")
    aug_type, severity = aug
    if aug_type == "hed":
        img_aug = apply_hed_shift(img_np, severity)
    elif aug_type == "blur":
        img_aug = apply_blur_shift(img_np, severity)
    elif aug_type == "hsv":
        img_aug = apply_hsv_jitter(img_np, severity)
    return Image.fromarray(img_aug).convert("RGB")
'''))

cells.append(md("## 04 — Run inference on all (sample × setting) combinations"))

cells.append(code('''
import pickle

predictions_by_setting = {s: [] for s in SETTINGS}
gt_counts = []
CHECKPOINT_DIR = f"{WORK}/phase_C_checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

gt_ckpt = f"{CHECKPOINT_DIR}/gt_counts.pkl"
if os.path.exists(gt_ckpt):
    with open(gt_ckpt, "rb") as f:
        gt_counts = pickle.load(f)
    print(f"Resumed GT counts: {len(gt_counts)} samples loaded from checkpoint")
else:
    print("GT cache miss -> compute GT counts...")
    for idx in tqdm(indices, desc="GT counts"):
        gt_counts.append(get_gt_counts(fold3[int(idx)]))
    with open(gt_ckpt, "wb") as f:
        pickle.dump(gt_counts, f)
    print(f"Saved GT checkpoint: {gt_ckpt}")

gt_counts = np.array(gt_counts)
print(f"GT counts shape: {gt_counts.shape}")
print(f"Mean count per class: {gt_counts.mean(axis=0)}")

t0_total = time.time()
for setting in SETTINGS:
    pkl = f"{CHECKPOINT_DIR}/preds_{setting}.pkl"
    if os.path.exists(pkl):
        with open(pkl, "rb") as f:
            predictions_by_setting[setting] = pickle.load(f)
        print(f"[RESUME] {setting}: {len(predictions_by_setting[setting])} preds loaded")
        continue

    print(f"\\n[INFER] setting={setting} | N={N_SAMPLES} patches")
    t0 = time.time()
    preds = []
    for idx in tqdm(indices, desc=f"  {setting}"):
        sample = fold3[int(idx)]
        pil = get_image_for_setting(sample, setting)
        preds.append(predict_image(pil))
    predictions_by_setting[setting] = preds

    with open(pkl, "wb") as f:
        pickle.dump(preds, f)
    t_set = time.time() - t0
    print(f"  Done {setting} in {t_set/60:.1f} min ({t_set/N_SAMPLES:.2f}s/patch)")
    print(f"  Checkpoint saved: {pkl}")

elapsed = time.time() - t0_total
print(f"\\nTotal inference: {elapsed/60:.1f} min for {N_SAMPLES} samples x {len(SETTINGS)} settings")
'''))

cells.append(code('''
import pickle
with open(f"{WORK}/phase_C_predictions.pkl", "wb") as f:
    pickle.dump({
        "predictions_by_setting": predictions_by_setting,
        "gt_counts": gt_counts,
        "indices": indices,
        "settings": SETTINGS,
    }, f)
print(f"Saved predictions: {WORK}/phase_C_predictions.pkl")
'''))

cells.append(md("## 05 — Benchmark 5 conformal methods × 3 settings"))

cells.append(code('''
ALPHA = 0.1
GAMMA_0 = 0.05
LAMBDA = 3.0
GAMMA_MAX = 0.15
DETECTOR_WINDOW = 100
PBJCI_WINDOW = 300
LOCAL_WINDOW = 100

EVAL_SETTINGS = ["in_dist", "mild_shift", "severe_shift", "temporal_drift"]
METHODS = ["marginal_split", "aci", "sa_aci", "pb_jci", "pb_jci_online", "class_strat"]

def get_nonconformity_scores(preds, gt_list):
    scores = []
    for p, gt in zip(preds, gt_list):
        if len(p["scores"]) == 0:
            scores.append(float(abs(gt).max()))
            continue
        n_p = pb_count(p["scores"], p["probs"])
        sigma = np.sqrt(pb_variance(p["scores"], p["probs"]) + 1e-6)
        S = max(abs(gt[k] - n_p[k]) / sigma[k] for k in range(5))
        scores.append(S)
    return np.array(scores)

def _interval(p, q, K=5):
    if len(p["scores"]) == 0:
        return np.zeros(K), np.zeros(K)
    n_p = pb_count(p["scores"], p["probs"])
    sigma = np.sqrt(pb_variance(p["scores"], p["probs"]) + 1e-6)
    return np.maximum(0, n_p - q * sigma), n_p + q * sigma

def _score_one(p, gt, K=5):
    if len(p["scores"]) == 0:
        return float(abs(gt).max())
    n_p = pb_count(p["scores"], p["probs"])
    sigma = np.sqrt(pb_variance(p["scores"], p["probs"]) + 1e-6)
    return max(abs(gt[k] - n_p[k]) / sigma[k] for k in range(K))

def _summary(los, his, covered_list, gt_arr, online=False):
    cov_pc = coverage_per_class(los, his, gt_arr)
    width = avg_width_per_class(los, his)
    jc = float(np.mean(covered_list)) if online else joint_coverage(los, his, gt_arr)
    loc = local_coverage_stats(covered_list, window=LOCAL_WINDOW)
    return {"cov_per_class": cov_pc.tolist(),
            "marginal_coverage": float(cov_pc.mean()),
            "joint_coverage": jc,
            "width_per_class": width.tolist(),
            "macro_width": float(width.mean()),
            **loc}

def eval_static_method(method, test_preds, test_gt):
    los, his = [], []
    for p in test_preds:
        p["K"] = 5
        lo, hi = method.predict_interval(p)
        los.append(lo); his.append(hi)
    los = np.array(los); his = np.array(his); gt_arr = np.array(test_gt)
    covered_list = ((gt_arr >= los) & (gt_arr <= his)).all(axis=1).tolist()
    return _summary(los, his, covered_list, gt_arr, online=False)

def eval_aci_method(method, test_preds, test_gt, cal_scores, detector=None):
    method.reset()
    method.history_scores = list(cal_scores)
    los_list, his_list, covered_list = [], [], []
    K = 5
    for p, gt in zip(test_preds, test_gt):
        q = method.get_quantile()
        lo, hi = _interval(p, q, K)
        los_list.append(lo); his_list.append(hi)
        covered = bool(((gt >= lo) & (gt <= hi)).all())
        covered_list.append(covered)
        S_t = _score_one(p, gt, K)
        if isinstance(method, ShiftAwareACI):
            delta = detector.step(S_t) if detector is not None else 0.0
            method.update(S_t, covered, delta_t=delta)
        else:
            method.update(S_t, covered)
    los = np.array(los_list); his = np.array(his_list); gt_arr = np.array(test_gt)
    return _summary(los, his, covered_list, gt_arr, online=True)

def eval_online_window(method, test_preds, test_gt, cal_scores):
    method.warmstart(cal_scores)
    los_list, his_list, covered_list = [], [], []
    K = 5
    for p, gt in zip(test_preds, test_gt):
        q = method.get_quantile()
        lo, hi = _interval(p, q, K)
        los_list.append(lo); his_list.append(hi)
        covered = bool(((gt >= lo) & (gt <= hi)).all())
        covered_list.append(covered)
        method.update(_score_one(p, gt, K))
    los = np.array(los_list); his = np.array(his_list); gt_arr = np.array(test_gt)
    return _summary(los, his, covered_list, gt_arr, online=True)

def make_split(n, cal_ratio, seed):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    n_cal = int(n * cal_ratio)
    return idx[:n_cal], idx[n_cal:]

def run_benchmark(cal_seed, verbose=False):
    n = len(gt_counts)
    cal_idx, test_idx = make_split(n, 0.5, cal_seed)

    cal_preds = [predictions_by_setting["in_dist"][i] for i in cal_idx]
    cal_gt    = [gt_counts[i] for i in cal_idx]
    test_gt   = [gt_counts[i] for i in test_idx]
    test_preds = {s: [predictions_by_setting[s][i] for i in test_idx] for s in SETTINGS}

    cal_scores = get_nonconformity_scores(cal_preds, cal_gt)

    third = len(test_idx) // 3
    drift_preds = (test_preds["in_dist"][:third]
                   + test_preds["mild_shift"][third:2 * third]
                   + test_preds["severe_shift"][2 * third:])
    eval_streams = {
        "in_dist":        (test_preds["in_dist"],      test_gt),
        "mild_shift":     (test_preds["mild_shift"],    test_gt),
        "severe_shift":   (test_preds["severe_shift"],  test_gt),
        "temporal_drift": (drift_preds,                 test_gt),
    }

    msc = MarginalSplitConformal(alpha=ALPHA).fit(cal_preds, cal_gt)
    pb_jci = PBAwareJointConformal(alpha=ALPHA).fit(cal_preds, cal_gt)
    csc = ClassStratifiedConformal(alpha=ALPHA, bonferroni=True).fit(cal_preds, cal_gt)

    res = {s: {} for s in eval_streams}
    for setting, (tp, tg) in eval_streams.items():
        res[setting]["marginal_split"] = eval_static_method(msc, tp, tg)
        res[setting]["pb_jci"] = eval_static_method(pb_jci, tp, tg)
        res[setting]["class_strat"] = eval_static_method(csc, tp, tg)

        aci = AdaptiveConformalInference(alpha_target=ALPHA, gamma=GAMMA_0)
        res[setting]["aci"] = eval_aci_method(aci, tp, tg, cal_scores)

        sa_aci = ShiftAwareACI(alpha_target=ALPHA, gamma_0=GAMMA_0,
                               lambda_=LAMBDA, gamma_max=GAMMA_MAX)
        detector = RollingShiftDetector(window=DETECTOR_WINDOW).fit_baseline(cal_scores)
        res[setting]["sa_aci"] = eval_aci_method(sa_aci, tp, tg, cal_scores, detector=detector)

        pbo = PBAwareJointConformalOnline(alpha=ALPHA, window=PBJCI_WINDOW)
        res[setting]["pb_jci_online"] = eval_online_window(pbo, tp, tg, cal_scores)

        if verbose:
            print(f"\\n=== {setting} ===")
            for m in METHODS:
                r = res[setting][m]
                print(f"  {m:16s}: marg={r['marginal_coverage']:.3f} "
                      f"joint={r['joint_coverage']:.3f} width={r['macro_width']:7.2f} "
                      f"minLocal={r['min_local_cov']:.3f} missRun={r['max_miss_run']}")
    return res, len(cal_idx), len(test_idx)

results, n_cal, n_test = run_benchmark(cal_seed=42, verbose=True)
print(f"\\nSingle-seed verify done. Cal={n_cal}, Test={n_test}")
'''))

cells.append(md("## 06 — Main table for paper Section 4.3"))

cells.append(code('''
method_names = {
    "marginal_split": "Marginal Split",
    "aci": "ACI (Gibbs-Candes)",
    "sa_aci": "SA-ACI (Ours)",
    "pb_jci": "PB-Aware JCI (Ours)",
    "pb_jci_online": "PB-JCI Online (Ours)",
    "class_strat": "Class-Strat Bonf",
}

print("=" * 116)
print(f"PHASE C MAIN TABLE (single seed=42) | N_test={n_test}, alpha={ALPHA}")
print("=" * 116)
hdr = (f"{'Setting':<15s} | {'Method':<21s} | {'MargCov':>8s} | {'JointCov':>8s} | "
       f"{'Width':>8s} | {'MinLocal':>8s} | {'MissRun':>7s}")
print("\\n" + hdr)
print("-" * 116)
for setting in EVAL_SETTINGS:
    for m in METHODS:
        r = results[setting][m]
        print(f"{setting:<15s} | {method_names[m]:<21s} | "
              f"{r['marginal_coverage']*100:>7.1f}% | {r['joint_coverage']*100:>7.1f}% | "
              f"{r['macro_width']:>8.2f} | {r['min_local_cov']*100:>7.1f}% | "
              f"{r['max_miss_run']:>7d}")
    print("-" * 116)

with open(f"{WORK}/phase_C_results.json", "w") as f:
    json.dump({
        "config": {"alpha": ALPHA, "gamma_0": GAMMA_0, "lambda": LAMBDA,
                   "gamma_max": GAMMA_MAX, "n_cal": n_cal, "n_test": n_test,
                   "eval_settings": EVAL_SETTINGS, "methods": METHODS},
        "results": results,
    }, f, indent=2)
print(f"\\nSaved: {WORK}/phase_C_results.json")
'''))

cells.append(md(
    "## 07 — Cal-seed multi-seed (free, reuses cached predictions)",
    "",
    "Re-run the conformal benchmark over 5 calibration splits to get mean ± std.",
    "No GPU inference here — only the cal/test split changes. Model-seed CI added",
    "later after Vast multi-seed A2/A3.",
))

cells.append(code('''
CAL_SEEDS = [42, 100, 200, 300, 400]
multi = {s: {m: {"marginal_coverage": [], "joint_coverage": [],
                  "macro_width": [], "min_local_cov": [], "max_miss_run": []}
              for m in METHODS} for s in EVAL_SETTINGS}

for sd in CAL_SEEDS:
    res_sd, _, _ = run_benchmark(cal_seed=sd, verbose=False)
    for s in EVAL_SETTINGS:
        for m in METHODS:
            for key in multi[s][m]:
                multi[s][m][key].append(res_sd[s][m][key])
    print(f"  seed {sd} done")

def ms(vals):
    a = np.asarray(vals, dtype=float)
    return float(a.mean()), float(a.std())

print("\\n" + "=" * 120)
print(f"PHASE C MULTI-SEED (cal seeds={CAL_SEEDS}) | mean +/- std")
print("=" * 120)
hdr = (f"{'Setting':<15s} | {'Method':<21s} | {'MargCov':>14s} | {'JointCov':>14s} | "
       f"{'Width':>15s} | {'MinLocal':>14s}")
print("\\n" + hdr)
print("-" * 120)
agg = {s: {} for s in EVAL_SETTINGS}
for s in EVAL_SETTINGS:
    for m in METHODS:
        mc_m, mc_s = ms(multi[s][m]["marginal_coverage"])
        jc_m, jc_s = ms(multi[s][m]["joint_coverage"])
        w_m, w_s   = ms(multi[s][m]["macro_width"])
        ml_m, ml_s = ms(multi[s][m]["min_local_cov"])
        agg[s][m] = {"marg": [mc_m, mc_s], "joint": [jc_m, jc_s],
                     "width": [w_m, w_s], "min_local": [ml_m, ml_s]}
        print(f"{s:<15s} | {method_names[m]:<21s} | "
              f"{mc_m*100:>6.1f}+/-{mc_s*100:>4.1f}% | "
              f"{jc_m*100:>6.1f}+/-{jc_s*100:>4.1f}% | "
              f"{w_m:>8.2f}+/-{w_s:>5.2f} | "
              f"{ml_m*100:>6.1f}+/-{ml_s*100:>4.1f}%")
    print("-" * 120)

with open(f"{WORK}/phase_C_multiseed_results.json", "w") as f:
    json.dump({"config": {"cal_seeds": CAL_SEEDS, "alpha": ALPHA,
                          "eval_settings": EVAL_SETTINGS, "methods": METHODS},
               "raw": multi, "aggregate": agg}, f, indent=2)
print(f"\\nSaved: {WORK}/phase_C_multiseed_results.json")
'''))

cells.append(md(
    "## Phase C PASS criteria",
    "",
    "**In-distribution (efficiency story):**",
    "- PB-JCI joint coverage ≥ 88% (target 90%) with width < ACI width (tighter)",
    "- PB-JCI width < Class-Strat Bonferroni width × 0.85",
    "",
    "**Under static shift (mild/severe):**",
    "- PB-JCI (split) coverage drops — expected, motivates adaptive methods",
    "- PB-JCI Online + ACI maintain joint coverage ≥ 80%",
    "- SA-ACI no longer pathological: width within ~2× of ACI (not 3-5×)",
    "",
    "**Temporal drift (SA-ACI showcase):**",
    "- SA-ACI min-local-coverage > ACI (faster recovery at change points)",
    "- SA-ACI max-miss-run < ACI (fewer consecutive misses)",
    "",
    "**Outputs:**",
    "- `phase_C_results.json` — single-seed benchmark (4 settings × 6 methods)",
    "- `phase_C_multiseed_results.json` — cal-seed mean ± std",
    "- `phase_C_predictions.pkl` — cached predictions (inference reused on re-run)",
))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

import json
with OUT.open("w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"Wrote {OUT.name}: {len(cells)} cells")
