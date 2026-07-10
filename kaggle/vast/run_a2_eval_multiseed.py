import os, sys, time, json
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from config_vast import WORK, DATA_ROOT, CHECKPOINT_PATH, CHECKPOINTS_OUT, verify_env
print("[Vast.ai] A2 Eval Multi-seed (full Fold 3)")
print("=" * 70)
verify_env()

from pannuke_loader import PanNukeFold, CELL_TYPES, DEFAULT_ROOT
from metrics import ClassWiseAccumulator, PerPromptClassAccumulator
from lora_sam3 import inject_lora, freeze_non_lora, load_lora_state, DEFAULT_LORA_TARGETS
from sam3_train import (make_transform, encode_image_frozen, encode_text,
                         forward_decoder_with_grad, inference_to_binary)
from sam3.model_builder import build_sam3_image_model
from sam3.model.data_misc import FindStage

SEEDS = [42, 100, 200]
SCORE_THRESH = 0.3

PROMPTS_MEDICAL = {
    "Neoplastic":   ["histopathology image of neoplastic tissue"],
    "Inflammatory": ["histopathology image of inflammatory tissue"],
    "Connective":   ["histopathology image of connective tissue"],
    "Dead":         ["histopathology image of dead tissue"],
    "Epithelial":   ["histopathology image of epithelial tissue"],
}
PROMPTS_LLM = {
    "Neoplastic":   ["Neoplastic cell", "Tumor cell", "Cancer cell", "Malignant cell"],
    "Inflammatory": ["Inflammatory cell", "Lymphocyte", "Immune cell", "Leukocyte"],
    "Connective":   ["Connective tissue cell", "Fibroblast", "Stromal cell"],
    "Dead":         ["Dead cell", "Apoptotic cell", "Necrotic cell"],
    "Epithelial":   ["Epithelial cell", "Epithelium", "Lining cell",
                      "Surface cell", "Mucosal cell nucleus"],
}
PROMPT_GENERIC = "cell"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}\n")

fold3 = PanNukeFold(DATA_ROOT, 3)
print(f"Fold 3: {len(fold3)} patches\n")

all_results = {}
for seed in SEEDS:
    print("=" * 70)
    print(f"SEED {seed} — eval start")
    print("=" * 70)

    lora_ck = f"{CHECKPOINTS_OUT}/sam3_lora_seed{seed}_final.pt"
    assert os.path.exists(lora_ck), f"Missing: {lora_ck}"

    model = build_sam3_image_model(
        device=device, eval_mode=True,
        checkpoint_path=CHECKPOINT_PATH, load_from_HF=False,
    )
    model.eval()
    inject_lora(model, target_module_names=DEFAULT_LORA_TARGETS,
                r=16, alpha=32, dropout=0.0, path_must_contain="decoder")
    load_lora_state(model, lora_ck)
    for p in model.parameters():
        p.requires_grad = False

    transform = make_transform(resolution=1008)
    find_stage = FindStage(
        img_ids=torch.tensor([0], device=device, dtype=torch.long),
        text_ids=torch.tensor([0], device=device, dtype=torch.long),
        input_boxes=None, input_boxes_mask=None, input_boxes_label=None,
        input_points=None, input_points_mask=None,
    )

    @torch.no_grad()
    def encode_cached(pil): return encode_image_frozen(model, transform, pil, device=device)

    @torch.no_grad()
    def predict(state, prompt):
        st = dict(state)
        text_out = encode_text(model, prompt, device=device)
        st.update(text_out)
        outputs = forward_decoder_with_grad(model, st, find_stage, model._get_dummy_prompt())
        pm = inference_to_binary(outputs, target_hw=(256, 256), score_threshold=SCORE_THRESH)
        return pm.cpu().numpy().astype(bool)

    acc_med = ClassWiseAccumulator(CELL_TYPES)
    acc_llm = PerPromptClassAccumulator(CELL_TYPES, PROMPTS_LLM)
    acc_gen = ClassWiseAccumulator(CELL_TYPES)

    t0 = time.time()
    for i in tqdm(range(len(fold3)), desc=f"Seed{seed} eval"):
        sample = fold3[i]
        pil = Image.fromarray(sample["image"]).convert("RGB")
        gt = {c: (sample["masks"][CELL_TYPES.index(c)] > 0) for c in CELL_TYPES}
        state = encode_cached(pil)

        pred_gen = predict(state, PROMPT_GENERIC)
        for c in CELL_TYPES: acc_gen.update(pred_gen, gt[c], c)
        for c in CELL_TYPES:
            pred_m = predict(state, PROMPTS_MEDICAL[c][0])
            acc_med.update(pred_m, gt[c], c)
        for c, prompts in PROMPTS_LLM.items():
            for p in prompts:
                pred_l = predict(state, p)
                acc_llm.update(pred_l, gt[c], c, p)

    elapsed = time.time() - t0
    res = {
        "Medical": acc_med.summary(),
        "LLM": acc_llm.summary(),
        "Generic": acc_gen.summary(),
    }
    print(f"\nSeed {seed} eval done in {elapsed/60:.1f}min")
    for name, r in res.items():
        print(f"  {name:8s}: mIoU={r['mIoU']*100:.2f}% Dice={r.get('Dice', 0)*100:.2f}%")
    all_results[seed] = res

    del model
    torch.cuda.empty_cache()

print(f"\n{'='*70}\nAGGREGATE (mean ± std across {len(SEEDS)} seeds)\n{'='*70}")
agg = {}
for strategy in ["Medical", "LLM", "Generic"]:
    miou = [all_results[s][strategy]["mIoU"]*100 for s in SEEDS]
    agg[strategy] = {
        "mIoU_mean": float(np.mean(miou)),
        "mIoU_std":  float(np.std(miou)),
    }
    print(f"  {strategy:10s}: mIoU = {agg[strategy]['mIoU_mean']:.2f}% ± {agg[strategy]['mIoU_std']:.2f}%")

out = f"{WORK}/phase_A2_eval_multiseed.json"
with open(out, "w") as f:
    json.dump({"per_seed": all_results, "aggregate": agg}, f, indent=2)
print(f"\nSaved: {out}")
