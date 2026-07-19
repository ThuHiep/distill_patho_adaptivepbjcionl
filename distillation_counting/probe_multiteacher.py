#!/usr/bin/env python3
"""PROBE (concept-gate) cho Hướng A — multi-teacher agreement distillation.

Có sẵn 2 teacher KHÁC KIẾN TRÚC trên cùng 665 ảnh NuInsSeg (MIỄN PHÍ, đã tính):
  - PathoSAM        : data/pathosam_nuinsseg_preds.pkl
  - SAM3 + A2 LoRA  : weights/phase_E_nuinsseg_preds.pkl  (phase_E)
Mỗi item = {scores: per-instance detection scores}; count = len(scores) (và biến thể ngưỡng 0.5).

Câu hỏi gate (rẻ, trước khi build density cache đa-teacher):
  (1) Consensus 2-teacher có HẠ MAE so với teacher đơn không?
  (2) Bất đồng giữa 2 teacher có TƯƠNG QUAN với lỗi không (→ dùng làm σ epistemic)?
Nếu (1) hạ MAE và (2) corr>~0.3 → ĐÁNG làm đầy đủ. Nếu không → ghi honest, dừng.

Align: 2 pkl khác thứ tự, không có img-id → ghép theo (organ, gt) greedy (xấp xỉ cho probe;
bản đầy đủ sẽ build density cache align chuẩn).
"""
import pickle
from collections import defaultdict
import numpy as np

A = pickle.load(open("data/pathosam_nuinsseg_preds.pkl", "rb"))       # teacher 1
B = pickle.load(open("weights/phase_E_nuinsseg_preds.pkl", "rb"))     # teacher 2


def gtf(x):
    return float(np.ravel(x)[0])


def counts(D):
    out = []
    for p, g, o in zip(D["preds"], D["gts"], D["organs"]):
        s = np.asarray(p["scores"], float)
        out.append({"n": len(s), "n50": int((s > 0.5).sum()), "gt": gtf(g), "organ": o})
    return out


ca, cb = counts(A), counts(B)
print(f"[load] teacher1(PathoSAM) N={len(ca)} | teacher2(SAM3+LoRA) N={len(cb)}")

# --- align greedy theo (organ, gt) ---
buckets = defaultdict(list)
for i, x in enumerate(cb):
    buckets[(x["organ"], round(x["gt"]))].append(i)
pairs, used = [], set()
for x in ca:
    key = (x["organ"], round(x["gt"]))
    cand = [j for j in buckets.get(key, []) if j not in used]
    if cand:
        j = cand[0]; used.add(j); pairs.append((x, cb[j]))
print(f"[align] ghép được {len(pairs)}/{len(ca)} ảnh theo (organ,gt)")

for tag, key in [("count=len(scores)", "n"), ("count=#(score>0.5)", "n50")]:
    a = np.array([p[0][key] for p in pairs], float)
    b = np.array([p[1][key] for p in pairs], float)
    gt = np.array([p[0]["gt"] for p in pairs], float)
    cons = (a + b) / 2.0
    disagree = np.abs(a - b)
    mae_a, mae_b, mae_c = (np.abs(a - gt).mean(), np.abs(b - gt).mean(), np.abs(cons - gt).mean())
    mae_med = np.abs(np.median(np.stack([a, b]), 0) - gt).mean()

    def corr(u, v):
        return float(np.corrcoef(u, v)[0, 1])

    print(f"\n===== {tag} =====")
    print(f" MAE teacher1 {mae_a:6.2f} | teacher2 {mae_b:6.2f} | consensus(mean) {mae_c:6.2f} | consensus(median) {mae_med:6.2f}")
    print(f" consensus hạ MAE so với teacher tốt hơn? {'CÓ' if mae_c < min(mae_a, mae_b) else 'KHÔNG'} "
          f"(min đơn = {min(mae_a, mae_b):.2f})")
    print(f" corr(disagree, |consensus-gt|) = {corr(disagree, np.abs(cons - gt)):+.3f}   <- KEY (σ epistemic?)")
    print(f" corr(disagree, |teacher1-gt|)  = {corr(disagree, np.abs(a - gt)):+.3f}")
    print(f" disagree: mean {disagree.mean():.1f} ({100*disagree.mean()/max(gt.mean(),1):.0f}% gt) | max {disagree.max():.0f}")

print("\nVERDICT gate: cần (consensus hạ MAE) VÀ (corr(disagree,|err|) > ~0.3) thì ĐÁNG đi tiếp.")
