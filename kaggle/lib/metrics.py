from __future__ import annotations
from typing import Sequence
import numpy as np

def binary_iou(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool); b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union > 0 else 0.0

def binary_dice(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool); b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    sa, sb = a.sum(), b.sum()
    return float(2 * inter) / float(sa + sb) if (sa + sb) > 0 else 0.0

def match_pred_to_gt(pred_masks: Sequence[np.ndarray], gt_masks: Sequence[np.ndarray],
                     iou_thresh: float = 0.5) -> dict:
    if not pred_masks and not gt_masks:
        return {"tp": 0, "fp": 0, "fn": 0, "mean_iou": 0.0}
    if not pred_masks:
        return {"tp": 0, "fp": 0, "fn": len(gt_masks), "mean_iou": 0.0}
    if not gt_masks:
        return {"tp": 0, "fp": len(pred_masks), "fn": 0, "mean_iou": 0.0}

    iou_matrix = np.zeros((len(pred_masks), len(gt_masks)), dtype=np.float32)
    for i, pm in enumerate(pred_masks):
        for j, gm in enumerate(gt_masks):
            iou_matrix[i, j] = binary_iou(pm, gm)

    matched_pred, matched_gt = set(), set()
    ious = []
    pairs = np.dstack(np.unravel_index(np.argsort(-iou_matrix.ravel()), iou_matrix.shape))[0]
    for i, j in pairs:
        if iou_matrix[i, j] < iou_thresh:
            break
        if i in matched_pred or j in matched_gt:
            continue
        matched_pred.add(int(i)); matched_gt.add(int(j))
        ious.append(float(iou_matrix[i, j]))

    tp = len(matched_pred)
    fp = len(pred_masks) - tp
    fn = len(gt_masks)  - len(matched_gt)
    return {"tp": tp, "fp": fp, "fn": fn,
            "mean_iou": float(np.mean(ious)) if ious else 0.0}

def panoptic_quality(stats: dict) -> dict:
    tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
    sq = stats["mean_iou"]
    denom = tp + 0.5 * fp + 0.5 * fn
    rq = tp / denom if denom > 0 else 0.0
    pq = sq * rq
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"PQ": pq, "SQ": sq, "RQ": rq, "F1": f1, "P": precision, "R": recall}

def aggregate_iou_image(pred_masks: Sequence[np.ndarray], gt_masks: Sequence[np.ndarray]) -> float:
    H, W = (gt_masks[0].shape if gt_masks else
            (pred_masks[0].shape if pred_masks else (256, 256)))
    pu = np.zeros((H, W), dtype=bool)
    for m in pred_masks: pu |= m.astype(bool)
    gu = np.zeros((H, W), dtype=bool)
    for m in gt_masks:   gu |= m.astype(bool)
    return binary_iou(pu, gu)

def aggregate_iou_dice_image(pred_masks: Sequence[np.ndarray],
                              gt_masks: Sequence[np.ndarray]) -> tuple:
    H, W = (gt_masks[0].shape if gt_masks else
            (pred_masks[0].shape if pred_masks else (256, 256)))
    pu = np.zeros((H, W), dtype=bool)
    for m in pred_masks: pu |= m.astype(bool)
    gu = np.zeros((H, W), dtype=bool)
    for m in gt_masks:   gu |= m.astype(bool)
    return binary_iou(pu, gu), binary_dice(pu, gu)

def union_masks(masks: Sequence[np.ndarray], shape=(256, 256)) -> np.ndarray:
    u = np.zeros(shape, dtype=bool)
    for m in masks:
        u |= m.astype(bool)
    return u.astype(np.uint8)

class ClassWiseAccumulator:

    def __init__(self, class_names):
        self.class_names = list(class_names)
        self.tp = {c: 0 for c in self.class_names}
        self.fp = {c: 0 for c in self.class_names}
        self.fn = {c: 0 for c in self.class_names}

    def update(self, pred_mask: np.ndarray, gt_mask: np.ndarray, class_name: str):
        p = pred_mask.astype(bool)
        g = gt_mask.astype(bool)
        self.tp[class_name] += int(np.logical_and(p, g).sum())
        self.fp[class_name] += int(np.logical_and(p, np.logical_not(g)).sum())
        self.fn[class_name] += int(np.logical_and(np.logical_not(p), g).sum())

    def class_iou(self, class_name: str) -> float:
        tp, fp, fn = self.tp[class_name], self.fp[class_name], self.fn[class_name]
        denom = tp + fp + fn
        return float(tp) / float(denom) if denom > 0 else 0.0

    def class_dice(self, class_name: str) -> float:
        tp, fp, fn = self.tp[class_name], self.fp[class_name], self.fn[class_name]
        denom = 2 * tp + fp + fn
        return float(2 * tp) / float(denom) if denom > 0 else 0.0

    def mIoU(self) -> float:
        return float(np.mean([self.class_iou(c) for c in self.class_names]))

    def mDice(self) -> float:
        return float(np.mean([self.class_dice(c) for c in self.class_names]))

    def summary(self) -> dict:
        per_class = {c: {"IoU": self.class_iou(c), "Dice": self.class_dice(c),
                          "TP": self.tp[c], "FP": self.fp[c], "FN": self.fn[c]}
                      for c in self.class_names}
        return {
            "mIoU": self.mIoU(),
            "mDice": self.mDice(),
            "per_class": per_class,
        }

class PerPromptClassAccumulator:

    def __init__(self, class_names, prompts_per_class):
        self.class_names = list(class_names)
        self.prompts_per_class = {c: list(prompts_per_class[c]) for c in self.class_names}
        
        self.accs = {}
        for c, prompts in self.prompts_per_class.items():
            for p in prompts:
                self.accs[(c, p)] = ClassWiseAccumulator([c])

    def update(self, pred_mask: np.ndarray, gt_mask: np.ndarray,
               class_name: str, prompt: str):
        self.accs[(class_name, prompt)].update(pred_mask, gt_mask, class_name)

    def summary(self) -> dict:
        per_class = {}
        for c in self.class_names:
            per_prompt = []
            for p in self.prompts_per_class[c]:
                acc = self.accs[(c, p)]
                per_prompt.append({
                    "prompt": p,
                    "IoU": acc.class_iou(c),
                    "Dice": acc.class_dice(c),
                    "TP": acc.tp[c], "FP": acc.fp[c], "FN": acc.fn[c],
                })
            ious = [pp["IoU"] for pp in per_prompt]
            dices = [pp["Dice"] for pp in per_prompt]
            per_class[c] = {
                "IoU": float(np.mean(ious)),   
                "Dice": float(np.mean(dices)),
                "per_prompt": per_prompt,
            }
        mIoU = float(np.mean([per_class[c]["IoU"] for c in self.class_names]))
        mDice = float(np.mean([per_class[c]["Dice"] for c in self.class_names]))
        return {"mIoU": mIoU, "mDice": mDice, "per_class": per_class}

def bootstrap_ci(values, n_boot: int = 1000, alpha: float = 0.05, seed: int = 0):
    if len(values) == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    vals = np.asarray(values, dtype=np.float64)
    boots = [rng.choice(vals, size=len(vals), replace=True).mean() for _ in range(n_boot)]
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return lo, hi
