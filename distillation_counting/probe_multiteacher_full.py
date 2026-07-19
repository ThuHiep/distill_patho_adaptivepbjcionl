#!/usr/bin/env python3
"""PROBE đầy đủ Hướng A — committee 3-4 teacher trên NuInsSeg.

Teacher (mỗi ảnh = 1 count):
  - PathoSAM     : REPO/data/pathosam_nuinsseg_preds.pkl        (count = len(scores))
  - SAM3+A2 LoRA : REPO/weights/phase_E_nuinsseg_preds.pkl      (count = len(scores))
  - NuLite-T     : nulite_preds.csv  (image,pred_count)         <- dump_cellvit_counts.py --nulite
  - LKCell-L     : lkcell_preds.csv  (image,pred_count)         <- dump_cellvit_counts.py --lkcell
GT/organ: REPO/work/nuinsseg_png/gt_counts.csv (image,gt,organ)

Đo: (1) consensus (mean/median) có HẠ MAE so teacher đơn tốt nhất không?
    (2) corr(std-disagreement giữa teacher, |lỗi|) — std nhiều teacher giàu hơn |a-b| của 2.
Align: csv teacher khớp theo IMAGE (chính xác); 2 pkl cũ khớp theo (organ,gt) greedy (xấp xỉ).
Cần >=2 teacher; teacher thiếu (csv chưa có) tự bỏ qua.
"""
import argparse, csv, os, pickle
from collections import defaultdict
import numpy as np

REPO = os.environ.get("REPO", "/kaggle/working/repo")
DC = os.path.join(REPO, "distillation_counting")
ap = argparse.ArgumentParser()
ap.add_argument("--gt", default=f"{REPO}/work/nuinsseg_png/gt_counts.csv")
ap.add_argument("--nulite", default=f"{DC}/nulite_preds.csv")
ap.add_argument("--lkcell", default=f"{DC}/lkcell_preds.csv")
ap.add_argument("--pathosam", default=f"{REPO}/data/pathosam_nuinsseg_preds.pkl")
ap.add_argument("--sam3", default=f"{REPO}/weights/phase_E_nuinsseg_preds.pkl")
args = ap.parse_args()


def read_gt(path):
    rows = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            img = r.get("image") or r.get("name")
            gt = float(r.get("gt") or r.get("gt_count") or r.get("count"))
            organ = r.get("organ", "?")
            rows[img] = (gt, organ)
    return rows


def read_csv_counts(path):
    out = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            img = r.get("image") or r.get("name")
            out[img] = float(r.get("pred_count") or r.get("count") or r.get("pred"))
    return out


def read_pkl_by_organgt(path):
    """list các (organ, round(gt)) -> deque count; count = len(scores)."""
    d = pickle.load(open(path, "rb"))
    buckets = defaultdict(list)
    for p, g, o in zip(d["preds"], d["gts"], d["organs"]):
        gt = float(np.ravel(g)[0])
        n = len(np.asarray(p["scores"], float))
        buckets[(o, round(gt))].append(n)
    return buckets


gt = read_gt(args.gt)
images = list(gt.keys())
print(f"[gt] {len(images)} ảnh")

teachers = {}          # name -> dict image->count
# --- csv teacher (khớp image chính xác) ---
for name, path in [("NuLite", args.nulite), ("LKCell", args.lkcell)]:
    if os.path.exists(path):
        c = read_csv_counts(path)
        teachers[name] = {im: c[im] for im in images if im in c}
        print(f"[teacher] {name:9s} {len(teachers[name])}/{len(images)} ảnh (csv, khớp image)")
    else:
        print(f"[teacher] {name:9s} — KHÔNG có {path} (bỏ qua)")

# --- pkl teacher (khớp (organ,gt) greedy) ---
for name, path in [("PathoSAM", args.pathosam), ("SAM3", args.sam3)]:
    if not os.path.exists(path):
        print(f"[teacher] {name:9s} — KHÔNG có {path} (bỏ qua)")
        continue
    buckets = read_pkl_by_organgt(path)
    used = defaultdict(int)
    tc = {}
    for im in images:
        g, o = gt[im]
        key = (o, round(g))
        lst = buckets.get(key, [])
        idx = used[key]
        if idx < len(lst):
            tc[im] = lst[idx]; used[key] += 1
    teachers[name] = tc
    print(f"[teacher] {name:9s} {len(tc)}/{len(images)} ảnh (pkl, khớp organ+gt)")

names = list(teachers.keys())
assert len(names) >= 2, "cần >=2 teacher"

# --- giữ ảnh có ĐỦ mọi teacher ---
common = [im for im in images if all(im in teachers[n] for n in names)]
print(f"\n[align] {len(common)} ảnh có đủ {len(names)} teacher: {names}")
G = np.array([gt[im][0] for im in common], float)
M = np.stack([[teachers[n][im] for im in common] for n in names], 0).astype(float)  # (T, N)

cons_mean = M.mean(0)
cons_med = np.median(M, 0)
disagree = M.std(0)                     # std giữa các teacher / ảnh


def mae(x):
    return float(np.abs(x - G).mean())


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1])


print("\n===== MAE từng teacher =====")
for i, n in enumerate(names):
    print(f"  {n:9s} {mae(M[i]):6.2f}")
best_single = min(mae(M[i]) for i in range(len(names)))
print(f"\n  consensus(mean)   {mae(cons_mean):6.2f}")
print(f"  consensus(median) {mae(cons_med):6.2f}")
print(f"  -> consensus HẠ MAE so teacher tốt nhất ({best_single:.2f})? "
      f"{'CÓ' if min(mae(cons_mean), mae(cons_med)) < best_single else 'KHÔNG'}")

print("\n===== Disagreement (std giữa teacher) làm σ epistemic =====")
print(f"  corr(std, |consensus_mean - gt|) = {corr(disagree, np.abs(cons_mean - G)):+.3f}   <- KEY")
print(f"  corr(std, |consensus_med  - gt|) = {corr(disagree, np.abs(cons_med - G)):+.3f}")
print(f"  std: mean {disagree.mean():.1f} ({100*disagree.mean()/max(G.mean(),1):.0f}% gt) | max {disagree.max():.0f}")

print("\nVERDICT: cần (consensus hạ MAE) VÀ (corr(std,|err|) > ~0.3) → ĐÁNG làm thí nghiệm σ-disagreement đầy đủ.")
