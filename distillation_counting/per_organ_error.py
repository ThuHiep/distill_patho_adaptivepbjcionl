#!/usr/bin/env python3
"""per_organ_error.py — lỗi ĐẾM theo MÔ (over/under-count) để tìm mô model yếu nhất.

Mục đích: xác định đặc điểm "tế bào máu/bạch huyết" (mô lympho/tạo máu xếp dày) có phải
chỗ model sai nhất không → định hướng module + loss mới. Data dẫn đường, không đoán.

Nguồn (như compute_r2_counting):
  --pkl   PACT/efflite0 pkl {"preds":[{"mu"}],"gts":[[gt]],"organs":[...]}   -> pred=mu
  --teacher_pkl  {"preds":[{"scores"}],...}  -> pred=len(scores)

VD:
  python per_organ_error.py --pkl work/baseline_countonly_efflite0.pkl
  python per_organ_error.py --teacher_pkl ../data/pathosam_nuinsseg_preds.pkl
"""
import argparse, pickle
import numpy as np
from collections import defaultdict


def load(path, teacher):
    d = pickle.load(open(path, "rb"))
    if teacher:
        pred = np.array([len(p["scores"]) for p in d["preds"]], float)
    else:
        pred = np.array([p["mu"] for p in d["preds"]], float)
    gt = np.array([float(np.ravel(g)[0]) for g in d["gts"]], float)
    org = list(d["organs"])
    return gt, pred, org


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl")
    ap.add_argument("--teacher_pkl")
    args = ap.parse_args()
    path = args.pkl or args.teacher_pkl
    assert path, "cần --pkl hoặc --teacher_pkl"
    gt, pred, org = load(path, teacher=bool(args.teacher_pkl))

    g = defaultdict(lambda: {"n": 0, "gt": 0.0, "pred": 0.0, "ae": 0.0, "bias": 0.0})
    for i in range(len(gt)):
        o = org[i]; e = g[o]
        e["n"] += 1; e["gt"] += gt[i]; e["pred"] += pred[i]
        e["ae"] += abs(pred[i] - gt[i]); e["bias"] += pred[i] - gt[i]

    rows = []
    for o, e in g.items():
        n = e["n"]
        rows.append((e["bias"] / n, o, n, e["gt"] / n, e["pred"] / n, e["ae"] / n))
    print(f"{'organ':30s} {'n':>3s} {'GTmean':>7s} {'PREDmn':>7s} {'MAE':>7s} {'bias(P-G)':>10s}")
    for bias, o, n, gm, pm, mae in sorted(rows):   # âm nhất (đếm thiếu) lên đầu
        flag = "  <-- THIẾU" if bias < -0.15 * gm else ("  <-- DƯ" if bias > 0.15 * gm else "")
        print(f"{o:30s} {n:>3d} {gm:>7.1f} {pm:>7.1f} {mae:>7.1f} {bias:>+10.1f}{flag}")
    print(f"\nTỔNG n={len(gt)} | MAE={np.abs(pred-gt).mean():.2f} | bias tổng={np.mean(pred-gt):+.2f}")


if __name__ == "__main__":
    main()
