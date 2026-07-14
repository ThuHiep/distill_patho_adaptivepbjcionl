"""
eval_heavy_count.py — Chấm count-MAE Phần B (Bước 2): heavy net (CellViT-SAM-H / LKCell-L) chạy
OFF-THE-SHELF trên NuInsSeg (OOD → leak-free), so với student R2 (1.9M) + teacher PathoSAM.

ĐẦU VÀO (tách sạch khỏi inference của họ — mình chỉ chấm số):
  --gt        gt_counts.csv  (image,gt,organ)   <- prep_nuinsseg_as_pannuke.py
  --preds     preds.csv      (image,pred_count)  <- DUMP từ inference CellViT/LKCell:
                 count/ảnh = len(predictions["instance_types"][i]) (xem runbook md Bước 2).
  --tiles_map tiles_map.csv  (tile,image)  [chỉ mode tile] -> count ảnh = TỔNG count các tile.
  --student_pkl work/student_r2_nuinsseg_cv5_poisson_feat.pkl  [tùy] -> thêm dòng student (mu=count).
  --label     nhãn method cho bảng (vd "CellViT-SAM-H").

RA: MAE, RMSE, MAPE + per-organ MAE (min/worst) — cùng khung tổ chức organ như eval_r2_grouped.
So sánh trực tiếp: heavy net OOD vs student in-domain. MAE thấp = đếm tốt.

Chạy:
  python eval_heavy_count.py --gt ../work/nuinsseg_png/gt_counts.csv --preds cellvit_preds.csv \
      --label CellViT-SAM-H --student_pkl ../work/student_r2_nuinsseg_cv5_poisson_feat.pkl
"""
from __future__ import annotations
import argparse, csv, pickle
from collections import defaultdict
import numpy as np


def read_csv(path):
    with open(path) as f:
        r = csv.reader(f); header = next(r); return header, [row for row in r]


def load_gt(path):
    _, rows = read_csv(path)
    gt = {row[0]: float(row[1]) for row in rows}
    organ = {row[0]: (row[2] if len(row) > 2 else "_all_") for row in rows}
    return gt, organ


def load_preds(path, tiles_map=None):
    _, rows = read_csv(path)
    pred = {row[0]: float(row[1]) for row in rows}
    if tiles_map:
        _, trows = read_csv(tiles_map)
        agg = defaultdict(float)
        for tile, image in trows:
            agg[image] += pred.get(tile, 0.0)   # count ảnh = tổng count các tile
        return dict(agg)
    return pred


def stats(gt, pred, organ, label):
    keys = [k for k in gt if k in pred]
    miss = [k for k in gt if k not in pred]
    y = np.array([gt[k] for k in keys]); p = np.array([pred[k] for k in keys])
    ae = np.abs(p - y)
    mae = float(ae.mean()); rmse = float(np.sqrt(((p - y) ** 2).mean()))
    mape = float((ae / np.maximum(y, 1)).mean() * 100)
    # per-organ
    by = defaultdict(list)
    for k in keys:
        by[organ.get(k, "_all_")].append(abs(pred[k] - gt[k]))
    org_mae = {o: float(np.mean(v)) for o, v in by.items() if len(v) >= 5}
    worst = max(org_mae.items(), key=lambda kv: kv[1]) if org_mae else (None, None)
    return {"label": label, "n": len(keys), "miss": len(miss), "mae": mae, "rmse": rmse,
            "mape": mape, "org_mae": org_mae, "worst": worst}


def student_row(pkl, gt_from_pkl=True):
    """Student pkl: preds[{mu}], gts, organs -> count-MAE (mu vs gt) trên CHÍNH tập cv5 của nó."""
    obj = pickle.load(open(pkl, "rb"))
    mu = np.array([float(pp["mu"]) for pp in obj["preds"]])
    y = np.array([float(np.asarray(g).reshape(-1)[0]) for g in obj["gts"]])
    organs = list(obj.get("organs", ["_all_"] * len(mu)))
    ae = np.abs(mu - y)
    by = defaultdict(list)
    for i, o in enumerate(organs):
        by[o].append(ae[i])
    org_mae = {o: float(np.mean(v)) for o, v in by.items() if len(v) >= 5}
    worst = max(org_mae.items(), key=lambda kv: kv[1]) if org_mae else (None, None)
    return {"label": "Student R2 (ours, 1.9M)", "n": len(mu), "miss": 0,
            "mae": float(ae.mean()), "rmse": float(np.sqrt((ae ** 2).mean())),
            "mape": float((ae / np.maximum(y, 1)).mean() * 100), "org_mae": org_mae, "worst": worst}


def teacher_row(pkl):
    """Teacher PathoSAM cache: list {density,gt,organ}. count = density.sum() (= tổng density teacher).
    Reference: student distill TỪ teacher này -> so student giữ/vượt khả năng đếm của thầy."""
    data = pickle.load(open(pkl, "rb"))
    pred = np.array([float(d["density"].sum()) for d in data])
    y = np.array([float(d["gt"]) for d in data])
    organs = [d.get("organ", "_all_") for d in data]
    ae = np.abs(pred - y)
    by = defaultdict(list)
    for i, o in enumerate(organs):
        by[o].append(ae[i])
    org_mae = {o: float(np.mean(v)) for o, v in by.items() if len(v) >= 5}
    worst = max(org_mae.items(), key=lambda kv: kv[1]) if org_mae else (None, None)
    return {"label": "Teacher PathoSAM (~640M)", "n": len(pred), "miss": 0,
            "mae": float(ae.mean()), "rmse": float(np.sqrt((ae ** 2).mean())),
            "mape": float((ae / np.maximum(y, 1)).mean() * 100), "org_mae": org_mae, "worst": worst}


def pretty(rows):
    print("\n" + "=" * 84)
    print("PHẦN B — COUNT-MAE trên NuInsSeg (heavy net OOD off-the-shelf vs student in-domain)")
    print("=" * 84)
    h = f"{'method':26} | {'N':>4} | {'MAE':>7} | {'RMSE':>7} | {'MAPE%':>6} | {'worst-organ MAE':>22}"
    print(h); print("-" * len(h))
    for d in rows:
        wo = f"{d['worst'][0]}={d['worst'][1]:.2f}" if d["worst"][0] else "n/a"
        miss = f" (miss {d['miss']})" if d.get("miss") else ""
        print(f"{d['label']:26} | {d['n']:>4} | {d['mae']:7.2f} | {d['rmse']:7.2f} | "
              f"{d['mape']:6.1f} | {wo:>22}{miss}")
    print("-" * len(h))
    print("[GHI CHÚ] student in-domain (cv5) vs heavy net OOD → story: adapt 700M rất đắt, distill 1.9M rẻ mà bám sát.")
    print("          + student là model DUY NHẤT có σ/interval (bảng A). MAE thấp hơn = đếm tốt hơn.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--preds", default=None, help="preds.csv của heavy net (image,pred_count)")
    ap.add_argument("--tiles_map", default=None)
    ap.add_argument("--label", default="HeavyNet")
    ap.add_argument("--student_pkl", default=None)
    ap.add_argument("--teacher_pkl", default=None, help="cache teacher density (teacher_density_nuinsseg.pkl)")
    args = ap.parse_args()

    gt, organ = load_gt(args.gt)
    rows = []
    if args.preds:
        pred = load_preds(args.preds, args.tiles_map)
        rows.append(stats(gt, pred, organ, args.label))
    if args.teacher_pkl:
        rows.append(teacher_row(args.teacher_pkl))
    if args.student_pkl:
        rows.append(student_row(args.student_pkl))
    if not rows:
        print("Chưa có --preds hay --student_pkl để chấm."); return
    pretty(rows)


if __name__ == "__main__":
    main()
