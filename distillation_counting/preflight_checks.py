"""
preflight_checks.py — TEST RỦI RO TRƯỚC KHI TRAIN (chạy trên vast, ~vài phút, RẺ).

Mục đích: bắt mọi điểm chết TRƯỚC khi tốn tiền/giờ GPU cho full run. Kiểm theo thứ tự RẺ->ĐẮT,
fail-fast, mỗi bước in PASS/FAIL + gợi ý sửa. Chỉ đụng 3 ảnh nên chạy nhanh.

Chạy:
    cd distillation_counting
    REPO=/workspace/sam3_research python preflight_checks.py

Nếu TẤT CẢ pass -> yên tâm chạy full:
    python distill_student_pbud.py --loss pbud  --student_ch 32 --epochs 60 --out work/student_pbud.pkl
"""
from __future__ import annotations
import os, sys, time, tempfile, traceback
import numpy as np

REPO = os.environ.get("REPO", "/workspace/sam3_research")
for p in (f"{REPO}/kaggle/vast", f"{REPO}/kaggle/lib", REPO, os.path.dirname(__file__)):
    if p not in sys.path:
        sys.path.insert(0, p)

OK, BAD = "\033[92m✓\033[0m", "\033[91m✗\033[0m"
_fail = []


def step(name):
    def deco(fn):
        def wrap():
            t = time.time()
            try:
                msg = fn() or ""
                print(f"  {OK} {name}  {msg}  ({time.time()-t:.1f}s)")
                return True
            except Exception as e:
                print(f"  {BAD} {name}\n      -> {e}")
                traceback.print_exc()
                _fail.append(name)
                return False
        return wrap
    return deco


# ---- các bước, rẻ -> đắt ----
@step("1. deps (torch/scipy/numpy/PIL)")
def s1():
    import torch, scipy, numpy, PIL  # noqa
    return f"torch {torch.__version__}"


@step("2. GPU khả dụng")
def s2():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("KHÔNG có CUDA — train sẽ rất chậm. Kiểm image/driver vast.")
    return torch.cuda.get_device_name(0)


@step("3. import repo libs (conformal, pbud_losses, trainer)")
def s3():
    import conformal, pbud_losses  # noqa
    import distill_student_nuinsseg, distill_student_pbud  # noqa
    return "ok"


@step("4. loss math (self-test nhanh)")
def s4():
    import torch
    from pbud_losses import pbud_loss, ccad_loss, pb_moments
    s = torch.rand(8, requires_grad=True)
    out = pbud_loss(s, torch.ones(8, 1), torch.rand(8), torch.ones(8, 1), torch.tensor([4.0]))
    out["loss"].backward()
    assert torch.isfinite(out["loss"]) and s.grad.abs().sum() > 0, "pbud không khả vi/không hữu hạn"
    return "pbud khả vi OK"


@step("5. NuInsSeg data tìm thấy + index")
def s5():
    from distill_student_nuinsseg import build_index, find_root
    root = find_root()
    samples = build_index(root)
    assert len(samples) > 0, "0 cặp (image,mask) — kiểm cấu trúc NuInsSeg"
    global _SAMPLES
    _SAMPLES = samples
    return f"{len(samples)} cặp, {len(set(s['organ'] for s in samples))} organ @ {root}"


@step("6. PathoSAM load")
def s6():
    import torch
    from pathosam_lib import load_pathosam
    global _PRED, _SEG
    _PRED, _SEG = load_pathosam("cuda" if torch.cuda.is_available() else "cpu")
    return "loaded"


@step("7. teacher chạy 1 ảnh (masks/scores/foreground hợp lệ)")
def s7():
    from PIL import Image
    from pathosam_lib import pathosam_instances
    img = np.asarray(Image.open(_SAMPLES[0]["image"]).convert("RGB"))
    masks, scores, feat = pathosam_instances(img, _PRED, _SEG)
    assert len(scores) == len(masks), "scores/masks lệch"
    assert (scores >= -1e-6).all() and (scores <= 1 + 1e-6).all(), "score ngoài [0,1]"
    assert hasattr(_SEG, "_foreground"), "thiếu _foreground (teacher signal)"
    return f"{len(masks)} instance, score∈[{scores.min():.2f},{scores.max():.2f}]"


@step("8. build teacher targets 3 ảnh (cache PBUD: fg+label+s_T)")
def s8():
    import torch
    from distill_student_pbud import build_teacher_targets_pbud
    global _DATA, _T_PER_IMG
    t0 = time.time()
    tmp = os.path.join(tempfile.gettempdir(), "preflight_targets.pkl")
    if os.path.exists(tmp):
        os.remove(tmp)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    _DATA = build_teacher_targets_pbud(_SAMPLES[:3], dev, tmp)
    _T_PER_IMG = (time.time() - t0) / 3
    d = _DATA[0]
    assert set(d) >= {"img", "fg", "label", "s_T", "gt", "organ"}, "cache thiếu field"
    assert d["label"].max() >= 1 or len(d["s_T"]) == 0, "label instance rỗng bất thường"
    return f"{_T_PER_IMG:.1f}s/ảnh, ảnh0: {len(_DATA[0]['s_T'])} instance"


@step("9. 1 bước train PBUD (loss hữu hạn + params đổi)")
def s9():
    import torch
    from distill_student_pbud import train
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m0 = None
    model = train(_DATA, dev, epochs=1, ch=16, loss_kind="pbud_ccad", lr=1e-3,
                  train_idx=list(range(len(_DATA))), alpha=0.4, beta=0.3, gamma=0.3, bs=3)
    p = next(model.parameters())
    assert torch.isfinite(p).all(), "params NaN sau train"
    global _MODEL
    _MODEL = model
    return "train step OK (pbud_ccad)"


@step("10. inference -> schema preds hợp lệ cho eval_coverage_transfer")
def s10():
    import torch
    from distill_student_nuinsseg import student_predict
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = student_predict(_MODEL, _DATA, dev, thresh=0.5, min_area=5)
    assert set(out) >= {"preds", "gts", "organs"}, "thiếu key"
    p0 = out["preds"][0]
    assert set(p0) >= {"scores", "probs", "K"}, "pred thiếu scores/probs/K"
    assert p0["probs"].shape[0] == len(p0["scores"]), "probs/scores lệch"
    return f"{len(out['preds'])} preds, ảnh0 {len(p0['scores'])} instance"


@step("11. ước tính chi phí full teacher-build")
def s11():
    n = len(_SAMPLES)
    total = _T_PER_IMG * n / 60.0
    return f"~{total:.0f} phút cho {n} ảnh (chỉ phase A, 1 lần; cache lại dùng nhiều lần)"


def main():
    print("=" * 70)
    print(f"PREFLIGHT — test rủi ro trước train | REPO={REPO}")
    print("=" * 70)
    steps = [s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11]
    for st in steps:
        if not st() and st.__name__ in ("s1", "s3", "s5", "s6"):
            print(f"\n[DỪNG] bước nền {st.__name__} fail — sửa trước khi tiếp.")
            break
    print("=" * 70)
    if _fail:
        print(f"CÓ {len(_fail)} bước FAIL: {_fail}\n=> SỬA rồi chạy lại preflight TRƯỚC khi train.")
        sys.exit(1)
    print("TẤT CẢ PASS ✓  => sẵn sàng chạy full:")
    print("  python distill_student_pbud.py --loss kd    --student_ch 32 --epochs 60 --out work/student_kd.pkl")
    print("  python distill_student_pbud.py --loss pbud   --student_ch 32 --epochs 60 --out work/student_pbud.pkl")
    print("  python distill_student_pbud.py --loss pbud_ccad --student_ch 32 --epochs 60 --out work/student_pbudccad.pkl")


if __name__ == "__main__":
    main()
