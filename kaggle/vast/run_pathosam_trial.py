"""
PathoSAM DISCOVERY trial — run inside the conda env `pathosam` on Vast.

  micromamba run -n pathosam python run_pathosam_trial.py

Answers 3 questions before committing to a full pipeline:
  (1) Does the GENERALIST give nucleus TYPE (5-class) or instance-only?
  (2) Counting quality on PanNuke Fold 3 (MAE vs GT) — is it the STRONG predictor
      we want vs SAM3 (~weak)?
  (3) Fold-3 tissue distribution -> how many COLON images to exclude (Lizard overlap).

Leakage note: PathoSAM generalist trained on CoNSeP/CPM17/Lizard/MoNuSeg/MoNuSAC/TNBC
— NOT PanNuke (documented) -> fold-clean by construction. Only residual = Lizard
contains PanNuke-colon -> exclude colon tissue.

Env: needs micro_sam + patho_sam (conda) + PanNuke at /workspace/sam3_research/data/pannuke.
Use --n to cap images for a quick/CPU pass (default 60).
"""
from __future__ import annotations
import argparse, os, sys, time
from collections import Counter
from pathlib import Path
import numpy as np

REPO = "/workspace/sam3_research"
DATA_ROOT = f"{REPO}/data/pannuke"
for p in (f"{REPO}/kaggle/lib", REPO):
    if p not in sys.path:
        sys.path.insert(0, p)


def load_fold3():
    """Try the repo loader; on path mismatch, fall back to a direct npy search."""
    from pannuke_loader import PanNukeFold, CELL_TYPES  # noqa
    try:
        return PanNukeFold(DATA_ROOT, 3), CELL_TYPES
    except Exception as e:
        print(f"[loader] PanNukeFold failed: {repr(e)[:140]}")
        print(f"[loader] scanning {DATA_ROOT} for images.npy ...")
        hits = []
        for dp, _dn, fn in os.walk(DATA_ROOT):
            if "images.npy" in fn:
                hits.append(os.path.join(dp, "images.npy"))
        for h in hits:
            print("   ", h)
        raise SystemExit(
            "Adjust DATA_ROOT / loader layout above, then re-run. "
            "(Expected Fold-3 images.npy somewhere under data/pannuke.)"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60,
                    help="cap #Fold-3 images for counting (CPU-friendly). 0 = all.")
    ap.add_argument("--model", default="vit_l_histopathology")
    args = ap.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device} | torch {torch.__version__}")

    fold3, CELL_TYPES = load_fold3()
    n_total = len(fold3)
    n = n_total if args.n in (0, None) else min(args.n, n_total)
    print(f"Fold 3: {n_total} images | using {n} for counting\n")

    # ---- Q3 first (no model needed): tissue distribution ------------------
    print("=" * 64)
    print("Q3 — Fold-3 tissue distribution (colon = only leakage to exclude)")
    print("=" * 64)
    tissues = Counter(fold3[i]["tissue"] for i in range(n_total))
    for t, c in tissues.most_common():
        flag = "  <-- COLON: exclude (Lizard overlap)" if "colon" in t.lower() else ""
        print(f"  {t:18s}: {c:4d}{flag}")
    colon_n = sum(c for t, c in tissues.items() if "colon" in t.lower())
    print(f"  COLON total: {colon_n}/{n_total} ({100*colon_n/n_total:.1f}%)\n")

    # ---- load generalist + AIS -------------------------------------------
    print("=" * 64)
    print(f"Loading generalist '{args.model}' + AIS (amg=False)...")
    print("=" * 64)
    from micro_sam.automatic_segmentation import (
        get_predictor_and_segmenter, automatic_instance_segmentation,
    )
    predictor, segmenter = get_predictor_and_segmenter(
        model_type=args.model, device=device, segmentation_mode="ais")
    print("loaded.\n")

    def pathosam_count(img_rgb):
        inst = automatic_instance_segmentation(
            predictor=predictor, segmenter=segmenter, input_path=img_rgb, ndim=2)
        inst = np.asarray(inst)
        k = int(len(np.unique(inst)) - (1 if (inst == 0).any() else 0))
        return inst, k

    # ---- Q1 — type vs instance -------------------------------------------
    print("=" * 64)
    print("Q1 — does output carry TYPE (5-class) or instance-only?")
    print("=" * 64)
    s0 = fold3[0]
    inst0, k0 = pathosam_count(s0["image"])
    print(f"  instance map dtype={inst0.dtype} unique={len(np.unique(inst0))} "
          f"-> values are nucleus IDs (instance), NOT class labels.")
    try:
        import patho_sam, pkgutil
        subs = [m.name for m in pkgutil.iter_modules(patho_sam.__path__)]
        print("  patho_sam submodules:", subs)
        print("  -> look for 'semantic'/'classification' (a 5-class model would be "
              "PanNuke-trained = LEAKY, so we AVOID it and use our own TypeHead F1+2).")
    except Exception as e:
        print("  patho_sam introspection failed:", repr(e)[:120])
    print("  EXPECTED: instance-only -> attach OUR TypeHead (Fold 1+2, clean) for p_ik.\n")

    # ---- Q2 — counting MAE vs GT -----------------------------------------
    print("=" * 64)
    print(f"Q2 — counting on {n} Fold-3 images (MAE vs GT)")
    print("=" * 64)
    abs_err, t0 = [], time.time()
    for i in range(n):
        s = fold3[i]
        _, k = pathosam_count(s["image"])
        gt = int(s["counts"].sum())
        abs_err.append(abs(k - gt))
        if i < 8:
            print(f"  img {i:3d} | {s['tissue']:14s} | GT={gt:3d} | PathoSAM={k:3d} | "
                  f"|err|={abs(k-gt):3d}")
    mae = float(np.mean(abs_err))
    dt = time.time() - t0
    print(f"\n  MAE over {n} imgs = {mae:.2f}  ({dt:.0f}s, {dt/n:.2f}s/img)")
    print(f"  (SAM3+LoRA reference total-count MAE ~ 15.7 on NuInsSeg; lower = stronger.)")

    print("\n" + "=" * 64)
    print("VERDICT INPUTS:")
    print(f"  Q1 type      : instance-only (see above) -> +our TypeHead")
    print(f"  Q2 count MAE : {mae:.2f} on {n} Fold-3 imgs (device={device})")
    print(f"  Q3 colon     : {colon_n}/{n_total} imgs to exclude")
    print("=" * 64)


if __name__ == "__main__":
    main()
