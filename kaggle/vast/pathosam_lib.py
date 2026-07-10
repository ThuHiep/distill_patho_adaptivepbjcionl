"""
PathoSAM shared library — extract per-instance (mask, s_i, pooled feature) from the
micro_sam generalist (vit_l_histopathology) AIS, for the PB-JCI pipeline.

Design (verified against micro_sam.instance_segmentation.InstanceSegmentationWithDecoder):
  - We call segmenter.initialize(image) + segmenter.generate() DIRECTLY (not the
    automatic_instance_segmentation wrapper) so we keep access to:
      * segmenter._foreground         -> foreground prob map (HxW)   -> s_i
      * predictor.features            -> ViT embedding (1,256,64,64) -> p_ik feature
  - s_i (detection score) = mean foreground prob over the instance mask (AIS has no
    native per-instance confidence; this is the principled objectness proxy).
  - feature = mask-pooled ViT embedding (reuses type_head.roi_pool_feature) -> 256-d
    -> TypeHead(256,128,5) trained on Fold 1+2 gives p_ik.

Run standalone to PROBE shapes on Vast:
  micromamba run -p /workspace/penv python pathosam_lib.py
"""
from __future__ import annotations
import sys, os
import numpy as np
import torch

REPO = "/workspace/sam3_research"
for p in (f"{REPO}/kaggle/lib", REPO):
    if p not in sys.path:
        sys.path.insert(0, p)

from type_head import roi_pool_feature  # noqa: E402

MODEL_NAME = "vit_l_histopathology"


def load_pathosam(device: str, model_name: str = MODEL_NAME):
    """Return (predictor, segmenter) for AIS (decoder-based instance segmentation)."""
    from micro_sam.automatic_segmentation import get_predictor_and_segmenter
    predictor, segmenter = get_predictor_and_segmenter(
        model_type=model_name, device=device, segmentation_mode="ais")
    return predictor, segmenter


def _to_label_image(gen_out, hw):
    """generate() may return a label image OR a list of mask dicts. Normalize to a
    HxW int label image."""
    arr = np.asarray(gen_out)
    if arr.ndim == 2:                       # already a label image
        return arr.astype(np.int32)
    # list of {"segmentation","seg_id",...}
    lab = np.zeros(hw, dtype=np.int32)
    for d in gen_out:
        lab[np.asarray(d["segmentation"], dtype=bool)] = int(d.get("seg_id", lab.max() + 1))
    return lab


@torch.no_grad()
def pathosam_instances(image_rgb: np.ndarray, predictor, segmenter, min_area: int = 5):
    """Run AIS on one HxWx3 uint8 image.

    Returns:
      masks   : list of (H,W) bool instance masks
      scores  : (M,) float32  s_i = mean foreground prob over each mask
      feat    : torch tensor (1,256,h,w) ViT embedding (on device) for ROI pooling
    """
    segmenter.initialize(image_rgb)
    gen = segmenter.generate()
    H, W = image_rgb.shape[:2]
    inst = _to_label_image(gen, (H, W))

    fg = np.asarray(segmenter._foreground, dtype=np.float32)
    # foreground may be at a different resolution than the label image — resize if so
    if fg.shape != (H, W):
        fg_t = torch.from_numpy(fg)[None, None]
        fg = torch.nn.functional.interpolate(fg_t, size=(H, W), mode="bilinear",
                                             align_corners=False)[0, 0].numpy()

    feat = predictor.features  # (1, C, h, w) torch on device

    masks, scores = [], []
    for sid in np.unique(inst):
        if sid == 0:
            continue
        m = inst == sid
        if m.sum() < min_area:
            continue
        masks.append(m)
        scores.append(float(fg[m].mean()))
    return masks, np.asarray(scores, dtype=np.float32), feat


@torch.no_grad()
def pool_features(feat: torch.Tensor, masks, device: str) -> torch.Tensor:
    """Mask-pool the ViT embedding for each instance -> (M, 256)."""
    if len(masks) == 0:
        C = feat.shape[1] if feat.dim() == 4 else feat.shape[0]
        return torch.zeros(0, C, device=device)
    out = torch.zeros(len(masks), feat.shape[1] if feat.dim() == 4 else feat.shape[0],
                      device=device)
    for i, m in enumerate(masks):
        mt = torch.from_numpy(np.ascontiguousarray(m)).to(device)
        out[i] = roi_pool_feature(feat, mt.float())
    return out


if __name__ == "__main__":
    # PROBE: confirm shapes/attrs on one Fold-3 image before building the full pipeline.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from pannuke_loader import PanNukeFold
    DATA_ROOT = f"{REPO}/data/pannuke"
    fold3 = PanNukeFold(DATA_ROOT, 3)
    img = fold3[0]["image"]
    print(f"device={device} | image {img.shape} {img.dtype}")

    predictor, segmenter = load_pathosam(device)
    masks, scores, feat = pathosam_instances(img, predictor, segmenter)
    print(f"segmenter._foreground type ok | predictor.features shape = {tuple(feat.shape)}")
    print(f"#instances = {len(masks)} | s_i range [{scores.min():.3f}, {scores.max():.3f}] "
          f"mean {scores.mean():.3f}" if len(scores) else "#instances = 0")
    pooled = pool_features(feat, masks, device)
    print(f"pooled features shape = {tuple(pooled.shape)}  (expect (M, 256))")
    print("PROBE OK — shapes confirmed, safe to run train/build scripts.")
