from __future__ import annotations
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import v2

def make_transform(resolution: int = 1008):
    return v2.Compose([
        v2.ToDtype(torch.uint8, scale=True),
        v2.Resize(size=(resolution, resolution)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

@torch.no_grad()
def encode_image_frozen(model, transform, pil_img, device: str = "cuda"):
    image = v2.functional.to_image(pil_img).to(device)
    image = transform(image).unsqueeze(0)
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        backbone_out = model.backbone.forward_image(image)

        
        inst_pred = getattr(model, "inst_interactive_predictor", None)
        if inst_pred is not None and "sam2_backbone_out" in backbone_out:
            sam2_bb = backbone_out["sam2_backbone_out"]
            sam2_bb["backbone_fpn"][0] = (
                inst_pred.model.sam_mask_decoder.conv_s0(sam2_bb["backbone_fpn"][0])
            )
            sam2_bb["backbone_fpn"][1] = (
                inst_pred.model.sam_mask_decoder.conv_s1(sam2_bb["backbone_fpn"][1])
            )
    return backbone_out

def encode_text(model, prompt: str, device: str = "cuda"):
    with torch.no_grad():
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            return model.backbone.forward_text([prompt], device=device)

def forward_decoder_with_grad(model, backbone_out: Dict, find_stage,
                              geometric_prompt, device: str = "cuda") -> Dict:
    was_training = model.training
    if was_training:
        model.eval()
    try:
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            outputs = model.forward_grounding(
                backbone_out=backbone_out,
                find_input=find_stage,
                geometric_prompt=geometric_prompt,
                find_target=None,
            )
    finally:
        if was_training:
            model.train()
    return outputs

def semantic_union_mask(outputs: Dict, target_hw: Tuple[int, int]) -> torch.Tensor:
    
    pred_logits = outputs["pred_logits"].float()         
    pred_masks  = outputs["pred_masks"].float()          
    pres_logit  = outputs["presence_logit_dec"].float()  

    
    cls_prob = pred_logits.sigmoid()                   
    pres     = pres_logit.sigmoid().unsqueeze(1)       
    prob     = (cls_prob * pres).squeeze(-1)           

    
    masks = F.interpolate(
        pred_masks, size=target_hw, mode="bilinear", align_corners=False
    ).sigmoid()  

    
    weighted = prob[:, :, None, None] * masks   

    
    
    
    eps = 1e-4
    log_one_minus = torch.log(torch.clamp(1.0 - weighted, min=eps, max=1.0 - eps))
    log_prod = log_one_minus.sum(dim=1)         
    union = 1.0 - torch.exp(torch.clamp(log_prod, min=-20.0))  
    union = torch.clamp(union, min=eps, max=1.0 - eps)  
    return union.squeeze(0)                     

def dice_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    pred = pred.float()
    target = target.float()
    inter = (pred * target).sum()
    return 1.0 - (2.0 * inter + eps) / (pred.sum() + target.sum() + eps)

def bce_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    pred = pred.float()
    target = target.float()
    p = torch.clamp(pred, eps, 1 - eps)
    return -(target * torch.log(p) + (1 - target) * torch.log(1 - p)).mean()

def semantic_seg_loss(pred: torch.Tensor, target: torch.Tensor,
                      bce_weight: float = 0.5,
                      dice_weight: float = 1.0) -> Tuple[torch.Tensor, Dict[str, float]]:
    bce = bce_loss(pred, target)
    dice = dice_loss(pred, target)
    total = bce_weight * bce + dice_weight * dice
    return total, {"bce": float(bce.item()), "dice": float(dice.item()),
                   "loss": float(total.item())}

def compute_loss(pred: torch.Tensor, target: torch.Tensor,
                 bce_weight: float = 0.5, dice_weight: float = 1.0) -> torch.Tensor:
    total, _ = semantic_seg_loss(pred, target, bce_weight, dice_weight)
    return total

@torch.no_grad()
def inference_to_binary(outputs: Dict, target_hw: Tuple[int, int],
                        score_threshold: float = 0.3) -> torch.Tensor:
    pred_logits = outputs["pred_logits"]
    pred_masks  = outputs["pred_masks"]
    pres_logit  = outputs["presence_logit_dec"]

    cls_prob = pred_logits.sigmoid()
    pres = pres_logit.sigmoid().unsqueeze(1)
    prob = (cls_prob * pres).squeeze(-1).squeeze(0)  

    masks = F.interpolate(
        pred_masks, size=target_hw, mode="bilinear", align_corners=False
    ).sigmoid().squeeze(0)  

    keep = prob > score_threshold
    if keep.sum() == 0:
        return torch.zeros(target_hw, dtype=torch.bool, device=pred_logits.device)

    selected = (masks[keep] > 0.5)
    union = selected.any(dim=0)
    return union
