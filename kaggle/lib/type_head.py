from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TypeHead(nn.Module):
    def __init__(self, in_dim: int = 256, hidden_dim: int = 128,
                 num_classes: int = 5, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

def roi_pool_feature(backbone_features: torch.Tensor,
                     mask: torch.Tensor) -> torch.Tensor:
    if backbone_features.dim() == 4:
        backbone_features = backbone_features[0]  
    D, Hf, Wf = backbone_features.shape

    
    mask_resized = F.interpolate(
        mask.float()[None, None],
        size=(Hf, Wf),
        mode='bilinear',
        align_corners=False,
    )[0, 0]  

    
    mass = mask_resized.sum()
    if mass < 1e-3:
        return backbone_features.mean(dim=(1, 2))  
    pooled = (backbone_features * mask_resized).sum(dim=(1, 2)) / mass
    return pooled  

def compute_iou_matrix(pred_masks: List[np.ndarray],
                       gt_masks: List[np.ndarray]) -> np.ndarray:
    N_p, N_g = len(pred_masks), len(gt_masks)
    iou_matrix = np.zeros((N_p, N_g), dtype=np.float32)
    for i, p in enumerate(pred_masks):
        for j, g in enumerate(gt_masks):
            inter = np.logical_and(p, g).sum()
            union = np.logical_or(p, g).sum()
            iou_matrix[i, j] = inter / (union + 1e-6) if union > 0 else 0.0
    return iou_matrix

def hungarian_match(iou_matrix: np.ndarray,
                    iou_thresh: float = 0.3) -> List[Tuple[int, int]]:
    try:
        from scipy.optimize import linear_sum_assignment
        
        row_ind, col_ind = linear_sum_assignment(-iou_matrix)
        matches = []
        for r, c in zip(row_ind, col_ind):
            if iou_matrix[r, c] >= iou_thresh:
                matches.append((int(r), int(c)))
        return matches
    except ImportError:
        
        matches = []
        used_g = set()
        
        flat = [(iou_matrix[i, j], i, j)
                for i in range(iou_matrix.shape[0])
                for j in range(iou_matrix.shape[1])]
        flat.sort(reverse=True)
        used_p = set()
        for iou, i, j in flat:
            if iou < iou_thresh:
                break
            if i in used_p or j in used_g:
                continue
            matches.append((i, j))
            used_p.add(i)
            used_g.add(j)
        return matches

def extract_gt_instances(sample: dict, cell_types: List[str]
                         ) -> Tuple[List[np.ndarray], List[int]]:
    masks_per_type = sample["masks"]
    gt_masks = []
    gt_classes = []
    for type_idx in range(5):
        inst_ids = np.unique(masks_per_type[type_idx])
        for inst_id in inst_ids:
            if inst_id == 0:
                continue
            mask = (masks_per_type[type_idx] == inst_id).astype(bool)
            if mask.sum() < 5:  
                continue
            gt_masks.append(mask)
            gt_classes.append(type_idx)
    return gt_masks, gt_classes

def per_class_counts(pred_scores: np.ndarray,
                     pred_probs: np.ndarray) -> np.ndarray:
    counts = (pred_scores[:, None] * pred_probs).sum(axis=0)  
    return counts

def per_class_variance(pred_scores: np.ndarray,
                       pred_probs: np.ndarray) -> np.ndarray:
    weighted = pred_scores[:, None] * pred_probs  
    var = (weighted * (1.0 - weighted)).sum(axis=0)
    return var
