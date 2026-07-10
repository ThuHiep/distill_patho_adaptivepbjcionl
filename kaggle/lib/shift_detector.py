from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
import torch

def gaussian_kernel(x: torch.Tensor, y: torch.Tensor,
                    sigmas=(1.0, 2.0, 4.0, 8.0, 16.0)) -> torch.Tensor:
    dist = torch.cdist(x, y, p=2) ** 2  
    k = torch.zeros_like(dist)
    for s in sigmas:
        k = k + torch.exp(-dist / (2 * s ** 2))
    return k / len(sigmas)

def mmd_squared(x: torch.Tensor, y: torch.Tensor,
                sigmas=(1.0, 2.0, 4.0, 8.0, 16.0)) -> float:
    n, m = x.shape[0], y.shape[0]
    Kxx = gaussian_kernel(x, x, sigmas)
    Kyy = gaussian_kernel(y, y, sigmas)
    Kxy = gaussian_kernel(x, y, sigmas)

    
    Kxx_off = (Kxx.sum() - Kxx.diag().sum()) / max(n * (n - 1), 1)
    Kyy_off = (Kyy.sum() - Kyy.diag().sum()) / max(m * (m - 1), 1)
    Kxy_avg = Kxy.mean()

    mmd2 = Kxx_off + Kyy_off - 2 * Kxy_avg
    return float(max(mmd2.item(), 0.0))  

def wasserstein_1d_mean(x: np.ndarray, y: np.ndarray) -> float:
    try:
        from scipy.stats import wasserstein_distance
    except ImportError:
        raise RuntimeError("scipy required for Wasserstein")
    D = x.shape[1]
    wds = [wasserstein_distance(x[:, d], y[:, d]) for d in range(D)]
    return float(np.mean(wds))

def energy_distance_mean(x: np.ndarray, y: np.ndarray) -> float:
    try:
        from scipy.stats import energy_distance
    except ImportError:
        raise RuntimeError("scipy required for Energy distance")
    D = x.shape[1]
    eds = [energy_distance(x[:, d], y[:, d]) for d in range(D)]
    return float(np.mean(eds))

def _as_torch(a):
    return a.float() if torch.is_tensor(a) else torch.as_tensor(np.asarray(a)).float()

def _as_np(a):
    return a.detach().cpu().numpy() if torch.is_tensor(a) else np.asarray(a)

def compute_mmd(x, y):
    return mmd_squared(_as_torch(x), _as_torch(y))

def compute_wasserstein(x, y):
    return wasserstein_1d_mean(_as_np(x), _as_np(y))

def compute_energy(x, y):
    return energy_distance_mean(_as_np(x), _as_np(y))

@torch.no_grad()
def extract_sam3_features(model, transform, pil_imgs, device: str = "cuda",
                          pool: str = "mean",
                          desc: str = "extract") -> torch.Tensor:
    from torchvision.transforms import v2
    try:
        from tqdm import tqdm
        iterator = tqdm(pil_imgs, desc=desc, leave=False)
    except ImportError:
        iterator = pil_imgs

    feats = []
    for idx, pil in enumerate(iterator):
        image = v2.functional.to_image(pil).to(device)
        image = transform(image).unsqueeze(0)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            backbone_out = model.backbone.forward_image(image)

        
        if "vision_features" in backbone_out:
            f = backbone_out["vision_features"]
        elif "image_features" in backbone_out:
            f = backbone_out["image_features"]
        elif "backbone_fpn" in backbone_out:
            
            f = backbone_out["backbone_fpn"][-1]
        else:
            
            f = next((v for v in backbone_out.values()
                      if isinstance(v, torch.Tensor) and v.dim() == 4), None)
            if f is None:
                raise RuntimeError(f"Cannot extract features from "
                                   f"backbone_out keys: {list(backbone_out.keys())}")

        
        if f.dim() == 4:
            f = f.mean(dim=(2, 3))
        
        
        
        feats.append(f.float().detach().cpu())

        
        
        
        del image, backbone_out, f
        if (idx + 1) % 10 == 0:
            torch.cuda.empty_cache()

    return torch.cat(feats, dim=0)

class ShiftDetector:

    def __init__(self, method: str = "mmd", device: str = "cuda"):
        assert method in ("mmd", "wasserstein", "energy"), method
        self.method = method
        self.device = device
        self.ref_feats: Optional[torch.Tensor] = None

    def fit_reference(self, model, transform, ref_pil_imgs):
        self.ref_feats = extract_sam3_features(
            model, transform, ref_pil_imgs, device=self.device
        )
        return self

    def score(self, model, transform, test_pil_imgs) -> float:
        if self.ref_feats is None:
            raise RuntimeError("Call fit_reference first")
        test_feats = extract_sam3_features(
            model, transform, test_pil_imgs, device=self.device
        )
        if self.method == "mmd":
            
            return mmd_squared(
                self.ref_feats.to(self.device).float(),
                test_feats.to(self.device).float(),
            )
        elif self.method == "wasserstein":
            return wasserstein_1d_mean(self.ref_feats.numpy(), test_feats.numpy())
        else:  
            return energy_distance_mean(self.ref_feats.numpy(), test_feats.numpy())

def apply_hed_shift(img_np: np.ndarray, severity: str = "mild") -> np.ndarray:
    try:
        from skimage.color import rgb2hed, hed2rgb
    except ImportError:
        raise RuntimeError("scikit-image required")

    scales = {"mild": (0.05, 0.05), "moderate": (0.15, 0.1), "severe": (0.3, 0.2)}
    sigma, bias = scales.get(severity, scales["mild"])

    rgb = img_np.astype(np.float32) / 255.0
    hed = rgb2hed(rgb)
    
    for c in range(3):
        hed[..., c] = hed[..., c] * (1 + np.random.normal(0, sigma)) +                      np.random.normal(0, bias)
    rgb_shift = hed2rgb(hed)
    rgb_shift = np.clip(rgb_shift * 255, 0, 255).astype(np.uint8)
    return rgb_shift

def apply_blur_shift(img_np: np.ndarray, severity: str = "mild") -> np.ndarray:
    try:
        from scipy.ndimage import gaussian_filter
    except ImportError:
        raise RuntimeError("scipy required")
    sigmas = {"mild": 1.0, "moderate": 2.5, "severe": 5.0}
    s = sigmas.get(severity, 1.0)
    return gaussian_filter(img_np, sigma=(s, s, 0)).astype(np.uint8)

def apply_hsv_jitter(img_np: np.ndarray, severity: str = "mild") -> np.ndarray:
    try:
        import cv2
    except ImportError:
        
        scales = {"mild": 0.1, "moderate": 0.25, "severe": 0.5}
        s = scales.get(severity, 0.1)
        shift = np.random.normal(0, s * 255, size=3)
        return np.clip(img_np + shift, 0, 255).astype(np.uint8)

    scales = {"mild": (10, 20), "moderate": (30, 50), "severe": (60, 100)}
    h_shift, s_shift = scales.get(severity, (10, 20))
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV).astype(np.int32)
    hsv[..., 0] = (hsv[..., 0] + np.random.randint(-h_shift, h_shift)) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] + np.random.randint(-s_shift, s_shift), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
