from __future__ import annotations
from pathlib import Path
from typing import Iterator, List, Optional, Tuple
import numpy as np

CELL_TYPES: List[str] = [
    "Neoplastic", "Inflammatory", "Connective", "Dead", "Epithelial",
]

DEFAULT_ROOT = Path("/kaggle/input/datasets/hipinhththu/pannuke")

def _to_uint8(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.uint8:
        return arr
    if arr.max() <= 1.0:
        return (arr * 255).round().clip(0, 255).astype(np.uint8)
    return arr.clip(0, 255).astype(np.uint8)

class PanNukeFold:

    def __init__(self, root, fold: int):
        self.root = Path(root)
        self.fold = fold
        sub = f"Fold {fold}"
        f = f"fold{fold}"
        candidates = [
            self.root / f / sub,   
            self.root / sub,        
        ]
        base = next((c for c in candidates if (c / "images" / f / "images.npy").exists()),
                    None)
        if base is None:
            raise FileNotFoundError(
                f"Không tìm thấy Fold {fold} ở: " + " hoặc ".join(str(c) for c in candidates)
            )
        self.images_path = base / "images" / f / "images.npy"
        self.masks_path  = base / "masks"  / f / "masks.npy"

        
        type_candidates = [
            base / "images" / f / "types.npy",   
            base / "types" / f / "types.npy",     
            base / "types.npy",
            self.root / f / "types.npy",
            self.root / "types" / f / "types.npy",
        ]
        self.types_path = next((p for p in type_candidates if p.exists()), None)

        for p in (self.images_path, self.masks_path):
            if not p.exists():
                raise FileNotFoundError(f"Missing: {p}")

        self.images = np.load(self.images_path, mmap_mode="r")
        self.masks  = np.load(self.masks_path,  mmap_mode="r")
        if self.types_path is not None:
            self.tissue_types = np.load(self.types_path, allow_pickle=True)
        else:
            n = self.images.shape[0]
            self.tissue_types = np.array(["unknown"] * n, dtype=object)
            print(f"[Fold {fold}] WARN: types.npy không tìm thấy ở:")
            for c in type_candidates:
                print(f"  - {c}")
            print(f"[Fold {fold}] Dùng placeholder 'unknown' cho {n} ảnh.")

        assert self.images.shape[0] == self.masks.shape[0] == self.tissue_types.shape[0]
        assert self.images.shape[1:] == (256, 256, 3), f"unexpected: {self.images.shape}"
        assert self.masks.shape[1:]  == (256, 256, 6), f"unexpected: {self.masks.shape}"

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, idx: int) -> dict:
        img = _to_uint8(np.array(self.images[idx]))
        m_all = np.array(self.masks[idx], dtype=np.int32)
        masks_per_type = m_all[..., :5].transpose(2, 0, 1)
        counts = np.array(
            [int(np.unique(masks_per_type[k]).size - 1) for k in range(5)],
            dtype=np.int32,
        )
        return {
            "image": img,
            "masks": masks_per_type,
            "counts": counts,
            "tissue": str(self.tissue_types[idx]),
            "fold": self.fold,
            "idx": idx,
        }

    def iter_samples(self, start: int = 0, end: Optional[int] = None) -> Iterator[dict]:
        end = end or len(self)
        for i in range(start, end):
            yield self[i]

def load_all_folds(root=DEFAULT_ROOT) -> Tuple[PanNukeFold, PanNukeFold, PanNukeFold]:
    root = Path(root)
    return PanNukeFold(root, 1), PanNukeFold(root, 2), PanNukeFold(root, 3)
