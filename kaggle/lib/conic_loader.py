"""
CoNIC loader (K=6) — dataset da lop THU HAI de chung minh PB-JCI tong quat theo K
(khong an may K=5 cua PanNuke). Doc release numpy chuan cua CoNIC challenge:

  CONIC_ROOT/
    images.npy      (N, 256, 256, 3) uint8
    labels.npy      (N, 256, 256, 2) int  -> [...,0]=instance map, [...,1]=class map
    counts.csv      N hang, 6 cot = so nhan moi lop (thu tu CONIC_CLASSES)
    patch_info.csv  (tuy chon) cot source de tach train/test theo WSI (tranh leakage)

Class map: 0=background, 1..6 = 6 lop (xem CONIC_CLASSES). type_map tra ve dung quy uoc
PathoSAM: -1=background, 0..5 = lop.

Interface giong PanNukeFold de tai su dung run_pathosam_* scripts:
  ds = ConicSet(CONIC_ROOT, indices)
  s = ds[i] -> {"image":(H,W,3)uint8, "type_map":(H,W)int8, "inst":(H,W)int,
                "counts":(6,)float32, "source":str}
  conic_split(CONIC_ROOT, frac_cal=0.5, seed=0) -> (cal_idx, test_idx)  [theo source neu co]
"""
from __future__ import annotations
import os
import numpy as np

# Thu tu lop CHUAN cua CoNIC (Lizard taxonomy)
CONIC_CLASSES = ["Neutrophil", "Epithelial", "Lymphocyte", "Plasma", "Eosinophil", "Connective"]
K = len(CONIC_CLASSES)


def _paths(root):
    return (os.path.join(root, "images.npy"),
            os.path.join(root, "labels.npy"),
            os.path.join(root, "counts.csv"),
            os.path.join(root, "patch_info.csv"))


class ConicSet:
    """Lazy view tren images.npy/labels.npy + counts.csv, gioi han boi `indices`."""
    def __init__(self, root, indices=None, mmap=True):
        img_p, lab_p, cnt_p, _ = _paths(root)
        mode = "r" if mmap else None
        self.images = np.load(img_p, mmap_mode=mode)            # (N,H,W,3)
        self.labels = np.load(lab_p, mmap_mode=mode)            # (N,H,W,2)
        self.counts = _load_counts(cnt_p, len(self.images))     # (N,6) float32
        n = len(self.images)
        self.indices = np.arange(n) if indices is None else np.asarray(indices, dtype=int)
        self.source = _load_source(root, n)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        gi = int(self.indices[i])
        lab = np.asarray(self.labels[gi])
        inst = lab[..., 0].astype(np.int32)
        cls = lab[..., 1].astype(np.int32)                     # 0=bg, 1..6
        tmap = np.full(cls.shape, -1, dtype=np.int8)
        fg = cls > 0
        tmap[fg] = (cls[fg] - 1).astype(np.int8)               # -> 0..5
        return {"image": np.asarray(self.images[gi]).astype(np.uint8),
                "type_map": tmap, "inst": inst,
                "counts": self.counts[gi].astype(np.float32),
                "source": self.source[gi]}


def _load_counts(cnt_p, n):
    if not os.path.exists(cnt_p):
        raise FileNotFoundError(f"counts.csv not found: {cnt_p}")
    import csv
    rows = []
    with open(cnt_p, newline="") as f:
        rd = csv.reader(f)
        header = next(rd)
        # neu header la so (khong phai ten lop) thi do la dong du lieu -> reset
        try:
            float(header[0]); rows.append([float(x) for x in header])
        except ValueError:
            pass
        for r in rd:
            if r:
                rows.append([float(x) for x in r])
    arr = np.asarray(rows, dtype=np.float32)
    assert arr.shape[0] == n, f"counts rows {arr.shape[0]} != #images {n}"
    assert arr.shape[1] == K, f"counts cols {arr.shape[1]} != K={K}"
    return arr


def _load_source(root, n):
    """Cot 'source' tu patch_info.csv neu co (de tach theo WSI); else id rieng tung patch."""
    _, _, _, pi_p = _paths(root)
    if not os.path.exists(pi_p):
        return np.array([f"patch_{i}" for i in range(n)], dtype=object)
    import csv
    src = []
    with open(pi_p, newline="") as f:
        rd = csv.reader(f)
        header = next(rd)
        col = 0
        for j, h in enumerate(header):
            if "source" in h.lower() or "slide" in h.lower() or "wsi" in h.lower():
                col = j; break
        for r in rd:
            src.append(r[col] if r else "?")
    if len(src) != n:                  # header bi nham la data, hoac thieu -> fallback an toan
        return np.array([f"patch_{i}" for i in range(n)], dtype=object)
    return np.array(src, dtype=object)


def conic_split(root, frac_cal=0.5, seed=0):
    """Tach cal/test. Neu co patch_info.source -> tach THEO SOURCE (khong de patch cung WSI
    nam ca hai ben -> tranh leakage). Else tach theo index. Tra ve (cal_idx, test_idx)."""
    img_p, _, _, _ = _paths(root)
    n = len(np.load(img_p, mmap_mode="r"))
    src = _load_source(root, n)
    rng = np.random.RandomState(seed)
    uniq = np.array(sorted(set(src.tolist())), dtype=object)
    if len(uniq) >= 4:                  # du source de tach theo WSI
        perm = rng.permutation(len(uniq))
        n_cal = max(1, int(round(frac_cal * len(uniq))))
        cal_src = set(uniq[perm[:n_cal]].tolist())
        mask = np.array([s in cal_src for s in src])
        cal_idx = np.where(mask)[0]; test_idx = np.where(~mask)[0]
        mode = f"by-source ({n_cal}/{len(uniq)} sources -> cal)"
    else:                               # fallback: tach theo index
        perm = rng.permutation(n)
        n_cal = int(round(frac_cal * n))
        cal_idx, test_idx = np.sort(perm[:n_cal]), np.sort(perm[n_cal:])
        mode = "by-index (khong co source info)"
    print(f"[conic_split] N={n} | {mode} | cal={len(cal_idx)} test={len(test_idx)}")
    return cal_idx, test_idx


if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "/workspace/sam3_research/data/conic"
    cal, test = conic_split(root)
    ds = ConicSet(root, test)
    s = ds[0]
    print("classes:", CONIC_CLASSES)
    print("sample0:", {k: (v.shape if hasattr(v, "shape") else v) for k, v in s.items()})
    print("type_map uniq:", np.unique(s["type_map"]), "| counts:", s["counts"],
          "| inst max:", s["inst"].max())
