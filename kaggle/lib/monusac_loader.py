"""
MoNuSAC loader (K=4) — dataset da lop SACH cho PathoSAM (eval-only, da-organ, KHONG nam
trong Lizard nen PathoSAM chua tung train -> sach). Doc monusac_converted.pkl tao boi
monusac_converter.py. Interface giong ConicSet/conic_split de tai su dung run_*_typehead/
build/conformal.

  ds = MonusacSet(PKL, indices) ; s = ds[i] ->
     {"image":(H,W,3)uint8, "type_map":(H,W)int8 (-1 bg,0..3), "inst":(H,W)int,
      "counts":(4)float32, "source": patient}
  monusac_split(PKL, frac_cal=0.5, seed=0) -> (cal_idx, test_idx)  [tach THEO patient]

Anh MoNuSAC kich thuoc KHAC nhau (559x602 trung binh) -> luu list per-image, khong mmap.
"""
from __future__ import annotations
import io
import pickle
import numpy as np

MONUSAC_CLASSES = ["Epithelial", "Lymphocyte", "Macrophage", "Neutrophil"]
K = len(MONUSAC_CLASSES)


# ---- PNG codecs (pkl luu anh/label da nen PNG -> nho ~10x, self-contained) ----
def encode_rgb(arr):
    from PIL import Image
    b = io.BytesIO(); Image.fromarray(np.ascontiguousarray(arr), "RGB").save(b, format="PNG"); return b.getvalue()


def decode_rgb(buf):
    from PIL import Image
    return np.asarray(Image.open(io.BytesIO(buf)).convert("RGB"), dtype=np.uint8)


def encode_inst(arr):
    from PIL import Image
    a = np.clip(arr, 0, 65535).astype(np.uint16)
    b = io.BytesIO(); Image.fromarray(a, mode="I;16").save(b, format="PNG"); return b.getvalue()


def decode_inst(buf):
    from PIL import Image
    return np.asarray(Image.open(io.BytesIO(buf)), dtype=np.int32)


def encode_tmap(arr):
    from PIL import Image
    a = (arr.astype(np.int16) + 1).clip(0, 255).astype(np.uint8)   # -1..K-1 -> 0..K
    b = io.BytesIO(); Image.fromarray(a, mode="L").save(b, format="PNG"); return b.getvalue()


def decode_tmap(buf):
    from PIL import Image
    return (np.asarray(Image.open(io.BytesIO(buf)), dtype=np.int16) - 1).astype(np.int8)


def _decode_item(it):
    """Tra ve item voi mang numpy (decode PNG neu da nen)."""
    if not it.get("encoded"):
        return it
    return {"image": decode_rgb(it["image"]), "inst": decode_inst(it["inst"]),
            "type_map": decode_tmap(it["type_map"]),
            "counts": np.asarray(it["counts"], np.float32), "source": it["source"]}


def _load(pkl):
    with open(pkl, "rb") as f:
        d = pickle.load(f)
    return d["items"], d.get("classes", MONUSAC_CLASSES)


class MonusacSet:
    def __init__(self, pkl, indices=None):
        self.items, self.classes = _load(pkl)
        n = len(self.items)
        self.indices = np.arange(n) if indices is None else np.asarray(indices, dtype=int)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        it = _decode_item(self.items[int(self.indices[i])])
        return {"image": np.asarray(it["image"], dtype=np.uint8),
                "type_map": np.asarray(it["type_map"], dtype=np.int8),
                "inst": np.asarray(it["inst"], dtype=np.int32),
                "counts": np.asarray(it["counts"], dtype=np.float32),
                "source": it["source"]}


def monusac_split(pkl, frac_cal=0.5, seed=0):
    """Tach cal/test THEO patient (source) -> khong de nhan cua cung benh nhan ca hai ben."""
    items, _ = _load(pkl)
    src = np.array([it["source"] for it in items], dtype=object)
    n = len(items)
    rng = np.random.RandomState(seed)
    uniq = np.array(sorted(set(src.tolist())), dtype=object)
    if len(uniq) >= 4:
        perm = rng.permutation(len(uniq))
        n_cal = max(1, int(round(frac_cal * len(uniq))))
        cal_src = set(uniq[perm[:n_cal]].tolist())
        mask = np.array([s in cal_src for s in src])
        cal_idx, test_idx = np.where(mask)[0], np.where(~mask)[0]
        mode = f"by-patient ({n_cal}/{len(uniq)} patients -> cal)"
    else:
        perm = rng.permutation(n)
        n_cal = int(round(frac_cal * n))
        cal_idx, test_idx = np.sort(perm[:n_cal]), np.sort(perm[n_cal:])
        mode = "by-index (it patient)"
    print(f"[monusac_split] N={n} | {mode} | cal={len(cal_idx)} test={len(test_idx)}")
    return cal_idx, test_idx


if __name__ == "__main__":
    import sys
    pkl = sys.argv[1] if len(sys.argv) > 1 else "/workspace/sam3_research/data/monusac_converted.pkl"
    cal, test = monusac_split(pkl)
    ds = MonusacSet(pkl, test)
    s = ds[0]
    print("classes:", MONUSAC_CLASSES)
    print("sample0:", {k: (v.shape if hasattr(v, "shape") else v) for k, v in s.items()})
    print("type_map uniq:", np.unique(s["type_map"]), "| counts:", s["counts"],
          "| inst max:", s["inst"].max())
