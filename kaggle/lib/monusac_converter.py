"""
MoNuSAC converter: .svs/.tif + ImageScope XML (polygon per nhan, gan lop) -> per-image
(image, instance map, class map, counts) -> luu monusac_converted.pkl.

MoNuSAC release goc: 1 thu muc/patient, moi cap (anh .svs|.tif|.png, annotation .xml).
XML kieu Aperio/ImageScope:
  <Annotations>
    <Annotation> <Attributes><Attribute Name="Epithelial"/></Attributes>
       <Regions><Region><Vertices><Vertex X=".." Y=".."/>...</Vertices></Region>...</Regions>
    </Annotation> ...
  </Annotations>
Lop nam o Attribute@Name (hoac @Value). "Ambiguous" -> BO (khong tinh).

Output: MONUSAC_ROOT/../monusac_converted.pkl =
  {"items":[{image(H,W,3)uint8, inst(H,W)int32, type_map(H,W)int8 (-1 bg,0..3),
             counts(4)float32, source(patient str)}], "classes": MONUSAC_CLASSES}
=> monusac_loader.MonusacSet doc thang.

Doc anh: thu tifffile -> PIL -> openslide. Raster polygon bang PIL ImageDraw (khong can cv2).

Run:  python monusac_converter.py /path/to/MoNuSAC_root [out.pkl]
"""
from __future__ import annotations
import os, sys, glob, pickle
import numpy as np
import xml.etree.ElementTree as ET

MONUSAC_CLASSES = ["Epithelial", "Lymphocyte", "Macrophage", "Neutrophil"]
_CLS2IDX = {c.lower(): i for i, c in enumerate(MONUSAC_CLASSES)}
K = len(MONUSAC_CLASSES)


def _read_image(path):
    """Doc anh -> (H,W,3) uint8. Thu tifffile, PIL, openslide."""
    try:
        import tifffile
        arr = tifffile.imread(path)
        return _to_rgb_uint8(arr)
    except Exception:
        pass
    try:
        from PIL import Image
        return _to_rgb_uint8(np.asarray(Image.open(path).convert("RGB")))
    except Exception:
        pass
    try:
        import openslide
        sl = openslide.OpenSlide(path)
        img = sl.read_region((0, 0), 0, sl.level_dimensions[0]).convert("RGB")
        return _to_rgb_uint8(np.asarray(img))
    except Exception as e:
        raise RuntimeError(f"khong doc duoc anh {path}: {e}")


def _to_rgb_uint8(arr):
    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, -1)
    if arr.shape[0] in (3, 4) and arr.ndim == 3 and arr.shape[0] < arr.shape[-1]:
        arr = np.moveaxis(arr, 0, -1)        # (C,H,W)->(H,W,C)
    arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = (255 * (arr.astype(np.float32) / (arr.max() + 1e-6))).clip(0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def _class_of(annotation):
    """Tim ten lop trong mot <Annotation>. Tra ve idx 0..3, hoac None (bg/Ambiguous)."""
    for attr in annotation.iter():
        tag = attr.tag.lower()
        if tag.endswith("attribute"):
            name = (attr.get("Name") or attr.get("Value") or "").strip().lower()
            if name in _CLS2IDX:
                return _CLS2IDX[name]
            if "ambiguous" in name:
                return None
    # fallback: ten lop co the o attribute "Name" cua chinh Annotation
    nm = (annotation.get("Name") or "").strip().lower()
    return _CLS2IDX.get(nm, None)


def _polygons(annotation):
    """Tra ve list cac polygon [(x,y),...] tu cac <Region>/<Vertex>."""
    polys = []
    for region in annotation.iter():
        if not region.tag.lower().endswith("region"):
            continue
        pts = []
        for v in region.iter():
            if v.tag.lower().endswith("vertex"):
                try:
                    pts.append((float(v.get("X")), float(v.get("Y"))))
                except (TypeError, ValueError):
                    pass
        if len(pts) >= 3:
            polys.append(pts)
    return polys


def _rasterize(polys_by_cls, hw):
    """polys_by_cls: list (cls_idx, polygon). -> inst(H,W)int32, type_map(H,W)int8."""
    from PIL import Image, ImageDraw
    H, W = hw
    inst = np.zeros((H, W), dtype=np.int32)
    tmap = np.full((H, W), -1, dtype=np.int8)
    nid = 0
    for cls_idx, poly in polys_by_cls:
        nid += 1
        m = Image.new("L", (W, H), 0)
        ImageDraw.Draw(m).polygon(poly, outline=1, fill=1)
        mask = np.asarray(m, dtype=bool)
        if mask.sum() == 0:
            nid -= 1
            continue
        inst[mask] = nid
        tmap[mask] = cls_idx
    return inst, tmap


def convert(root, out_pkl=None):
    xmls = sorted(glob.glob(os.path.join(root, "**", "*.xml"), recursive=True))
    if not xmls:
        raise FileNotFoundError(f"khong thay .xml nao duoi {root}")
    print(f"found {len(xmls)} xml annotations under {root}")
    items, skipped = [], 0
    for xp in xmls:
        stem = os.path.splitext(xp)[0]
        img_path = None
        for ext in (".tif", ".tiff", ".png", ".jpg", ".svs"):   # .tif first (.svs can du openslide)
            if os.path.exists(stem + ext):
                img_path = stem + ext; break
        if img_path is None:
            skipped += 1; continue
        try:
            img = _read_image(img_path)
        except Exception as e:
            print("  WARN skip", img_path, e); skipped += 1; continue
        H, W = img.shape[:2]
        try:
            root_el = ET.parse(xp).getroot()
        except ET.ParseError as e:
            print("  WARN bad xml", xp, e); skipped += 1; continue
        polys_by_cls = []
        for ann in root_el.iter():
            if not ann.tag.lower().endswith("annotation"):
                continue
            cls = _class_of(ann)
            if cls is None:
                continue
            for poly in _polygons(ann):
                polys_by_cls.append((cls, poly))
        if not polys_by_cls:
            skipped += 1; continue
        inst, tmap = _rasterize(polys_by_cls, (H, W))
        counts = np.array([(tmap == k).any() and
                           len(np.unique(inst[tmap == k])) or 0 for k in range(K)],
                          dtype=np.float32)
        # counts dung hon: dem instance id rieng moi lop
        counts = np.zeros(K, dtype=np.float32)
        for k in range(K):
            ids = np.unique(inst[tmap == k]); ids = ids[ids > 0]
            counts[k] = len(ids)
        patient = os.path.basename(os.path.dirname(xp))
        from monusac_loader import encode_rgb, encode_inst, encode_tmap
        items.append({"image": encode_rgb(img), "inst": encode_inst(inst),
                      "type_map": encode_tmap(tmap), "counts": counts,
                      "source": patient, "encoded": True})       # PNG-nen -> pkl nho
    print(f"converted {len(items)} images | skipped {skipped}")
    if not items:
        raise RuntimeError("khong convert duoc anh nao - kiem tra format XML/anh")
    tot = np.sum([it["counts"] for it in items], axis=0)
    print("tong nhan moi lop:", dict(zip(MONUSAC_CLASSES, tot.astype(int).tolist())))
    out_pkl = out_pkl or os.path.join(os.path.dirname(root.rstrip("/\\")), "monusac_converted.pkl")
    with open(out_pkl, "wb") as f:
        pickle.dump({"items": items, "classes": MONUSAC_CLASSES}, f)
    print(f"saved -> {out_pkl}  ({len(items)} images)")
    return out_pkl


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python monusac_converter.py /path/to/MoNuSAC_root [out.pkl]"); sys.exit(1)
    convert(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
