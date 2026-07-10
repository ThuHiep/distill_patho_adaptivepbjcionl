"""Build sam3_pathosam_figures.ipynb from the self-contained figures .py.
  python kaggle/build_notebook_figures.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "sam3_pathosam_figures_kaggle.py"
OUT = HERE / "sam3_pathosam_figures.ipynb"

text = SRC.read_text(encoding="utf-8")
body = text.split('"""', 2)[2].lstrip("\n")

F1 = "# === FIG 1 ==="
F2 = "# === FIG 2 ==="
F3 = "# === FIG 3 ==="
F4 = "# === FIG 4 ==="
setup, rest = body.split(F1, 1)
f1, rest = rest.split(F2, 1)
f2, rest = rest.split(F3, 1)
f3, f4 = rest.split(F4, 1)

intro = """# PathoSAM → NuInsSeg — Figures cho báo cáo

Self-contained (không import lib ngoài). Tạo 4 hình minh họa đóng góp:
- **F1** — trade-off coverage–Winkler (Adaptive PB-JCI Online ở góc tốt nhất)
- **F2** — coverage theo thời gian dưới shift đột ngột (tĩnh sụp, adaptive kéo về 90%)
- **F3** — conditional coverage theo regime (tĩnh under-cover, adaptive giữ ~90%)
- **F4** — so baseline hiện đại trực diện (Bảng 9a: naive/weighted sụp; Winkler ours thấp nhất)

**Chạy trên Kaggle:**
1. Upload `pathosam_predictions.pkl` + `pathosam_nuinsseg_preds.pkl` thành Kaggle Dataset.
2. *Add Data* → chọn dataset (vào `/kaggle/input/...`).
3. Run all — hình hiện inline và lưu PNG vào working dir. Tên file khác thì sửa `PAN_PKL`/`NU_PKL`.

CPU, vài giây. Chỉ cần `numpy`, `matplotlib`."""


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.splitlines(keepends=True)}


def code(s):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": s.strip("\n").splitlines(keepends=True)}


nb = {
    "cells": [
        md(intro),
        code("# Setup: locate data + inlined conformal core + load + run helpers\n" + setup),
        code(F1 + f1),
        code(F2 + f2),
        code(F3 + f3),
        code(F4 + f4),
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4, "nbformat_minor": 5,
}
OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"wrote {OUT} ({len(nb['cells'])} cells)")
