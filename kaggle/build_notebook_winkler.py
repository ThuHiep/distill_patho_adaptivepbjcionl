"""Build sam3_pathosam_winkler.ipynb from the self-contained Kaggle cell .py.
Splits the script into logical cells (setup / Part 1 / Part 2) + a markdown intro.
  python kaggle/build_notebook_winkler.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "sam3_pathosam_winkler_kaggle.py"
OUT = HERE / "sam3_pathosam_winkler.ipynb"

text = SRC.read_text(encoding="utf-8")

# strip the module docstring (turn it into a markdown cell instead)
body = text.split('"""', 2)[2].lstrip("\n")

P1 = "# ================================================================= PART 1"
P2 = "# ================================================================= PART 2"
setup, rest = body.split(P1, 1)
part1, part2 = rest.split(P2, 1)

intro = """# PathoSAM → NuInsSeg — Winkler / Interval-score evaluation

Self-contained (no external lib). Evaluates the conformal mechanisms under extreme
cross-dataset shift with the **Winkler/Interval score** (lower = better; penalises
width **and** non-coverage).

**Run on Kaggle:**
1. Upload `pathosam_predictions.pkl` and `pathosam_nuinsseg_preds.pkl` as a Kaggle Dataset.
2. *Add Data* → select that dataset (lands under `/kaggle/input/...`).
3. Run all cells — the loader auto-finds the two pickles under `/kaggle/input`.
   If filenames differ, edit `PAN_PKL` / `NU_PKL` in the setup cell.

CPU only, a few seconds — no GPU needed.

**Output:** (1) Tables 8f/9a with a Winkler column; (2) mechanism analysis on 2
synthetic streams (W@90 matched-coverage width, per-segment conditional coverage, Winkler).
Headline: **Adaptive PB-JCI Online** has the lowest Winkler (108.67), and width is
*not* the contribution — conditional validity is."""


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": src.strip("\n").splitlines(keepends=True)}


nb = {
    "cells": [
        md(intro),
        code("# ===== Setup: data locate + inlined conformal core + load =====\n" + setup),
        code(P1 + part1),
        code(P2 + part2),
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4, "nbformat_minor": 5,
}

OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"wrote {OUT}  ({len(nb['cells'])} cells)")
