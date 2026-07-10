"""
Builder -> sam3_inspect_decoder.ipynb

Tiny CPU/GPU notebook: build SAM3 and ENUMERATE every nn.Linear inside the
decoder, grouped by leaf attribute name, with shapes + counts. This tells us
exactly which module names are LoRA-able (current targets = linear1/linear2 only;
we want to add attention projections to give LoRA more capacity).

nn.MultiheadAttention fuses q/k/v into `in_proj_weight` (a Parameter, NOT a
Linear) so only its `out_proj` is LoRA-able. RoPEAttention may or may not expose
q_proj/k_proj/v_proj as nn.Linear — this notebook shows the truth at runtime.

Attach: hipinhththu/sam3-native-pt. Seconds to run.
"""
from __future__ import annotations
import json
from pathlib import Path

OUT = Path(__file__).parent / "sam3_inspect_decoder.ipynb"

def md(*lines):
    src = [ln if ln.endswith("\n") else ln + "\n" for ln in lines]
    if src: src[-1] = src[-1].rstrip("\n")
    return {"cell_type": "markdown", "metadata": {}, "source": src}

def code(body):
    lines = body.lstrip("\n").splitlines(keepends=True)
    if lines: lines[-1] = lines[-1].rstrip("\n")
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": lines}

cells = []
cells.append(md(
    "# Inspect SAM3 decoder — which nn.Linear modules can LoRA target?",
    "",
    "Current LoRA hits only `linear1`/`linear2` (FFN). To add capacity we want attention",
    "projections too. This lists every `nn.Linear` under the decoder, grouped by leaf name.",
    "",
    "**Attach:** `hipinhththu/sam3-native-pt`.",
))
cells.append(code('''
# IMPORTANT: do ALL pip installs via subprocess BEFORE importing torch/numpy, so the
# numpy ABI fix takes effect in this fresh kernel without a restart.
import subprocess, sys, os
WORK = "/kaggle/working"; REPO = f"{WORK}/sam3_research"; SAM3_DIR = f"{REPO}/sam3"
CKPT = "/kaggle/input/datasets/hipinhththu/sam3-native-pt/sam3.pt"
if not os.path.exists(REPO):
    subprocess.run(["git","clone","https://github.com/duonguwu/sam3_research.git",REPO], check=True)
subprocess.run([sys.executable,"-m","pip","install","-q","-e",SAM3_DIR], check=True)
# `pip install -e sam3` downgrades numpy to 1.x; Kaggle torchvision needs numpy 2.x ->
# restore it (fixes "numpy.dtype size changed, Expected 96 got 88").
subprocess.run([sys.executable,"-m","pip","install","-q","-U","numpy>=2"], check=True)

import torch   # safe now (numpy 2.x on disk, not yet imported)
sys.path.insert(0, SAM3_DIR)
from sam3.model_builder import build_sam3_image_model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = build_sam3_image_model(device=device, eval_mode=True, checkpoint_path=CKPT, load_from_HF=False)
print("SAM3 built.")
'''))
cells.append(md("## Enumerate nn.Linear in the decoder, grouped by leaf attr name"))
cells.append(code('''
import torch.nn as nn
from collections import defaultdict, Counter

PATH_KEY = "decoder"   # same filter inject_lora uses (path_must_contain)
groups = defaultdict(list)     # leaf attr name -> list of (full_path, in, out)
mha_count = 0

for parent_name, parent in model.named_modules():
    if PATH_KEY not in parent_name:
        continue
    if isinstance(parent, nn.MultiheadAttention):
        mha_count += 1
    for attr, child in parent.named_children():
        if isinstance(child, nn.Linear):
            groups[attr].append((f"{parent_name}.{attr}", child.in_features, child.out_features))

print(f"nn.MultiheadAttention modules under '{PATH_KEY}': {mha_count} "
      f"(their q/k/v are fused in in_proj_weight -> NOT Linear; only out_proj is)")
print("=" * 78)
print(f"{'leaf attr name':22s} | {'count':>5s} | {'example shape (in->out)':>24s}")
print("-" * 78)
for attr in sorted(groups, key=lambda a: -len(groups[a])):
    n = len(groups[attr])
    ex_in, ex_out = groups[attr][0][1], groups[attr][0][2]
    cur = "  <-- CURRENT TARGET" if attr in ("linear1", "linear2") else ""
    print(f"{attr:22s} | {n:5d} | {ex_in:>10d} -> {ex_out:<10d}{cur}")
print("=" * 78)
print("\\nLoRA-able attention names present (besides linear1/linear2):")
attn_like = [a for a in groups if a not in ("linear1","linear2")]
print("  ", attn_like if attn_like else "(none besides out_proj?)")
'''))
cells.append(md(
    "## Recommendation",
    "",
    "- Any name listed above is `nn.Linear` → can be added to `DEFAULT_LORA_TARGETS`.",
    "- Expect at least `out_proj` (attention output). If `q_proj/k_proj/v_proj` appear,",
    "  add them too for max capacity.",
    "- Paste the table back so we set the exact stronger-LoRA target set + rank.",
))
nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name":"Python 3","language":"python","name":"python3"},
                   "language_info": {"name":"python","version":"3.11"}},
      "nbformat": 4, "nbformat_minor": 5}
OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {OUT.name}: {len(cells)} cells")
