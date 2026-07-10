import os
import sys

REPO_DIR = "/workspace/sam3_research"
SAM3_DIR = f"{REPO_DIR}/sam3"
LIB_DIR  = f"{REPO_DIR}/kaggle/lib"
WORK     = f"{REPO_DIR}/work"  

DATA_ROOT       = f"{REPO_DIR}/data/pannuke"
CHECKPOINT_PATH = f"{REPO_DIR}/checkpoints/sam3.pt"
LORA_CKPT_PATH  = f"{REPO_DIR}/checkpoints/sam3_lora_rank16_final.pt"
TYPEHEAD_PATH   = f"{REPO_DIR}/checkpoints/type_head_final.pt"

CHECKPOINTS_OUT = f"{REPO_DIR}/checkpoints_multiseed"

for p in [REPO_DIR, SAM3_DIR, LIB_DIR, WORK]:
    if p not in sys.path:
        sys.path.insert(0, p)
os.makedirs(WORK, exist_ok=True)
os.makedirs(CHECKPOINTS_OUT, exist_ok=True)

def verify_env():
    checks = {
        "Repo":        REPO_DIR,
        "SAM3 dir":    SAM3_DIR,
        "Lib dir":     LIB_DIR,
        "PanNuke":     DATA_ROOT,
        "SAM3 ckpt":   CHECKPOINT_PATH,
        "LoRA ckpt":   LORA_CKPT_PATH,
        "TypeHead":    TYPEHEAD_PATH,
    }
    for name, path in checks.items():
        if os.path.exists(path):
            print(f"  [OK]   {name:12s}: {path}")
        else:
            print(f"  [MISS] {name:12s}: {path}")

if __name__ == "__main__":
    print("Vast.ai environment paths:")
    verify_env()
