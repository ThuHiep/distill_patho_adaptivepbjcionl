# Vast.ai Setup Guide — SAM3 Q1 Multi-seed Pipeline

Hướng dẫn chi tiết step-by-step để chạy multi-seed Q1 pipeline trên Vast.ai.

## 0. Tổng quan

```
Local Windows  →  Vast.ai instance (A100)  →  Kaggle Datasets
   (SSH)            (run pipeline)              (backup results)
```

Total cost: **~$14-25** for full Q1 multi-seed work.

---

## STEP 1: Tạo account + nạp credit

1. Đăng ký https://vast.ai
2. **Billing** → **Add Credit** → nạp **$20** (đủ cho cả pipeline)

---

## STEP 2: Setup SSH key (local Windows)

Mở PowerShell:

```powershell
# Tạo SSH key cho Vast
ssh-keygen -t ed25519 -f $HOME\.ssh\vast_key
# Press Enter twice (no passphrase)

# Hiện public key (copy nội dung này)
cat $HOME\.ssh\vast_key.pub
```

Copy output (bắt đầu `ssh-ed25519 AAAA...`).

Trên Vast.ai:
- **Account** → **SSH Keys** → Paste → **Add**

---

## STEP 3: Search + Rent instance A100

1. Vast.ai **Console** → **Search**
2. **Filters**:
   - GPU Name: **A100** (40GB recommend) hoặc **RTX 3090** (cheaper)
   - Num GPUs: 1
   - CUDA: >= 11.7
   - Disk Space: >= 50 GB
   - Reliability: >= 0.95
   - Interruptible: **No**
3. **Sort**: $/h cheapest
4. Pick instance với **DLPerf >= 30**
5. Click **RENT** (⚡ icon)
6. Configure:
   - Image: **PyTorch (cuda 12.1)** preset
   - Disk: **50 GB**
7. Click **RENT**

→ Billing starts immediately.

---

## STEP 4: Connect via SSH (PowerShell)

Sau khi instance "Running" (~30s-2min), Console hiện SSH command:

```bash
ssh -p XXXXX root@x.y.z.w -i ~/.ssh/id_rsa
```

Trong PowerShell, modify path:

```powershell
ssh -p XXXXX root@x.y.z.w -i $HOME\.ssh\vast_key
```

Hoặc click button **Open** → mở Jupyter Lab tab trong browser.

---

## STEP 5: Setup environment (one-shot)

Sau khi SSH vào instance, copy paste lệnh sau:

```bash
# Download setup script
cd /workspace
wget https://raw.githubusercontent.com/duonguwu/sam3_research/main/kaggle/vast/setup_vast.sh
chmod +x setup_vast.sh

# Run
bash setup_vast.sh
```

**LƯU Ý**: Setup sẽ FAIL ở Kaggle API setup vì chưa có `kaggle.json`. Bước tiếp theo sẽ fix.

---

## STEP 6: Upload Kaggle API token

### 6.1 Lấy `kaggle.json` từ Kaggle account

1. Kaggle.com → **Settings** (avatar góc phải)
2. Scroll xuống **API** → **Create New API Token**
3. Download `kaggle.json`

### 6.2 Upload lên Vast (2 cách)

#### Cách A: Via Jupyter Lab (dễ nhất)

1. Click **Open** trên Vast Console → Jupyter Lab mở ra
2. Trong Jupyter Lab → navigate đến `/workspace/`
3. **Drag-drop** `kaggle.json` vào folder `/workspace/`

#### Cách B: Via SCP từ local PowerShell

```powershell
scp -P XXXXX -i $HOME\.ssh\vast_key `
    "$HOME\Downloads\kaggle.json" `
    root@x.y.z.w:/workspace/
```

### 6.3 Re-run setup

Trong SSH terminal Vast:

```bash
cd /workspace
bash setup_vast.sh
# Bây giờ Kaggle API sẽ setup OK
```

---

## STEP 7: Download data

```bash
cd /workspace/sam3_research/kaggle/vast
bash download_data.sh
```

Sẽ download:
- PanNuke dataset (~3GB)
- SAM3 native weights (~3.4GB)
- A2 LoRA weights (~10MB)
- A3 TypeHead weights (~50KB)

Total: ~7GB, ~10-15 phút.

---

## STEP 8: Verify environment

```bash
cd /workspace/sam3_research/kaggle/vast
python config_vast.py
```

Expected:
```
[OK]   Repo        : /workspace/sam3_research
[OK]   SAM3 dir    : /workspace/sam3_research/sam3
[OK]   PanNuke     : /workspace/sam3_research/data/pannuke
[OK]   SAM3 ckpt   : /workspace/sam3_research/checkpoints/sam3.pt
[OK]   LoRA ckpt   : /workspace/sam3_research/checkpoints/sam3_lora_rank16_final.pt
[OK]   TypeHead    : /workspace/sam3_research/checkpoints/type_head_final.pt
```

---

## STEP 9: Run pipeline (option A — master script)

```bash
# Setup tmux (giữ session khi disconnect SSH)
tmux new -s sam3

# Run all multi-seed pipeline
cd /workspace/sam3_research/kaggle/vast
bash run_all.sh

# Detach: Ctrl+B, then D
# Reattach: tmux attach -t sam3
```

→ Total ~25h on A100, ~$14 cost.

---

## STEP 9b: Run pipeline (option B — individual)

Nếu muốn chạy từng phase riêng để debug:

```bash
cd /workspace/sam3_research/kaggle/vast

# 1. A2 LoRA train × 3 seeds (~5h)
python run_a2_multiseed.py

# 2. A2 eval × 3 seeds (~1.5h)
python run_a2_eval_multiseed.py

# 3. A3 TypeHead train × 3 seeds (~6h)
python run_a3_multiseed.py

# 4. A3 eval × 3 seeds (~7h)
python run_a3_eval_multiseed.py

# 5. Phase B × 5 seeds (~1h)
python run_phaseB_multiseed.py

# 6. Phase C multi-seed (~5h, cached)
python run_phaseC_multiseed.py
```

---

## STEP 10: Monitor progress

### Trong SSH terminal (Vast):

```bash
# GPU usage
nvidia-smi -l 5    # update mỗi 5s

# Disk
df -h /workspace

# Logs
tail -f /workspace/sam3_research/kaggle/vast/logs/*.log
```

### Trong Jupyter Lab (browser):

Mở terminal trong Jupyter để chạy `nvidia-smi`, hoặc view file qua file browser.

---

## STEP 11: Backup results lên Kaggle

**TRƯỚC KHI DESTROY instance**, backup hết:

```bash
cd /workspace/sam3_research

# Create new Kaggle dataset (1 lần đầu)
kaggle datasets init -p checkpoints_multiseed/
# Edit checkpoints_multiseed/dataset-metadata.json:
#   "id": "<your-kaggle-username>/sam3-q1-multiseed-ckpts",
#   "title": "SAM3 Q1 Multi-seed Checkpoints"

kaggle datasets create -p checkpoints_multiseed/

# Hoặc version mới nếu đã tồn tại
kaggle datasets version -p checkpoints_multiseed/ -m "Q1 multi-seed v1"

# Same for results
kaggle datasets init -p work/
# Edit metadata...
kaggle datasets create -p work/

# Verify uploaded
kaggle datasets list --user $YOUR_USERNAME
```

---

## STEP 12: Download results về local (Windows)

### Option A: Via Kaggle (recommend)

```powershell
# Trên local Windows PowerShell
cd "d:/1LUANVAN/counting/sam3_research"
mkdir results -Force
cd results
kaggle datasets download -d YOUR_USERNAME/sam3-q1-multiseed-ckpts --unzip
kaggle datasets download -d YOUR_USERNAME/sam3-q1-results --unzip
```

### Option B: Via SCP

```powershell
scp -P XXXXX -i $HOME\.ssh\vast_key -r `
    root@x.y.z.w:/workspace/sam3_research/work/ `
    "d:/1LUANVAN/counting/sam3_research/results_vast/"

scp -P XXXXX -i $HOME\.ssh\vast_key -r `
    root@x.y.z.w:/workspace/sam3_research/checkpoints_multiseed/ `
    "d:/1LUANVAN/counting/sam3_research/checkpoints_multiseed/"
```

---

## STEP 13: DESTROY instance (STOP BILLING)

⚠️ **QUAN TRỌNG**: Verify đã download/backup HẾT trước khi destroy.

1. Vast.ai **Console** → **Instances**
2. Click icon **STOP** (⏹) → confirm
3. Click **DESTROY** (🗑) → confirm

→ Billing dừng ngay lập tức.

---

## Troubleshooting

### Q: `nvidia-smi` báo không có GPU
**A**: Chọn sai image. Destroy instance, rent lại với preset **PyTorch (cuda 12.1)**.

### Q: Setup fail "gcc not found"
**A**: Image thiếu build tools. Run `apt install -y build-essential` rồi re-run setup.

### Q: Kaggle download fail "401 Unauthorized"
**A**: `kaggle.json` chưa đúng. Re-upload, check `~/.kaggle/kaggle.json` permission = 600.

### Q: Disk full khi đang train
**A**: Cần rent instance disk lớn hơn. Pre-rent check `--disk 100GB+`.

### Q: SSH disconnect khi đang train
**A**: KHÔNG matter, tmux giữ process. Re-SSH + `tmux attach -t sam3`.

### Q: Phase C inference chậm hơn forecast
**A**: GPU không phải A100 (T4/3090 chậm hơn). Check `nvidia-smi`. Nếu vô tình rent T4 → time ×3.

### Q: Bị spot interrupted
**A**: Rent lại non-interruptible. Resume từ checkpoint.

---

## Cost breakdown reference

| Task | A100 hours | $0.55/h | Cumulative |
|------|-----------|---------|------------|
| Setup + download | 0.5h | $0.30 | $0.30 |
| A2 train × 3 | 5h | $2.75 | $3.05 |
| A2 eval × 3 | 1.5h | $0.85 | $3.90 |
| A3 train × 3 | 6h | $3.30 | $7.20 |
| A3 eval × 3 | 7h | $3.85 | $11.05 |
| Phase B × 5 | 1h | $0.55 | $11.60 |
| Phase C × multi | 5h | $2.75 | $14.35 |
| Buffer | 2h | $1.10 | **~$15.50** |

→ **Budget $20 đủ thoải mái**.

---

## Quick reference commands

| Action | Command |
|--------|---------|
| SSH | `ssh -p XXXXX root@IP -i ~/.ssh/vast_key` |
| Tmux new | `tmux new -s sam3` |
| Tmux detach | `Ctrl+B, D` |
| Tmux attach | `tmux attach -t sam3` |
| GPU check | `nvidia-smi -l 5` |
| Disk check | `df -h /workspace` |
| Backup all | `bash backup_results.sh` |
| Destroy | Vast Console → 🗑 |

---

## Action checklist

- [ ] Vast account + credit
- [ ] SSH key generated + uploaded
- [ ] Instance rented (A100, 50GB disk)
- [ ] SSH connection working
- [ ] `kaggle.json` uploaded
- [ ] `setup_vast.sh` ran OK
- [ ] `download_data.sh` ran OK
- [ ] `config_vast.py` verify ALL [OK]
- [ ] tmux session started
- [ ] `run_all.sh` started
- [ ] Periodic check progress
- [ ] Backup to Kaggle BEFORE destroy
- [ ] Verify download local
- [ ] DESTROY instance
