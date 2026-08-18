# 02 · GPU Training Guide

How to actually train the model on Google Colab — from free-tier T4 up to a
student A100 — without running out of memory or losing work to session timeouts.

---

## 1. Pick the right runtime

| | T4 (free) | A100 (Colab Pro / student credits) |
|---|---|---|
| VRAM | 16 GB | 40 GB |
| Best for | Smoke test, small CK+ runs, debugging | AffectNet-scale training, larger batch |
| Cost | Free | Credits |
| Batch size (this model) | 128–512 | 512–2048 |

The model is tiny (27k params) because features are extracted **ahead** of
training — so even a T4 is far more than enough. The bottleneck is your
**feature table + dataloader**, not the GPU.

**Set it:** `Runtime → Change runtime type → Hardware accelerator → GPU`.

---

## 2. What the training loop already does for you

The loop in `src/train.py` is already GPU-hardened:

- **Mixed precision** (`torch.cuda.amp.GradScaler` + `autocast`) — ~2× faster, ~half VRAM.
- **Gradient checkpointing** (`use_gradient_checkpointing=True`) — trades compute for VRAM.
- **Class-weighted loss** on both heads — handles emotion-class imbalance.
- **Early stopping** (patience 10) — stops before overfitting.
- **Google Drive checkpoints** — survives session timeout.
- **Resume** — restart and pick up where you left off.

You don't need to write any of this — just call `train(...)`.

---

## 3. Step-by-step run

```python
# In the notebook, after Data + Features have produced a feature CSV:
import pandas as pd, torch
from config import seed_everything
from model import build_model
from train import train, resume_from_checkpoint

seed_everything(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('GPU:', torch.cuda.get_device_name(0))

df = pd.read_csv('/content/data/features/frame_features.csv')

# SUBJECT-LEVEL split (not random!) to avoid leakage
subjects = sorted(df.sequence_id.unique())
n_train = int(0.8 * len(subjects))
train_ids, val_ids = set(subjects[:n_train]), set(subjects[n_train:])
train_df = df[df.sequence_id.isin(train_ids)].copy()
val_df   = df[df.sequence_id.isin(val_ids)].copy()

model = build_model(use_gradient_checkpointing=True).to(device)

model, history, ckpt = train(
    model, train_df, val_df, device,
    epochs=50, batch_size=512, lr=1e-3, weight_decay=1e-4,
    lambda_subtype=0.5, patience=10,
    use_amp=True, use_drive=True, checkpoint_name='best.pt', seed=42,
)
```

---

## 4. If you run out of memory (OOM)

In order of what to try:

1. **Lower `batch_size`** — 512 → 256 → 128.
2. **Ensure `use_amp=True`** (halves activations).
3. **Ensure `use_gradient_checkpointing=True`** (already set above).
4. **Reduce `num_workers`** to 0 if the dataloader is the problem.
5. **Restart runtime** — Colab often gives you a fresh 16 GB.

A `RuntimeError: CUDA out of memory` is **not** a bug in this code; it's a batch-size problem. Halving batch size roughly halves VRAM.

---

## 5. Surviving session timeouts

Colab free tier disconnects after ~90 min idle / ~12 h active. Two layers of protection:

- **Drive checkpoint** — `train()` writes `best.pt` to `MyDrive/ptsd_affect/checkpoints/` after every improvement.
- **Resume** — after reconnect, run:

```python
import torch
model = build_model().to(device)
opt  = torch.optim.AdamW(model.parameters(), lr=1e-3)
scaler = torch.cuda.amp.GradScaler(enabled=True)
meta = resume_from_checkpoint(
    '/content/drive/MyDrive/ptsd_affect/checkpoints/best.pt', model, opt, scaler)
print('resumed from epoch', meta['epoch'])
```

**Tip:** keep the Colab tab focused, and run a keep-alive by ticking
`Runtime → Keep alive` isn't available on free tier — instead re-run a cheap
cell periodically, or use Colab Pro's longer limits.

---

## 6. Monitoring training

The loop prints per-epoch:

```
epoch 003/050 | tr_loss 1.9821 | val_loss 2.0143 | val_emo_acc 0.4120 | val_sub_acc 0.4800 | 12.3s
```

Watch three things:

| Signal | Healthy | Warning |
|---|---|---|
| `val_loss` | decreasing | flat / rising → overfitting |
| `val_emo_acc` | > chance (≈16.7%) | stuck at chance → features/labels wrong |
| `val_sub_acc` | > chance (≈33.3%) | stuck at chance → labels are noise |

Both heads train simultaneously — the printed accuracies let you confirm each
head is actually learning.

---

## 7. A100 (student credits) — what changes

Nothing in the code changes. Only:
- Set runtime to **A100**.
- Raise `batch_size` to 1024–2048.
- Optionally **disable** `use_gradient_checkpointing` (you have VRAM to spare)
  to trade memory for speed.

The same checkpoint format works across T4 ↔ A100 — you can start on T4 and
finish on A100.

---

## 8. Quick reference

| Knob | Where | Default |
|---|---|---|
| Batch size | `train(..., batch_size=)` | 256 |
| Learning rate | `train(..., lr=)` | 1e-3 |
| Epochs | `train(..., epochs=)` | 50 |
| Early-stop patience | `train(..., patience=)` | 10 |
| Subtype loss weight | `train(..., lambda_subtype=)` | 0.5 |
| Gradient checkpointing | `build_model(use_gradient_checkpointing=)` | True |
| Mixed precision | `train(..., use_amp=)` | True (auto) |

→ Back to [01 · What is remaining](01_WHAT_IS_REMAINING.md)
