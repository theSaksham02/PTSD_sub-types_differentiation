"""
train.py — Colab-optimised training loop.

Features:
  - Mixed precision (torch.cuda.amp GradScaler + autocast)
  - Gradient checkpointing toggle on the fusion MLP (saves VRAM)
  - Early stopping (patience = 10) on validation loss
  - Class-weighted cross-entropy for both heads (handles imbalance)
  - Checkpoint saving to Google Drive (survives session timeout)
  - resume_from_checkpoint() for Colab runtime disconnects

Input is a feature DataFrame (see features.py). Rows are split into the four
model branches by column name.
"""
from __future__ import annotations

import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from config import (
    SEED, FEATURE_NAMES, N_AU, N_REGION, N_POSE, N_TEMPORAL,
    N_EMOTION, N_SUBTYPE, EMOTION_CLASSES, SUBTYPE_CLASSES,
    DRIVE_CHECKPOINT_DIR, LOCAL_CHECKPOINT_DIR,
)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class AffectFeatureDataset(Dataset):
    def __init__(self, df, emotion_col="emotion", subtype_col="subtype",
                 emo_to_idx=None, sub_to_idx=None):
        self.df = df.reset_index(drop=True)
        self.emo_to_idx = emo_to_idx or {c: i for i, c in enumerate(EMOTION_CLASSES)}
        self.sub_to_idx = sub_to_idx or {c: i for i, c in enumerate(SUBTYPE_CLASSES)}

    def __len__(self):
        return len(self.df)

    def _slice(self, row):
        au = row[[f"au_{a}" for a in _AU_COLS()]].to_numpy(dtype=np.float32)
        region = row[[f"region_{r}" for r in ["eye", "mouth", "upper_face", "lower_face"]]].to_numpy(dtype=np.float32)
        pose = row[["pose_yaw", "pose_pitch", "pose_roll"]].to_numpy(dtype=np.float32)
        temporal = row[[f"temp_{t}" for t in ["dwell_time", "transition_rate", "entropy", "au_variability"]]].to_numpy(dtype=np.float32)
        return au, region, pose, temporal

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        au, region, pose, temporal = self._slice(row)
        emo = self.emo_to_idx[row["emotion"]]
        sub = self.sub_to_idx[row["subtype"]]
        return (torch.from_numpy(au), torch.from_numpy(region),
                torch.from_numpy(pose), torch.from_numpy(temporal),
                torch.tensor(emo, dtype=torch.long),
                torch.tensor(sub, dtype=torch.long))


def _AU_COLS():
    # keep in sync with config.AU_NAMES without circular import at module load
    return ["AU01", "AU02", "AU04", "AU05", "AU06", "AU07", "AU09", "AU10",
            "AU12", "AU14", "AU15", "AU17", "AU20", "AU23", "AU25", "AU26", "AU45"]


# ---------------------------------------------------------------------------
# Class weights
# ---------------------------------------------------------------------------
def compute_class_weights(labels: list[int], n_classes: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=n_classes).astype(np.float32)
    counts = np.where(counts == 0, 1.0, counts)  # avoid div-by-zero
    weights = counts.sum() / (n_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------
class MultiTaskLoss(nn.Module):
    def __init__(self, emo_weight, sub_weight, lambda_subtype=0.5):
        super().__init__()
        self.emo = nn.CrossEntropyLoss(weight=emo_weight)
        self.sub = nn.CrossEntropyLoss(weight=sub_weight)
        self.lambda_subtype = lambda_subtype

    def forward(self, emo_logits, sub_logits, emo_y, sub_y):
        return self.emo(emo_logits, emo_y) + self.lambda_subtype * self.sub(sub_logits, sub_y)


# ---------------------------------------------------------------------------
# Checkpoint save / resume
# ---------------------------------------------------------------------------
def _checkpoint_dir(use_drive: bool) -> str:
    d = DRIVE_CHECKPOINT_DIR if use_drive else LOCAL_CHECKPOINT_DIR
    os.makedirs(d, exist_ok=True)
    return d


def save_checkpoint(path, model, optimizer, scaler, epoch, best_val_loss,
                    emo_weight, sub_weight, extra=None):
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict(),
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "emo_weight": emo_weight,
        "sub_weight": sub_weight,
        "extra": extra or {},
    }, path)
    print(f"[checkpoint] saved -> {path}")


def resume_from_checkpoint(path, model, optimizer, scaler):
    """Restore model/optimizer/scaler state and return training metadata."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"No checkpoint at {path}")
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    scaler.load_state_dict(ckpt["scaler_state"])
    print(f"[resume] restored epoch {ckpt['epoch']} "
          f"(best_val_loss={ckpt['best_val_loss']:.5f})")
    return ckpt


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, loss_fn, optimizer, scaler, device,
                    use_amp=True):
    model.train()
    total, emo_loss_sum, sub_loss_sum = 0.0, 0.0, 0.0
    for batch in loader:
        au, region, pose, temporal, emo_y, sub_y = [b.to(device) for b in batch]
        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            with torch.cuda.amp.autocast():
                emo_logits, sub_logits = model(au, region, pose, temporal)
                loss = loss_fn(emo_logits, sub_logits, emo_y, sub_y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            emo_logits, sub_logits = model(au, region, pose, temporal)
            loss = loss_fn(emo_logits, sub_logits, emo_y, sub_y)
            loss.backward()
            optimizer.step()
        total += loss.item() * au.size(0)
        emo_loss_sum += loss_fn.emo(emo_logits.detach(), emo_y).item() * au.size(0)
        sub_loss_sum += loss_fn.sub(sub_logits.detach(), sub_y).item() * au.size(0)
    n = len(loader.dataset)
    return total / n, emo_loss_sum / n, sub_loss_sum / n


@torch.no_grad()
def evaluate_loader(model, loader, loss_fn, device):
    model.eval()
    total, correct_emo, correct_sub = 0.0, 0, 0
    for batch in loader:
        au, region, pose, temporal, emo_y, sub_y = [b.to(device) for b in batch]
        emo_logits, sub_logits = model(au, region, pose, temporal)
        loss = loss_fn(emo_logits, sub_logits, emo_y, sub_y)
        total += loss.item() * au.size(0)
        correct_emo += (emo_logits.argmax(1) == emo_y).sum().item()
        correct_sub += (sub_logits.argmax(1) == sub_y).sum().item()
    n = len(loader.dataset)
    return total / n, correct_emo / n, correct_sub / n


def train(
    model, train_df, val_df, device,
    epochs=50, batch_size=256, lr=1e-3, weight_decay=1e-4,
    lambda_subtype=0.5, patience=10, use_amp=True,
    use_drive=True, checkpoint_name="best.pt",
    seed=SEED,
):
    """Full training loop with early stopping + Drive checkpointing.

    Returns (model, history dict). The best checkpoint (by val loss) is written
    to Drive (or local) as `checkpoint_name`.
    """
    from config import seed_everything
    seed_everything(seed)

    train_ds = AffectFeatureDataset(train_df)
    val_ds = AffectFeatureDataset(val_df)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)

    emo_weight = compute_class_weights(
        [train_ds.emo_to_idx[l] for l in train_df["emotion"]], N_EMOTION).to(device)
    sub_weight = compute_class_weights(
        [train_ds.sub_to_idx[l] for l in train_df["subtype"]], N_SUBTYPE).to(device)

    loss_fn = MultiTaskLoss(emo_weight, sub_weight, lambda_subtype)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    ckpt_dir = _checkpoint_dir(use_drive)
    ckpt_path = os.path.join(ckpt_dir, checkpoint_name)

    best_val_loss = float("inf")
    best_epoch = -1
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_emo_acc": [], "val_sub_acc": []}

    for epoch in range(epochs):
        t0 = time.time()
        tr_loss, tr_emo, tr_sub = train_one_epoch(
            model, train_loader, loss_fn, optimizer, scaler, device, use_amp)
        val_loss, val_emo, val_sub = evaluate_loader(model, val_loader, loss_fn, device)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["val_emo_acc"].append(val_emo)
        history["val_sub_acc"].append(val_sub)

        print(f"epoch {epoch+1:03d}/{epochs} | "
              f"tr_loss {tr_loss:.4f} | val_loss {val_loss:.4f} | "
              f"val_emo_acc {val_emo:.4f} | val_sub_acc {val_sub:.4f} | "
              f"{time.time()-t0:.1f}s")

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            save_checkpoint(ckpt_path, model, optimizer, scaler, epoch,
                            best_val_loss, emo_weight, sub_weight,
                            extra={"history": history})
        else:
            patience_counter += 1
            print(f"  (no improvement, patience {patience_counter}/{patience})")
            if patience_counter >= patience:
                print(f"[early-stop] stopping at epoch {epoch+1} "
                      f"(best epoch {best_epoch+1}, val_loss {best_val_loss:.4f})")
                break

    return model, history, ckpt_path
