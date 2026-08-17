"""
model_smoke_test.py — CPU verification of the PyTorch model, training mechanics
and Grad-CAM. (SHAP is verified by syntax compile only; it needs the shap
package which is installed in Colab.)
"""
import os, sys, json
SRC = os.path.join(os.path.dirname(__file__), "src")
sys.path.insert(0, SRC)

import torch
import numpy as np

import config
from config import (N_AU, N_REGION, N_POSE, N_TEMPORAL, N_EMOTION, N_SUBTYPE,
                    EMOTION_CLASSES, SUBTYPE_CLASSES)

results = {"checks": [], "failures": 0}
def check(name, cond, detail=""):
    results["checks"].append({"name": name, "status": "PASS" if cond else "FAIL",
                              "detail": detail})
    if not cond:
        results["failures"] += 1
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")

import features as F
import model as M
import train as T

config.seed_everything(0)

# ---- Model construction + forward -----------------------------------------
model = M.build_model()
s = model.summary()
model.eval()  # disable dropout/BatchNorm training stats for the shape checks below
print("\n" + s)
check("summary contains heads", "Emotion head" in s and "Subtype head" in s, "")

b = 8
au_b, rg_b, ps_b, tm_b = (torch.randn(b, N_AU), torch.randn(b, N_REGION),
                          torch.randn(b, N_POSE), torch.randn(b, N_TEMPORAL))
emo, sub = model(au_b, rg_b, ps_b, tm_b)
check("emotion logits shape", tuple(emo.shape) == (b, N_EMOTION), str(tuple(emo.shape)))
check("subtype logits shape", tuple(sub.shape) == (b, N_SUBTYPE), str(tuple(sub.shape)))

# forward_flat (SHAP path) splits 28 -> 4 branches correctly on the SAME data
X = torch.cat([au_b, rg_b, ps_b, tm_b], dim=1)
ef, sf = model.forward_flat(X)
check("forward_flat matches forward", torch.allclose(emo, ef, atol=1e-4) and
      torch.allclose(sub, sf, atol=1e-4), "")

# ---- Gradient checkpointing toggle -----------------------------------------
model_ckpt = M.build_model(use_gradient_checkpointing=True)
model_ckpt.train()
Xk = torch.randn(4, config.N_FEATURES, requires_grad=True)
e, s_ = model_ckpt.forward_flat(Xk)
(e.sum() + s_.sum()).backward()
check("gradient-checkpointed forward+backward runs", Xk.grad is not None and
      torch.isfinite(Xk.grad).all(), "")

# ---- Training mechanics (1 epoch, CPU, no AMP) -----------------------------
df, Xf, y_emo, y_sub = F.generate_synthetic_dataset(n_sequences=6, frames_per_seq=30, seed=1)
tr_df = df.iloc[:120].copy()
va_df = df.iloc[120:].copy()

train_ds = T.AffectFeatureDataset(tr_df)
val_ds = T.AffectFeatureDataset(va_df)
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=32, shuffle=False)

emo_w = T.compute_class_weights([train_ds.emo_to_idx[l] for l in tr_df["emotion"]], N_EMOTION)
sub_w = T.compute_class_weights([train_ds.sub_to_idx[l] for l in tr_df["subtype"]], N_SUBTYPE)
loss_fn = T.MultiTaskLoss(emo_w, sub_w)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
scaler = torch.cuda.amp.GradScaler(enabled=False)

tr_loss, tr_emo, tr_sub = T.train_one_epoch(model, train_loader, loss_fn, opt,
                                            scaler, device="cpu", use_amp=False)
check("train_one_epoch returns finite loss", np.isfinite(tr_loss), f"loss={tr_loss:.4f}")
val_loss, val_emo, val_sub = T.evaluate_loader(model, val_loader, loss_fn, "cpu")
check("evaluate_loader returns finite loss + acc",
      np.isfinite(val_loss) and 0 <= val_emo <= 1 and 0 <= val_sub <= 1,
      f"val_loss={val_loss:.4f} emo_acc={val_emo:.3f} sub_acc={val_sub:.3f}")

# ---- Checkpoint save + resume ----------------------------------------------
ckpt_path = os.path.join(os.path.dirname(__file__), "smoke_output", "test_ckpt.pt")
T.save_checkpoint(ckpt_path, model, opt, scaler, epoch=0, best_val_loss=val_loss,
                  emo_weight=emo_w, sub_weight=sub_w, extra={"history": {"train_loss": [tr_loss]}})
check("checkpoint file written", os.path.exists(ckpt_path), "")
model2 = M.build_model()
opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
scaler2 = torch.cuda.amp.GradScaler(enabled=False)
meta = T.resume_from_checkpoint(ckpt_path, model2, opt2, scaler2)
check("resume restores state", meta["epoch"] == 0 and meta["best_val_loss"] == val_loss, "")

# ---- Grad-CAM --------------------------------------------------------------
import explain as EX
backbone = M.GradCAMBackbone()
img_t = torch.rand(3, 96, 96)  # (C, H, W) float tensor for grad_cam
cam = EX.grad_cam(backbone, img_t, target_class=0, device="cpu")
check("grad_cam heatmap finite & normalised",
      np.isfinite(cam).all() and 0.0 <= cam.min() and cam.max() <= 1.0,
      f"shape={cam.shape} range=[{cam.min():.2f},{cam.max():.2f}]")

# save_gradcam_grid expects (H, W, 3) uint8 images
img_hwc = np.random.randint(0, 255, (96, 96, 3), dtype=np.uint8)
out = os.path.join(os.path.dirname(__file__), "smoke_output")
os.makedirs(out, exist_ok=True)
grid = EX.save_gradcam_grid(backbone, [img_hwc, img_hwc], EMOTION_CLASSES,
                            os.path.join(out, "gradcam_grid.png"), device="cpu")
check("gradcam grid saved", os.path.exists(grid), "")

with open(os.path.join(out, "model_smoke_summary.json"), "w") as f:
    json.dump(results, f, indent=2, default=str)

print(f"\nModel smoke test complete. {len(results['checks'])} checks, "
      f"{results['failures']} failures.")
sys.exit(1 if results["failures"] else 0)
