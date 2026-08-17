"""
build_notebook.py — Assemble the integrated Colab notebook from src/*.py.

Reads the modular source files and produces a single self-contained
`notebook/PTSD_Affect_Recognition.ipynb` with sections:
README -> Requirements -> Smoke test -> Data -> Features -> Model ->
Train -> Evaluate -> Explain.

The notebook writes the module source to /content/pipeline/ and imports it,
so the notebook mirrors src/*.py exactly (single source of truth) and the
`from config import ...` cross-imports work unchanged.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "src")
NB_DIR = os.path.join(HERE, "notebook")
os.makedirs(NB_DIR, exist_ok=True)

MODULE_ORDER = ["config.py", "features.py", "tables.py", "evaluate.py",
                "model.py", "train.py", "explain.py", "data_utils.py"]


def read_module(name):
    with open(os.path.join(SRC, name)) as f:
        return f.read()


# ---------------------------------------------------------------------------
# Cell builders
# ---------------------------------------------------------------------------
def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source.splitlines(keepends=True)}


def build_setup_cell():
    """Cell that writes all modules to /content/pipeline/ and imports them."""
    lines = [
        "# Write the modular pipeline source to disk and import it.\n",
        "# (This mirrors the `src/*.py` files exactly, so the notebook is a\n",
        "#  single self-contained file but keeps clean module boundaries.)\n",
        "import os, sys, json\n",
        "PIPE = '/content/pipeline'\n",
        "os.makedirs(PIPE, exist_ok=True)\n",
        "if PIPE not in sys.path:\n",
        "    sys.path.insert(0, PIPE)\n",
        "\n",
        "SRC = {}\n",
    ]
    for name in MODULE_ORDER:
        lines.append(f"SRC[{name!r}] = {read_module(name)!r}\n")
    lines += [
        "\n",
        "for _name, _text in SRC.items():\n",
        "    with open(os.path.join(PIPE, _name), 'w') as _f:\n",
        "        _f.write(_text)\n",
        "print('wrote', len(SRC), 'modules to', PIPE)\n",
        "\n",
        "import config\n",
        "from config import seed_everything, SEED, FEATURE_NAMES, EMOTION_CLASSES, SUBTYPE_CLASSES\n",
        "from features import build_feature_frame, generate_synthetic_dataset\n",
        "from tables import save_tables, emotion_table_markdown, subtype_table_markdown\n",
        "from evaluate import run_evaluation, per_class_metrics, attach_auc\n",
        "from model import build_model, MultiInputPTSDAffectModel, GradCAMBackbone\n",
        "from train import train, resume_from_checkpoint, save_checkpoint, AffectFeatureDataset\n",
        "from explain import shap_feature_importance, grad_cam, save_gradcam_grid\n",
        "import data_utils as du\n",
        "\n",
        "seed_everything(SEED)\n",
        "print('pipeline modules loaded; seed =', SEED)\n",
    ]
    return code("".join(lines))


# ---------------------------------------------------------------------------
# README cell
# ---------------------------------------------------------------------------
README = """# Automated Facial Affect Recognition for PTSD Subtype Classification

**Experimental research pipeline** — *not* a diagnostic tool.

## Study context

This notebook implements the feature-extraction → modelling → evaluation →
explainability pipeline for a psychology paper that aims to distinguish:

- **Classic PTSD**
- **Dissociative PTSD (D-PTSD)**
- **Healthy Controls**

…using four complementary facial signals: **action units (AUs)**, **region-weighted
facial scores**, **head-pose dynamics**, and **temporal behavioural features**,
fused in a late-fusion multi-input classifier.

## ⚠️ Clinical & ethical notice

- This is an **experimental research model**, **not** a clinical diagnostic or
  treatment tool.
- **No public affect dataset contains PTSD diagnoses.** The six emotion/AU states
  come from public datasets; the `Classic PTSD / D-PTSD / Control` labels must be
  **study-provided** or applied as an **explicitly documented proxy mapping**
  (see the Manifest).
- Only **public or request-authorized** data is used; **no private patient
  identifiers** are processed or stored.

## How to run

1. **Runtime → Change runtime type → T4 GPU** (free) or **A100** (Pro).
2. Run **Requirements**, then **Smoke test** (tiny synthetic end-to-end run).
3. Proceed **Data → Features → Model → Train → Evaluate → Explain**.
4. Checkpoints are saved to Google Drive (`MyDrive/ptsd_affect/checkpoints/`) and
   training **auto-resumes** after a session timeout.

## Sections

| Section | Purpose |
|---|---|
| Data | Acquire CK+ / AffectNet / RAF-DB (and request DISFA / BP4D) |
| Features | 28-dim frame features: AU + region + pose + temporal |
| Model | Multi-input late-fusion MLP with 6 + 3 output heads |
| Train | AMP + gradient checkpointing + early stopping + Drive checkpoints |
| Evaluate | Metrics, confusion matrix, ROC, subtype report, paper tables |
| Explain | SHAP feature importance + Grad-CAM region overlays |
"""


# ---------------------------------------------------------------------------
# Requirements cell
# ---------------------------------------------------------------------------
REQUIREMENTS = """# ============================================================
# Requirements  (run once)  +  fixed random seed
# ============================================================
import sys, os
print('Python', sys.version)

# Base numerical / ML stack
!pip install -q numpy pandas scikit-learn matplotlib seaborn pillow

# Deep learning (T4 / A100)
!pip install -q torch torchvision

# Facial feature extraction (choose one; py-feat is easiest on Colab)
!pip install -q py-feat opencv-python-headless

# Explainability
!pip install -q shap

# Dataset acquisition
!pip install -q gdown kagglehub kaggle

# Mount Google Drive for checkpoint persistence
from google.colab import drive
drive.mount('/content/drive')
os.makedirs('/content/drive/MyDrive/ptsd_affect/checkpoints', exist_ok=True)
print('Drive mounted.')
"""


# ---------------------------------------------------------------------------
# Smoke test cell
# ---------------------------------------------------------------------------
SMOKE = """# ============================================================
# Smoke test — tiny end-to-end run on SYNTHETIC data (run first!)
# Verifies the whole pipeline wiring before any real download.
# ============================================================
import torch
import numpy as np

df, X, y_emo, y_sub = generate_synthetic_dataset(n_sequences=6, frames_per_seq=60, seed=0)
print('Feature frame:', df.shape, '| feature dim:', X.shape[1], '(expect 28)')
assert X.shape[1] == len(FEATURE_NAMES) == 28, 'feature dimension mismatch'
assert df['temp_entropy'].notna().all(), 'temporal features contain NaN'

# Build + forward
model = build_model()
model.eval()
au = torch.randn(8, 17); rg = torch.randn(8, 4); ps = torch.randn(8, 3); tm = torch.randn(8, 4)
emo_logits, sub_logits = model(au, rg, ps, tm)
print('emotion logits:', tuple(emo_logits.shape), '| subtype logits:', tuple(sub_logits.shape))
assert emo_logits.shape[1] == 6 and sub_logits.shape[1] == 3
print(model.summary())
print('\\nSMOKE TEST PASSED ✅')
"""


# ---------------------------------------------------------------------------
# Data section
# ---------------------------------------------------------------------------
DATA_MD = """## 1. Data

Acquire the **top-3 freely downloadable** datasets — CK+, AffectNet, RAF-DB —
and document the request workflow for the request-gated gold standards
(DISFA, BP4D). Full ranking + rationale: see `DATASET_REPORT.md` / `MANIFEST.md`.
"""

DATA_CODE = """# ============================================================
# 1. Data acquisition  (Kaggle / gdown / direct URL)
# ============================================================
# --- Strategy A: Kaggle datasets (no manual key with kagglehub) ---
# CK+ (Extended Cohn-Kanade) — AU + emotion labels, small, ideal Colab starter
ck_path = du.download_kaggle('sharmaroshan/extended-cohn-kanade', '/content/data/ck+')

# AffectNet (8-emotion subset)
affect_path = du.download_kaggle('noamsegal/affectnet-training-data', '/content/data/affectnet')

# RAF-DB (single-label subset)
raf_path = du.download_kaggle('shuvoalok/raf-db-dataset', '/content/data/rafdb')

# --- Strategy B: direct URL / gdown (fallback) ---
# ck_zip = du.download_gdown('<FILE_ID>', '/content/data/ck+.zip')
# du.unzip_all(ck_zip, '/content/data/ck+_unzipped')

# --- Structure into train/val/test (class folders) ---
# du.split_image_folders('/content/data/ck+/CK+48', '/content/data/ck+_split',
#                        val_ratio=0.15, test_ratio=0.15, seed=SEED)

# --- Integrity check: show sample faces + labels ---
# du.display_sample_grid(image_paths, labels, au_labels=None,
#                        title='CK+ sample', n_rows=2, n_cols=4)

# --- Request-gated gold standards (manual, see DATASET_REPORT.md) ---
du.print_request_instructions('DISFA')
du.print_request_instructions('BP4D')
print('\\nData section ready. Set real paths once downloaded.')
"""


# ---------------------------------------------------------------------------
# Features section
# ---------------------------------------------------------------------------
FEATURES_MD = """## 2. Features

Extract, per face frame: **17 AU intensities**, **4 region-weighted scores**
(eye / mouth / upper / lower face), **3 head-pose angles** (yaw/pitch/roll), and
**4 temporal features** (dwell time, transition rate, entropy, AU variability)
→ a **28-dim vector** per frame.
"""

FEATURES_CODE = """# ============================================================
# 2. Feature extraction  (AU + region + pose + temporal)
# ============================================================
# Real pipeline: get AU + pose from py-feat (or OpenFace) per frame.
#   from feat import Detector
#   detector = Detector(au_model='rf', emotion_model='resmasknet')
#   preds = detector.detect_video('/path/to/video.mp4')
#   au = preds.aus()      # (n_frames, 20) -> subset to the 17 canonical AUs
#   pose = preds.pose()   # (n_frames, 6) -> yaw/pitch/roll
#
# Here we demonstrate the full frame-DataFrame builder on synthetic data.

df, X, y_emo, y_sub = generate_synthetic_dataset(n_sequences=6, frames_per_seq=60, seed=0)

# The 28 feature columns in order
print('Feature columns (28):')
for i, c in enumerate(FEATURE_NAMES):
    print(f'  {i:2d} {c}')

# Inspect a few rows
print(df[['sequence_id', 'frame_idx', 'au_AU04', 'au_AU12', 'region_mouth',
          'region_upper_face', 'pose_yaw', 'temp_dwell_time',
          'temp_transition_rate', 'temp_entropy', 'temp_au_variability',
          'emotion', 'subtype']].head(8).to_string(index=False))

# Save frame-level features
os.makedirs('/content/data/features', exist_ok=True)
df.to_csv('/content/data/features/frame_features.csv', index=False)
print('\\nSaved /content/data/features/frame_features.csv')

# Visualise the feature structure (one sample sequence)
import matplotlib.pyplot as plt
seq = df[df.sequence_id == 0]
fig, axs = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
axs[0].plot(seq.frame_idx, seq[['au_AU04', 'au_AU06', 'au_AU12']])
axs[0].set_ylabel('AU intensity'); axs[0].legend(['AU04 brow', 'AU06 cheek', 'AU12 lip'], fontsize=8, frameon=False)
axs[1].plot(seq.frame_idx, seq[['region_eye', 'region_mouth', 'region_upper_face', 'region_lower_face']])
axs[1].set_ylabel('Region score'); axs[1].legend(fontsize=8, frameon=False)
axs[2].plot(seq.frame_idx, seq[['temp_dwell_time', 'temp_transition_rate', 'temp_entropy', 'temp_au_variability']])
axs[2].set_ylabel('Temporal'); axs[2].set_xlabel('frame'); axs[2].legend(fontsize=8, frameon=False)
plt.tight_layout(); plt.show()
"""


# ---------------------------------------------------------------------------
# Model section
# ---------------------------------------------------------------------------
MODEL_MD = """## 3. Model

**MultiInputPTSDAffectModel** — four branch encoders (Linear → BatchNorm → ReLU
→ Dropout) → late-fusion concatenation → shared MLP → two heads:

1. **Emotion head (6):** Fear, Anger, Sadness, Neutral, Surprise, Flat/Blunted Affect
2. **Subtype head (3):** Classic PTSD, D-PTSD, Control
"""

MODEL_CODE = """# ============================================================
# 3. Model — multi-input late-fusion classifier
# ============================================================
model = build_model(use_gradient_checkpointing=True)  # toggle to save VRAM
print(model.summary())

# Sample forward pass (shapes only)
import torch
b = 8
emo_logits, sub_logits = model(
    torch.randn(b, 17), torch.randn(b, 4), torch.randn(b, 3), torch.randn(b, 4))
print('emotion logits:', tuple(emo_logits.shape), '| subtype logits:', tuple(sub_logits.shape))

# Auxiliary CNN used only for Grad-CAM visualisation
backbone = GradCAMBackbone()
print('GradCAM backbone params:',
      sum(p.numel() for p in backbone.parameters()))
"""


# ---------------------------------------------------------------------------
# Train section
# ---------------------------------------------------------------------------
TRAIN_MD = """## 4. Train

Colab-optimised loop: **mixed precision (AMP)**, **gradient checkpointing**,
**class-weighted loss** (both heads), **early stopping (patience 10)**, and
**Google Drive checkpointing** with resume-from-checkpoint for session timeouts.
"""

TRAIN_CODE = """# ============================================================
# 4. Training  (AMP + early stopping + Drive checkpoint / resume)
# ============================================================
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device, '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')

# Load frame features (real run) — or use synthetic for the demo
# import pandas as pd
# df = pd.read_csv('/content/data/features/frame_features.csv')
df, X, y_emo, y_sub = generate_synthetic_dataset(n_sequences=12, frames_per_seq=80, seed=SEED)

# Simple sequential split (use subject-level split for the real study to avoid leakage)
split = int(0.8 * df.sequence_id.nunique())
tr_mask = df.sequence_id < split
train_df, val_df = df[tr_mask].copy(), df[~tr_mask].copy()
print(f'train frames {len(train_df)} | val frames {len(val_df)}')

model = build_model(use_gradient_checkpointing=True).to(device)

# Train (saves best checkpoint to Drive; resume happens automatically if it exists)
model, history, ckpt_path = train(
    model, train_df, val_df, device,
    epochs=3,               # raise for the real run (e.g. 50)
    batch_size=256, lr=1e-3, weight_decay=1e-4,
    lambda_subtype=0.5, patience=10,
    use_amp=(device.type == 'cuda'),
    use_drive=True, checkpoint_name='best.pt',
    seed=SEED,
)

# --- Resume after a Colab session timeout (demonstration) ---
# import torch
# model2 = build_model().to(device)
# opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
# scaler2 = torch.cuda.amp.GradScaler(enabled=True)
# meta = resume_from_checkpoint('/content/drive/MyDrive/ptsd_affect/checkpoints/best.pt',
#                               model2, opt2, scaler2)
# print('resumed from epoch', meta['epoch'])
"""


# ---------------------------------------------------------------------------
# Evaluate section
# ---------------------------------------------------------------------------
EVALUATE_MD = """## 5. Evaluate

Per-class precision / recall / F1 for six emotions, a confusion matrix with
**fear–surprise** and **anger–disgust** pairs highlighted, **ROC curves**, the
**PTSD-subtype report** (Classic / D-PTSD / Control), and paper-ready summary
tables (Markdown + LaTeX).
"""

EVALUATE_CODE = """# ============================================================
# 5. Evaluation + figures + paper tables
# ============================================================
import numpy as np

# --- Run the model on the val split to get predictions / probabilities ---
# (demo uses synthetic labels; replace with real model outputs)
from evaluate import EMOTION_CLASSES as _EC
rng = np.random.default_rng(0)
y_emo_true = list(val_df['emotion'])
y_emo_pred = [e if rng.random() < 0.7 else _EC[rng.integers(6)] for e in y_emo_true]
y_emo_score = np.zeros((len(y_emo_true), 6))
for i, e in enumerate(y_emo_true):
    j = _EC.index(e); y_emo_score[i, j] = rng.uniform(0.5, 1.0)
y_emo_score = (y_emo_score + 0.05) / (y_emo_score + 0.05).sum(1, keepdims=True)

y_sub_true = list(val_df['subtype'])
y_sub_pred = [s if rng.random() < 0.6 else SUBTYPE_CLASSES[rng.integers(3)] for s in y_sub_true]

os.makedirs('/content/figures', exist_ok=True)
bundle = run_evaluation(y_emo_true, y_emo_pred, y_emo_score, y_sub_true, y_sub_pred,
                        '/content/figures')

# --- Paper-ready tables ---
emo_rows = [{'class': c, **{k: bundle['emotion_metrics'][c][k]
                            for k in ('precision', 'recall', 'f1', 'auc')}}
            for c in EMOTION_CLASSES]
tpaths = save_tables(emo_rows, bundle['subtype_metrics'], '/content/figures',
                     classes=EMOTION_CLASSES)

print('\\n===== Emotion metrics (Markdown) =====')
print(tpaths['emotion_markdown'])
print('\\n===== Subtype metrics (Markdown) =====')
print(tpaths['subtype_markdown'])
print('\\n===== Subtype report (sklearn) =====')
print(bundle['subtype_metrics']['report'])
print('\\nFigures:', bundle['figures'])
print('Tables :', {k: v for k, v in tpaths.items() if k.endswith(('_md', '_tex'))})
"""


# ---------------------------------------------------------------------------
# Explain section
# ---------------------------------------------------------------------------
EXPLAIN_MD = """## 6. Explain

- **SHAP** — which AU / region / pose / temporal features drive each subtype
  prediction (Fathom-style bar charts for the Results section).
- **Grad-CAM** — which facial regions the auxiliary image pathway attends to,
  per emotion class.
"""

EXPLAIN_CODE = """# ============================================================
# 6. Explainability  (SHAP + Grad-CAM)
# ============================================================
import numpy as np, torch

# --- SHAP: feature importance per PTSD subtype ---
# Use a small background/sample set (T4-friendly). Replace X with real features.
_, X_bg, _, _ = generate_synthetic_dataset(n_sequences=4, frames_per_seq=40, seed=7)
shap_paths = shap_feature_importance(
    model.to('cpu'), X_bg, X_bg,
    feature_names=FEATURE_NAMES, output_head='subtype',
    class_names=SUBTYPE_CLASSES, out_dir='/content/figures', n_background=80)
print('SHAP charts:', shap_paths)

# --- Grad-CAM: attended facial regions per emotion ---
# Replace `demo_faces` with real face crops (e.g. from AffectNet / CK+).
demo_faces = [np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
              for _ in range(6)]
grid_path = save_gradcam_grid(GradCAMBackbone(), demo_faces, EMOTION_CLASSES,
                              '/content/figures/gradcam_grid.png', device='cpu')
print('Grad-CAM grid:', grid_path)
print('\\nExplain section complete.')
"""


# ---------------------------------------------------------------------------
# Assemble notebook
# ---------------------------------------------------------------------------
cells = [
    md(README),
    code(REQUIREMENTS),
    build_setup_cell(),
    code(SMOKE),
    md(DATA_MD), code(DATA_CODE),
    md(FEATURES_MD), code(FEATURES_CODE),
    md(MODEL_MD), code(MODEL_CODE),
    md(TRAIN_MD), code(TRAIN_CODE),
    md(EVALUATE_MD), code(EVALUATE_CODE),
    md(EXPLAIN_MD), code(EXPLAIN_CODE),
]

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
        "colab": {"provenance": [], "name": "PTSD_Affect_Recognition.ipynb"},
    },
    "cells": cells,
}

out_path = os.path.join(NB_DIR, "PTSD_Affect_Recognition.ipynb")
with open(out_path, "w") as f:
    json.dump(nb, f, indent=1)

# Validate: parse back
with open(out_path) as f:
    parsed = json.load(f)
assert parsed["nbformat"] == 4
assert len(parsed["cells"]) == len(cells)
code_cells = [c for c in parsed["cells"] if c["cell_type"] == "code"]
print(f"Built {out_path}")
print(f"  cells: {len(parsed['cells'])} total, {len(code_cells)} code")
print(f"  bytes: {os.path.getsize(out_path):,}")
