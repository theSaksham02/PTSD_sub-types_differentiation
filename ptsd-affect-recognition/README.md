# Automated Facial Affect Recognition for PTSD Subtype Classification

**Experimental research pipeline** (not a clinical/diagnostic tool) for a psychology
research paper. It classifies facial-affect states from action units (AUs),
region-weighted facial scores, head-pose dynamics and temporal behavioural
features, then maps those patterns to three groups:

- **Classic PTSD**
- **Dissociative PTSD (D-PTSD)**
- **Healthy Controls**

## ⚠️ Clinical & ethical notice (must read)

- This is an **experimental research model**, **not** a diagnostic or treatment tool.
- **No public affect dataset contains PTSD diagnoses.** The six emotion/AU states
  come from public datasets; the `Classic PTSD / D-PTSD / Control` labels must be
  **study-provided** or applied as an **explicitly documented proxy mapping**.
- Only **public or properly authorised** data is used; **no private patient
  identifiers** are processed or stored.
- See `DATASET_REPORT.md` for the clinical-adjacency discussion (DAIC-WOZ/E-DAIC).

## What this repository contains

| Path | Purpose |
|---|---|
| `notebook/PTSD_Affect_Recognition.ipynb` | **The integrated Colab notebook** (Data → Features → Model → Train → Evaluate → Explain) |
| `src/config.py` | Single source of truth: AU list, class labels, feature dims, seed, paths |
| `src/data_utils.py` | Colab dataset acquisition (Kaggle/gdown/URL), split, integrity check |
| `src/features.py` | Frame-level feature extraction (AU, region, pose, temporal) |
| `src/model.py` | Multi-input late-fusion PyTorch model + Grad-CAM backbone |
| `src/train.py` | AMP + gradient checkpointing + early stopping + Drive checkpoint/resume |
| `src/evaluate.py` | Metrics, confusion matrix, ROC, subtype report, paper tables |
| `src/explain.py` | SHAP feature importance + Grad-CAM overlays |
| `src/tables.py` | LaTeX + Markdown result-table generators |
| `DATASET_REPORT.md` | Ranked dataset discovery report |
| `MANIFEST.md` | Data + artifact provenance ledger |
| `smoke_test.py` | Local CPU verification of the non-GPU parts |

The notebook is **self-contained** (all code inlined under the six section
headers). The `src/*.py` files are the same code in modular form, used for the
local smoke test and easier review/diffing.

## How to run (Google Colab)

1. Open `notebook/PTSD_Affect_Recognition.ipynb` in Colab (File → Upload, or
   `File → Open notebook → GitHub/Drive`).
2. **Runtime → Change runtime type → T4 GPU** (free tier) or A100 (Pro).
3. Run the **README** and **Requirements** cells first (installs everything,
   fixes the seed).
4. Run the **Smoke test** cell (tiny end-to-end run on synthetic data) to confirm
   the environment works before any real download.
5. Proceed through **Data → Features → Model → Train → Evaluate → Explain**.
6. Checkpoints land in Google Drive (`MyDrive/ptsd_affect/checkpoints/`) so a
   session timeout never loses progress; the Train section auto-resumes.

## Feature schema (one row per frame)

| Branch | Input dim | Columns |
|---|---|---|
| Action Units | 17 | `au_AU01 … au_AU45` |
| Region weights | 4 | `region_eye, region_mouth, region_upper_face, region_lower_face` |
| Head pose | 3 | `pose_yaw, pose_pitch, pose_roll` |
| Temporal | 4 | `temp_dwell_time, temp_transition_rate, temp_entropy, temp_au_variability` |

**Total feature vector = 28.** Temporal features use a causal sliding window
(`TEMPORAL_WINDOW = 15`) computed per sequence so windows never bleed across
subjects.

## Model

Four branch encoders (Linear → BatchNorm → ReLU → Dropout) → late-fusion
concatenation → shared MLP → two heads:

1. **Emotion head** (6): `Fear, Anger, Sadness, Neutral, Surprise, Flat/Blunted Affect`
2. **Subtype head** (3): `Classic PTSD, D-PTSD, Control`

## Reproducibility

- Fixed seed `SEED = 42` (`config.seed_everything`).
- `requirements.txt` cell pins the pip-installable stack.
- `MANIFEST.md` records dataset versions, links, licenses, checksums, label
  definitions, figures, checkpoints and result tables.

## Modify this later?

- **Labels / classes** → `src/config.py` (`EMOTION_CLASSES`, `SUBTYPE_CLASSES`).
- **AU list / region weights** → `config.py` (`AU_NAMES`, `REGION_AU_MAP`,
  `REGION_DIAGNOSTIC_WEIGHTS`).
- **Model size / dropout** → `build_model(...)` defaults in `model.py`.
- **Training hyperparameters** → `train.train(...)` args (epochs, lr, patience, …).
- **Colours / figure style** → the Fathom-style palette at the top of
  `evaluate.py` and `explain.py`.
