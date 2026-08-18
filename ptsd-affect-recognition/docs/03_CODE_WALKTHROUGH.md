# 03 · Code Walkthrough — What Is Happening, File by File

A plain-English map of the repository, in the order data flows through it.

```
data_utils.py  →  features.py  →  model.py  →  train.py  →  evaluate.py  →  explain.py
      │                                │                                     │
      └────────── config.py (constants live here) ────────────────────────────┘
                                             └── tables.py (paper output)
```

---

## `src/config.py` — the single source of truth

Everything shared lives here so nothing can drift out of sync:

| Constant | What it is |
|---|---|
| `EMOTION_CLASSES` | 6 affect states (Fear … Flat/Blunted Affect) |
| `SUBTYPE_CLASSES` | 3 groups (Classic PTSD, D-PTSD, Control) |
| `AU_NAMES` | the 17 canonical action units |
| `REGION_AU_MAP` | which AUs belong to eye / mouth / upper / lower face |
| `REGION_DIAGNOSTIC_WEIGHTS` | literature-informed weights for region scores |
| `FEATURE_NAMES` | the 28-dim feature vector, in order |
| `TEMPORAL_WINDOW` | sliding-window size (15 frames) |
| `seed_everything()` | fixes Python/NumPy/PyTorch seeds |

**Rule of thumb:** to change labels, AU lists, or region weights, edit *only*
this file — every other module reads from it.

---

## `src/data_utils.py` — get the data

- `download_kaggle(slug)` → uses `kagglehub` (no manual key) or the Kaggle API.
- `download_gdown(id)` / `download_url(url)` → fallback downloaders.
- `unzip_all(zip, dest)` → extract an archive.
- `split_image_folders(...)` → reorganise class folders into `train/val/test`.
- `display_sample_grid(...)` → show sample faces + labels (integrity check).
- `print_request_instructions(dataset)` → documents the manual workflow for
  **request-gated** datasets (DISFA, BP4D).

**Key idea:** CK+, AffectNet, RAF-DB auto-download; DISFA/BP4D require a signed
request and are handled with a clear printed checklist instead of a fake URL.

---

## `src/features.py` — turn faces into numbers

One row per face frame, 28 columns:

| Step | Function | Output |
|---|---|---|
| AU intensities | `validate_au_matrix` | 17 values, clipped 0–5 |
| Region scores | `region_scores` | 4 diagnosticity-weighted means |
| Head pose | `validate_pose` | yaw / pitch / roll |
| Temporal | `temporal_features` | dwell time, transition rate, entropy, AU variability |

**Temporal features are causal and per-sequence** — `build_feature_frame` loops
over each `sequence_id` and applies the 15-frame window *within* that sequence,
so statistics never leak between subjects. This is the single most important
correctness detail for a clinical study.

`generate_synthetic_dataset()` builds fake-but-plausible data so the whole
pipeline can be exercised **before** any real (possibly request-gated) data
arrives.

---

## `src/model.py` — the classifier

`MultiInputPTSDAffectModel`:

1. **Four branch encoders** (`BranchMLP`): each input (17 / 4 / 3 / 4) is
   projected to a 32-dim embedding with `Linear → BatchNorm → ReLU → Dropout`.
2. **Late fusion**: the four 32-dim embeddings are concatenated → 128-dim.
3. **Shared MLP** (`FusionMLP`): 128 → 64, with optional gradient checkpointing.
4. **Two heads**: `emotion_head` (→ 6) and `subtype_head` (→ 3).

`forward_flat(x)` splits a flat 28-vector into the four branches — this is what
SHAP calls.

`GradCAMBackbone` is a small CNN used **only** to visualise which face regions
the auxiliary image pathway attends to. The primary model never sees raw images.

`summary()` prints the parameter table (27,017 trainable).

---

## `src/train.py` — the loop

- `AffectFeatureDataset` slices a feature DataFrame into the four branch tensors.
- `compute_class_weights` → inverse-frequency weights (handles imbalance).
- `MultiTaskLoss` → `CE(emotion) + λ · CE(subtype)`.
- `train_one_epoch` → forward, backward, AMP-scaler step.
- `save_checkpoint` / `resume_from_checkpoint` → full state (model, optimizer,
  scaler, epoch, best loss, class weights, history).
- `train(...)` → the orchestrator with early stopping (patience 10) and Drive
  checkpoint writing.

---

## `src/evaluate.py` — measure it

- `per_class_metrics` → precision / recall / F1 per emotion.
- `attach_auc` → one-vs-rest ROC AUC per class.
- `plot_confusion_matrix` → heatmap with fear–surprise and anger–disgust cells
  outlined in the highlight colour.
- `plot_roc_curves` → per-class curves + macro-average + chance line.
- `subtype_report` → accuracy, macro-F1, Cohen's κ.
- `run_evaluation` → one call that does all of the above and returns paths.

Figures use the Fathom-style scientific palette (navy / grey / one warm
highlight) at 300 dpi.

---

## `src/explain.py` — interpret it

- `shap_feature_importance` → `shap.GradientExplainer` over a small background
  set; one horizontal bar chart per subtype, ranked by mean |SHAP|.
- `grad_cam` / `save_gradcam_grid` → heatmaps from the CNN backbone's final
  conv layer, overlaid on face crops.

---

## `src/tables.py` — write the paper

- `emotion_table_markdown/latex` → `Emotion Class | Precision | Recall | F1 | AUC`.
- `subtype_table_markdown/latex` → `Accuracy / Macro-F1 / Cohen's Kappa`.
- `save_tables` → writes all four files (`.md` + `.tex`), copy-paste ready.

---

## The notebook (`notebook/PTSD_Affect_Recognition.ipynb`)

Self-contained. It **writes the `src/*.py` files to `/content/pipeline/`** and
imports them, so it mirrors the repository exactly. Sections:

`README → Requirements → Smoke test → Data → Features → Model → Train → Evaluate → Explain`

The **Smoke test** cell runs a tiny synthetic end-to-end pass first — always run
it before any real download to catch environment problems cheaply.

→ Back to [01 · What is remaining](01_WHAT_IS_REMAINING.md) · Next: [04 · Architecture](04_ARCHITECTURE.md)
