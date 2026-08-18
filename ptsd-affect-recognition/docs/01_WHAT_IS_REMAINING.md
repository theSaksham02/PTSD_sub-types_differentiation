# 01 · What Is Remaining?

A plain-language status of the PTSD affect-recognition project — what is done,
what is left, and in what order to do it. **Green = shipped. Amber = you must
provide something. Grey = not started.**

---

## Done ✅ (shipped to `main`)

| # | Item | Evidence |
|---|---|---|
| 1 | Pipeline design + 28-dim feature schema | `src/config.py`, `src/features.py` |
| 2 | Multi-input late-fusion model (4 inputs → 2 heads) | `src/model.py` — 27,017 params, verified forward pass |
| 3 | Colab training loop (AMP, checkpointing, early-stop, Drive resume) | `src/train.py` |
| 4 | Evaluation + confusion matrix + ROC + subtype report | `src/evaluate.py` |
| 5 | SHAP + Grad-CAM explainability | `src/explain.py` |
| 6 | LaTeX/Markdown table generators | `src/tables.py` |
| 7 | Integrated Colab notebook | `notebook/PTSD_Affect_Recognition.ipynb` |
| 8 | Dataset discovery report (13 datasets, ranked) | `DATASET_REPORT.md` |
| 9 | Data/artifact provenance manifest | `MANIFEST.md` |
| 10 | Verification: 15/15 + 11/11 smoke checks pass | `smoke_test.py`, `model_smoke_test.py` |

## Remaining — in priority order

### 🔴 Blockers (need something from you / external approval)

| # | Item | Why it blocks | What you must do |
|---|---|---|---|
| R1 | **Real dataset acquisition** | The pipeline runs on synthetic data today | Request/accept licenses for CK+, AffectNet, RAF-DB; **sign request forms for DISFA & BP4D** (they are the best AU datasets and are not auto-downloadable) |
| R2 | **PTSD/subtype labels** | No public dataset contains `Classic PTSD / D-PTSD / Control` | Provide your study cohort labels **or** formally document a proxy mapping |
| R3 | **Ethics / IRB approval** | Real patient data + any claim about clinical groups | Get IRB/ethics sign-off before touching patient data |

### 🟡 High-value work (no external blocker — just run it)

| # | Item | Notes |
|---|---|---|
| R4 | **Run the notebook on T4/A100 end-to-end** | Real training, not the synthetic demo. See `02_GPU_TRAINING_GUIDE.md` |
| R5 | **Subject-level train/val/test split** | Current demo uses a sequential split → leakage risk for clinical data |
| R6 | **Wire py-feat / OpenFace output** into `build_feature_frame` | The AU+pose extraction call site is stubbed in the notebook |
| R7 | **Run SHAP at scale** | Verified by syntax; needs the `shap` package + a trained model on GPU |

### ⚪ Nice-to-have (paper polish)

| # | Item |
|---|---|
| R8 | Tune `REGION_DIAGNOSTIC_WEIGHTS` against literature |
| R9 | Add a CNN "image teacher" branch if you want end-to-end image inputs |
| R10 | Statistical significance tests (DeLong for AUC, bootstrap CIs) |
| R11 | Ablation study (drop each of the 4 feature streams) |

---

## One-line answer

> The **code is finished and verified**. What remains is **real data + real labels +
> ethics approval + one GPU training run** — none of which the code can fabricate.
> Everything after that is paper polish.

Next read: `02_GPU_TRAINING_GUIDE.md` → `03_CODE_WALKTHROUGH.md` → `05_UNDERGRAD_RESEARCHER_GUIDE.md`.
