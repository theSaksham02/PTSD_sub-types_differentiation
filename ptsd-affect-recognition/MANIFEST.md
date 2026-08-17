# Manifest — Data & Artifact Ledger

Provenance record for the paper. Update access dates, checksums and versions
after each real acquisition run. **Fill placeholders (marked `TODO`) when you
actually download data and receive request-gated datasets.**

---

## 1. Datasets

| Field | CK+ | AffectNet | RAF-DB | DISFA | BP4D |
|---|---|---|---|---|---|
| Version/edition | Extended Cohn–Kanade | AffectNet (8-emotion) | RAF-DB (single-label) | DISFA/DISFA+ | BP4D-Spontaneous |
| Access method | Direct URL / Kaggle | Request / Kaggle subset | Kaggle | Academic request | Academic request |
| License | Research-only | Research-only | Research-only | Academic agreement | Academic agreement |
| AU availability | FACS-coded (peak) | ❌ (68 landmarks) | ❌ | ✅ 12 AUs @0–5 | ✅ FACS + intensity |
| Subjects / size | 123 / 593 seq | ~450k labelled | 29,672 img | 27 subj | 41 subj |
| Clinical use | Baseline only | Non-clinical | Non-clinical | Reduced-expressivity studies | Affective-computing benchmark |
| Source URL | jeffcohn.net / Kaggle | mohammadmahoor.com | whdeng.cn / Kaggle | mohammadmahoor.com | binghamton.technologypublisher.com |
| Access date | `TODO` | `TODO` | `TODO` | `TODO` | `TODO` |
| Checksum (archive) | `TODO` | `TODO` | `TODO` | `TODO` | `TODO` |

### Label definitions (for the paper's Method section)

- **Emotion head (6):** `Fear, Anger, Sadness, Neutral, Surprise, Flat/Blunted Affect`
  — `Flat/Blunted Affect` is a *study-defined* class (reduced AU intensity /
  restricted repertoire), not a native label of any source dataset; it must be
  derived from AU-intensity thresholds or a proxy mapping, and this mapping must
  be reported.
- **Subtype head (3):** `Classic PTSD, D-PTSD, Control` — **study-provided only**
  (no public dataset provides these). Document the mapping/exclusion criteria
  used to assign them.
- **AU intensity scale:** 0–5 (FACS / OpenFace convention).
- **Head-pose:** yaw / pitch / roll in degrees.

---

## 2. Feature schema (28-dim)

`AU(17) + region(4) + pose(3) + temporal(4)` — see `README.md` for column names.
Temporal window = 15 frames (causal, per-sequence).

---

## 3. Generated artifacts (to be populated at run time)

| Artifact | Path pattern | Notes |
|---|---|---|
| Frame feature CSV | `/content/data/features/*.csv` | one row per frame |
| Checkpoints | `MyDrive/ptsd_affect/checkpoints/best.pt` | model + optimizer + scaler + epoch |
| Confusion matrix | `figures/confusion_matrix.png` | 300 dpi |
| ROC curves | `figures/roc_curves.png` | 300 dpi |
| SHAP bar charts | `figures/shap_*.png` | one per subtype |
| Grad-CAM grid | `figures/gradcam_grid.png` | 300 dpi |
| Emotion table | `figures/table_emotion_metrics.{md,tex}` | copy-paste ready |
| Subtype table | `figures/table_subtype_metrics.{md,tex}` | copy-paste ready |
| Training history | `history.json` (in checkpoint `extra`) | loss + acc curves |

---

## 4. Model record

- **Architecture:** `MultiInputPTSDAffectModel` — 4 branch MLPs → late-fusion →
  shared MLP → 2 heads (6 + 3). Dropout + BatchNorm throughout.
- **Auxiliary visual model:** `GradCAMBackbone` (small CNN) used **only** for
  Grad-CAM visualisation, not for the primary classification.
- **Training:** AdamW, AMP (`torch.cuda.amp`), cosine LR schedule, class-weighted
  cross-entropy (both heads), early stopping (patience 10), gradient-checkpointing
  toggle on the fusion MLP.

---

## 5. Compliance checklist (for IRB / ethics appendix)

- [ ] Only public or request-authorized data used.
- [ ] No private patient identifiers stored.
- [ ] `Flat/Blunted Affect` and subtype labels documented as proxy/study-defined.
- [ ] Model labelled **research-only**, not diagnostic.
- [ ] Request agreements (DISFA/BP4D/DAIC-WOZ) signed and on file.
