# 05 · Undergraduate Researcher Guide — What Else You Need (Beyond the Code)

The code is ~20% of a publishable research project. This is the other 80% —
organised as a checklist so you can track it.

---

## A. Ethics & approvals — do this first, it gates everything

| # | Item | Notes |
|---|---|---|
| 1 | **IRB / ethics approval** | Required before any real patient or participant data. Your supervisor applies through your university. |
| 2 | **Dataset licenses** | CK+/AffectNet/RAF-DB = research license. DISFA/BP4D = signed academic agreement. DAIC-WOZ = separate request. Keep copies in a folder. |
| 3 | **Data protection** | No names, no faces that can identify a person beyond what the license allows. Store on university-approved storage, not personal Drive. |
| 4 | **No diagnostic claim** | Your paper must state this is *exploratory*, not a diagnostic tool. |

**Why first:** if you collect data *then* discover you need approval, you've lost
the work. Get the paperwork started today.

---

## B. Experimental design — the part reviewers actually check

| # | Decision | Recommendation |
|---|---|---|
| 1 | **Subject-level split** | Train/val/test must split *by subject*, never by frame. Otherwise the model "sees" the same person in train and test → inflated results. |
| 2 | **Class imbalance** | PTSD subtypes are rarely balanced. Report the class distribution; use class-weighted loss (already in the code) and report per-class metrics, not just accuracy. |
| 3 | **Label definition** | Define *exactly* how `Classic PTSD / D-PTSD / Control` is assigned (e.g. CAPS-5 score, dissociation subscale). This is the single biggest validity question. |
| 4 | **Baseline** | Compare against a trivial baseline (majority class) so reviewers see your model beats chance. |
| 5 | **Ablation** | Drop each of the 4 feature streams in turn. Shows which signal matters (AU vs temporal vs pose). |
| 6 | **Significance** | DeLong test for AUC differences; bootstrap 95% CIs for F1/Kappa. |
| 7 | **Sample size** | With 3 groups, you need enough participants per group for any subgroup claim — discuss power with your supervisor *before* collecting. |

---

## C. Metrics to report (psychology-journal convention)

| Task | Metric | Why |
|---|---|---|
| 6 emotions | Precision / Recall / F1 per class | Handles imbalance |
| 6 emotions | ROC-AUC per class + macro | Threshold-independent |
| Confusion | Confusion matrix | Shows *fear–surprise*, *anger–disgust* confusions |
| 3 subtypes | Accuracy, Macro-F1, **Cohen's κ** | κ corrects for chance agreement |
| Explainability | SHAP rankings + Grad-CAM | Ties model to theory |

The code already emits **all of these** as 300-dpi figures and copy-paste LaTeX/Markdown.

---

## D. Literature — minimum reading to write the paper

| Topic | Key sources |
|---|---|
| FACS / AUs | Ekman & Friesen (1978) |
| PTSD facial expressivity | reduced/blunted affect literature (search: "PTSD reduced facial expressivity", "dissociative subtype affect") |
| AU datasets | CK+, DISFA, BP4D, AffectNet, RAF-DB papers (see README references) |
| Interpretability | SHAP (Lundberg & Lee 2017), Grad-CAM (Selvaraju et al. 2017) |

Keep a Zotero/Mendeley library; cite the dataset papers for *every* dataset you use.

---

## E. Reproducibility & version control

| # | Item |
|---|---|
| 1 | Pin the seed (`SEED=42`) — already done. |
| 2 | Record **exact package versions** (`pip freeze > requirements.lock.txt`) after your successful run. |
| 3 | Git-commit every change; tag the exact commit that produced your results. |
| 4 | Save the feature CSV + checkpoint + generated figures with a date stamp, and log them in `MANIFEST.md`. |
| 5 | Keep a lab notebook (even a `notes/` folder) of decisions — reviewers and your future self will need it. |

---

## F. Writing the paper — what the code *won't* do

1. **Abstract, Introduction, Related Work** — you write these.
2. **Methods** — describe the datasets, feature extraction, model, and training *in prose* (the docs here are a good skeleton).
3. **Results** — paste the generated tables/figures; don't cherry-pick, report all classes.
4. **Discussion** — interpret SHAP/Grad-CAM against theory; state limitations.
5. **Limitations** — small sample, posed vs spontaneous mismatch, proxy labels, no external validation.

---

## G. Skill checklist for you (undergrad)

- [ ] Python + PyTorch basics (you already ran the pipeline).
- [ ] Git (clone / commit / push — you've done this).
- [ ] A little statistics (κ, CI, DeLong — ask your supervisor for help).
- [ ] Reading primary papers, not just summaries.
- [ ] Writing a clear Methods section.

---

## TL;DR

> **Code is done.** What remains is, in order: **ethics approval → dataset
> licenses → real labels → one GPU run → significance tests → write-up.** The
> single most important thing to nail down *now* is **how your PTSD-subtype
> labels are defined**, because every downstream result depends on it.

→ Back to [01 · What is remaining](01_WHAT_IS_REMAINING.md)
