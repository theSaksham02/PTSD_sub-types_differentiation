# Dataset Discovery Report — Facial Affect Recognition for PTSD Subtype Research

**Purpose.** Rank publicly available facial-expression / affect datasets for an automated pipeline that distinguishes **Classic PTSD**, **Dissociative PTSD (D-PTSD)**, and **Healthy Controls** using facial-region diagnosticity, action units (AUs), head-pose dynamics, and temporal behavioural features.

**Critical clinical caveat (read first).** No public affect dataset carries PTSD diagnoses. The AU/emotion datasets below provide the *feature substrate*; the `Classic PTSD / D-PTSD / Control` labels must be **study-provided** (from the user's own clinical cohort) or applied as an **explicitly documented proxy mapping** (e.g., blunted/flat affect → D-PTSD dissociative profile). This limitation is stated throughout the notebook and README. Datasets that *do* contain psychiatric ground truth (DAIC-WOZ / E-DAIC, AVEC) are listed under *Clinical adjacency*.

---

## Ranking (suitability for PTSD-adjacent AU + temporal recognition)

Rank reflects fit to the pipeline's needs: **AU annotations ≥ sequential/video data ≥ spontaneous/naturalistic ≥ clinical proximity**, balanced against how easily each can be acquired on free Colab.

| Rank | Dataset | AU | Video/Seq | Spontaneous | Colab access | Verdict |
|---|---|---|---|---|---|---|
| 1 | **BP4D / BP4D+** | ✅ FACS + intensity | ✅ 2D+3D video | ✅ | Request form | Gold standard for AU+video |
| 2 | **DISFA / DISFA+** | ✅ 12 AUs @0–5 | ✅ 20 fps video | ✅ | Request form | Best spontaneous AU intensity |
| 3 | **CK+ (Cohn–Kanade)** | ✅ FACS-coded | ✅ 593 sequences | ❌ posed | Direct / Kaggle | Best *free-tier* starter |
| 4 | **AffectNet** | ❌ (landmarks only) | ❌ static | ❌ in-the-wild | Request / Kaggle | Scale for emotion classes |
| 5 | **RAF-DB** | ❌ | ❌ static | ❌ in-the-wild | Kaggle | In-the-wild realism |
| 6 | **OULU-CASIA** | ❌ | ✅ 2,880 sequences | ❌ posed | Request / direct | Illumination robustness |
| 7 | **MAHNOB-HCI** | ⚠️ limited | ✅ video | ✅ naturalistic | Request | Multimodal (EEG) |

**Recommended top-3 for Colab acquisition:** CK+, AffectNet, RAF-DB (all auto-downloadable). DISFA and BP4D are ranked highest *scientifically* but are request-gated, so the notebook provides their request workflow and downloads only the freely accessible three.

---

## Per-dataset detail

### 1. BP4D-Spontaneous (+ BP4D+)
- **What:** 3D dynamic (2D+3D) spontaneous facial expressions; **BP4D** = 41 subjects (23F/18M), 8 emotion-elicitation tasks; **BP4D+** = 140 subjects, ~1,400 video sequences.
- **AU:** ✅ FACS-coded AUs **with intensity**, per frame — exactly what the pipeline needs.
- **License/access:** Free for academic use via **request form** (Binghamton University technology transfer). Not auto-downloadable.
- **Clinical/psychiatric use:** Widely used in affective computing; not PTSD-specific but the *de facto* benchmark for dynamic AU intensity. Strong fit for region diagnosticity + temporal modelling.
- **Suitability:** **#1** — best combination of AU + spontaneous video.

### 2. DISFA (+ DISFA+)
- **What:** Denver Intensity of Spontaneous Facial Action — **27 subjects** viewing emotion-eliciting clips; **~4,844 frames/subject at 20 fps** (DISFA+ adds posed+spontaneous).
- **AU:** ✅ **12 AUs with 0–5 intensity**, manually FACS-coded — the reference dataset for AU *intensity* regression.
- **License/access:** Free academic, **request** from University of Denver (mohammadmahoor.com).
- **Clinical/psychiatric use:** Core dataset for spontaneous AU intensity and reduced-expressivity studies; directly relevant to blunted/flat affect.
- **Suitability:** **#2** — best spontaneous AU intensity, but request-gated.

### 3. CK+ (Extended Cohn–Kanade)
- **What:** **593 sequences** from **123 subjects**; 327 sequences carry one of **7 emotion labels** (anger, contempt, disgust, fear, happiness, sadness, surprise).
- **AU:** ✅ FACS-coded AUs for the sequences (peak frame).
- **License/access:** Free for research; **direct download** (jeffcohn.net / GitHub mirrors) and **Kaggle** mirrors — the easiest dataset to load on free Colab.
- **Clinical/psychiatric use:** Ubiquitous in FER; small and posed, so useful for *pipeline validation* rather than clinical generalisation.
- **Suitability:** **#3** — small, fast, AU-labelled, and trivially Colab-compatible. Ideal smoke-test / baseline dataset.

### 4. AffectNet
- **What:** **~1M in-the-wild images** (~450k manually annotated; ~550k auto); **8 emotion labels** (neutral, happy, sad, surprise, fear, disgust, anger, contempt) + valence/arousal.
- **AU:** ❌ No FACS AUs (provides **68 facial landmarks** + valence/arousal). AUs must be inferred via OpenFace/py-feat at extraction time.
- **License/access:** Research license via **request** (mohammadmahoor.com); numerous **Kaggle** subsets available.
- **Clinical/psychiatric use:** Mostly non-clinical FER; provides the largest emotion-class base for the 6-class head (including a "neutral" class that can seed flat/blunted-affect modelling).
- **Suitability:** **#4** — scale + emotion classes; AU must be re-derived.

### 5. RAF-DB (Real-world Affective Faces)
- **What:** **~29,672 in-the-wild images**; **15,339 single-label** across 7 basic emotions + **3,959 compound** (12 compound classes).
- **AU:** ❌ No AUs.
- **License/access:** Research license; **Kaggle** and whdeng.cn (request).
- **Clinical/psychiatric use:** Non-clinical FER benchmark; compound labels are a poor fit for discrete AU modelling.
- **Suitability:** **#5** — in-the-wild diversity, no AU.

### 6. OULU-CASIA
- **What:** **80 subjects**, **6 expressions** (anger, disgust, fear, happiness, sadness, surprise), **2,880 sequences**, two imaging systems (**NIR + VIS**) × three illumination conditions.
- **AU:** ❌ Expression labels only (no FACS).
- **License/access:** Research agreement via University of Oulu (also mirrored downloads).
- **Clinical/psychiatric use:** Illumination-invariance benchmark; sequential video useful for temporal features but no AU.
- **Suitability:** **#6** — video/sequential but no AU.

### 7. MAHNOB-HCI
- **What:** **30 subjects** (27 usable), multimodal — **32-channel EEG**, video, eye gaze, audio, physiological — responding to movie clips.
- **AU:** ⚠️ Limited; discrete emotion labels + valence/arousal rather than dense FACS.
- **License/access:** Free academic, **request** (mahnob-db.eu).
- **Clinical/psychiatric use:** Multimodal emotion elicitation; naturalistic but EEG-centric. Useful if the study adds physiological correlates, not primary for AU.
- **Suitability:** **#7** — naturalistic + multimodal, weak AU.

---

## Additional finds

| Dataset | Why it matters | AU | Access |
|---|---|---|---|
| **EmotioNet** | ~1M images with **12 AUs** (auto-annotated) — large-scale AU detection | ✅ (auto) | Research request |
| **UNBC-McMaster Shoulder Pain** | Spontaneous pain expressions, **FACS-coded AUs (PSPI)** | ✅ | Request |
| **FER2013** | 35,887 grayscale 48×48 images, 7 emotions — tiny, free (Kaggle) | ❌ | Kaggle |
| **CASME II / SAMM** | **Micro-expression** video — relevant to blunted/flat affect detection | ❌ (micro AUs) | Request |
| **MMI** | 3,400+ video/image samples, posed, some AU (subset) | ⚠️ | Request |

---

## Clinical adjacency (psychiatric ground truth — *not* AU/emotion-labelled)

These contain depression / PTSD labels and multimodal signals but **not** dense AU emotion annotations. They are the correct reference point if the study later wants *clinical* labels:

- **DAIC-WOZ / E-DAIC** — semi-structured interviews (virtual agent "Ellie"); audio, video, text, PHQ-8/PTSD screens. Used in **AVEC 2019**. Request via `dcapswoz.ict.usc.edu`. **Closest public resource to PTSD screening**, but signal is interview-level, not frame-level AU.
- **AVEC 2013 / 2014** — depression detection (audio/video). Request via organisers.
- **AVEC 2019** — cross-cultural depression + PTSD from the E-DAIC corpus.

**Recommendation for the paper's Method section:** state that AU/emotion features are derived from CK+/AffectNet/RAF-DB (and DISFA/BP4D where an academic agreement exists), while the `Classic PTSD / D-PTSD / Control` labels come from the study cohort, with the emotion→subtype head acting as a **documented proxy model**, not a diagnostic claim.

---

## Access-method summary

| Method | Datasets | Colab-friendly |
|---|---|---|
| **Kaggle API / kagglehub** | CK+, AffectNet (subsets), RAF-DB, FER2013 | ✅ auto |
| **gdown / direct URL** | CK+ (GitHub mirrors) | ✅ auto |
| **Academic request form** | BP4D(+), DISFA(+), OULU-CASIA, MAHNOB-HCI, EmotioNet, UNBC | ❌ manual |
| **Kaggle JSON key (upload)** | any Kaggle dataset | ✅ one-time setup |

*Sources verified 2026-08-17 via web search: DISFA (mohammadmahoor.com; 27 subjects, 4,844 frames @20fps), BP4D (binghamton.technologypublisher.com; 41 subjects), AffectNet (mohammadmahoor.com; ~1M images / 450k manually labelled), RAF-DB (~29,672 images), OULU-CASIA (80 subjects, 2,880 sequences), MAHNOB-HCI (30 subjects, 32-ch EEG), DAIC-WOZ/E-DAIC (dcapswoz.ict.usc.edu).*
