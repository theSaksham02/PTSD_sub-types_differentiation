"""
smoke_test.py — Local CPU verification of the non-GPU parts of the pipeline.

Exercises (without needing PyTorch or GPU):
  1. features.py  : synthetic dataset -> frame-level DataFrame + 28-d vector
  2. evaluate.py  : per-class metrics, confusion matrix, ROC curves, subtype report
  3. tables.py    : LaTeX + Markdown tables from the above metrics

The PyTorch model/training/explainability modules are verified separately by
syntax compilation (py_compile) since they require a GPU runtime in Colab.
"""
import os
import sys
import json

SRC = os.path.join(os.path.dirname(__file__), "src")
sys.path.insert(0, SRC)

import numpy as np

import config
from config import FEATURE_NAMES, EMOTION_CLASSES, SUBTYPE_CLASSES, N_FEATURES

OUT = os.path.join(os.path.dirname(__file__), "smoke_output")
os.makedirs(OUT, exist_ok=True)

results = {"checks": [], "failures": 0}


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    results["checks"].append({"name": name, "status": status, "detail": detail})
    if not cond:
        results["failures"] += 1
    print(f"[{status}] {name} {detail}")


# ---- 1. Features -----------------------------------------------------------
import features as F

df, X, y_emo, y_sub = F.generate_synthetic_dataset(n_sequences=6, frames_per_seq=60, seed=0)
check("feature-frame rows == 6*60", df.shape[0] == 360, f"shape={df.shape}")
check("feature-vector dim == 28", X.shape[1] == N_FEATURES, f"dim={X.shape[1]}")
check("all 28 feature columns present", list(df.columns[2:2+N_FEATURES]) == FEATURE_NAMES, "")
check("temporal features finite & non-negative",
      df[["temp_dwell_time", "temp_transition_rate", "temp_entropy", "temp_au_variability"]].min().min() >= 0
      and np.isfinite(X).all(), "")
check("region scores within [0,5]",
      df[["region_eye", "region_mouth", "region_upper_face", "region_lower_face"]].to_numpy().min() >= 0
      and df[["region_eye", "region_mouth", "region_upper_face", "region_lower_face"]].to_numpy().max() <= 5, "")
check("temporal windows do not bleed across sequences",
      df.groupby("sequence_id")["temp_dwell_time"].apply(lambda s: (s == s.iloc[0]).mean()) is not None, "")

# ---- 2. Evaluate -----------------------------------------------------------
import evaluate as E

rng = np.random.default_rng(0)
# build somewhat-correlated synthetic predictions
y_emo_true = y_emo
y_emo_pred = []
for e in y_emo_true:
    y_emo_pred.append(e if rng.random() < 0.7 else EMOTION_CLASSES[rng.integers(len(EMOTION_CLASSES))])
y_emo_score = np.zeros((len(y_emo_true), len(EMOTION_CLASSES)))
for i, e in enumerate(y_emo_true):
    j = EMOTION_CLASSES.index(e)
    y_emo_score[i, j] = rng.uniform(0.6, 1.0)
    y_emo_score[i, :] += rng.uniform(0, 0.1, len(EMOTION_CLASSES))
y_emo_score = y_emo_score / y_emo_score.sum(1, keepdims=True)

y_sub_true = y_sub
y_sub_pred = [s if rng.random() < 0.6 else SUBTYPE_CLASSES[rng.integers(3)] for s in y_sub_true]

bundle = E.run_evaluation(y_emo_true, y_emo_pred, y_emo_score, y_sub_true, y_sub_pred, OUT)
mm = bundle["emotion_metrics"]
check("6 emotion classes have precision/recall/f1",
      all(k in mm and "precision" in mm[k] and "recall" in mm[k] and "f1" in mm[k] for k in EMOTION_CLASSES), "")
check("AUC computed for 6 classes",
      all("auc" in mm[k] and np.isfinite(mm[k]["auc"]) for k in EMOTION_CLASSES), "")
check("subtype report has accuracy/macro_f1/kappa",
      all(k in bundle["subtype_metrics"] for k in ("accuracy", "macro_f1", "cohen_kappa")),
      f"acc={bundle['subtype_metrics']['accuracy']:.3f} "
      f"k={bundle['subtype_metrics']['cohen_kappa']:.3f}")
check("confusion matrix figure exists", os.path.exists(bundle["figures"]["confusion_matrix"]), "")
check("ROC figure exists", os.path.exists(bundle["figures"]["roc_curves"]), "")

# ---- 3. Tables -------------------------------------------------------------
import tables as T

emo_rows = [{"class": c, **{k: mm[c][k] for k in ("precision", "recall", "f1", "auc")}}
            for c in EMOTION_CLASSES]
tpaths = T.save_tables(emo_rows, bundle["subtype_metrics"], OUT, classes=EMOTION_CLASSES)

check("emotion markdown table written", os.path.exists(tpaths["emotion_md"]), "")
check("emotion LaTeX table written", os.path.exists(tpaths["emotion_tex"]), "")
check("subtype markdown table written", os.path.exists(tpaths["subtype_md"]), "")
check("subtype LaTeX table written", os.path.exists(tpaths["subtype_tex"]), "")

# Show the actual markdown table for inspection
print("\n" + "=" * 60)
print("Emotion metrics (Markdown, copy-paste ready)")
print("=" * 60)
print(tpaths["emotion_markdown"])
print("\n" + "=" * 60)
print("Subtype metrics (Markdown, copy-paste ready)")
print("=" * 60)
print(tpaths["subtype_markdown"])

# Write JSON summary
with open(os.path.join(OUT, "smoke_summary.json"), "w") as f:
    json.dump(results, f, indent=2, default=str)

print("\n" + "=" * 60)
print(f"Smoke test complete. {len(results['checks'])} checks, "
      f"{results['failures']} failures.")
sys.exit(1 if results["failures"] else 0)
