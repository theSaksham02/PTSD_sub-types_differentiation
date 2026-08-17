"""
evaluate.py — Evaluation, confusion analysis, ROC curves and paper-ready tables.

Produces:
  1. Per-class precision / recall / F1 for the six emotion states.
  2. Confusion-matrix heatmap with fear-surprise and anger-disgust pairs
     visually highlighted.
  3. PTSD subtype classification report (Classic PTSD / D-PTSD / Control).
  4. One-vs-rest ROC curves per emotion class (macro/micro averages).
  5. A summary table for direct paste into a psychology journal paper.

Plots follow a Fathom Information Design "scientific journal" aesthetic:
neutral grays + navy + one highlight colour, restrained gridlines, high DPI.
"""
from __future__ import annotations

import os
import numpy as np
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_fscore_support, cohen_kappa_score, accuracy_score,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import EMOTION_CLASSES, SUBTYPE_CLASSES, CONFUSION_PAIRS

# Fathom-style palette: navy, mid-gray, one warm highlight, neutral face
NAVY = "#1F3A5F"
GRAY = "#8A919C"
HIGHLIGHT = "#C75B39"
LIGHT = "#ECEFF3"
FACE = "#F7F6F3"


def _apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": FACE,
        "axes.facecolor": FACE,
        "axes.edgecolor": "#444",
        "axes.grid": True,
        "grid.color": LIGHT,
        "grid.linewidth": 0.6,
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelcolor": "#222",
        "xtick.color": "#444",
        "ytick.color": "#444",
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def per_class_metrics(y_true: list[str], y_pred: list[str],
                      classes: list[str]) -> dict:
    """Return per-class precision/recall/f1 + AUC placeholders (filled later)."""
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=classes,
                                                 average=None, zero_division=0)
    return {c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i])}
            for i, c in enumerate(classes)}


def attach_auc(metrics: dict, y_true: list[str], y_scores: np.ndarray,
               classes: list[str]) -> dict:
    """Attach one-vs-rest ROC AUC per class given predicted probabilities."""
    y_true_bin = _one_hot(y_true, classes)
    for i, c in enumerate(classes):
        try:
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
            metrics[c]["auc"] = float(auc(fpr, tpr))
            metrics[c]["fpr"] = fpr
            metrics[c]["tpr"] = tpr
        except ValueError:
            metrics[c]["auc"] = float("nan")
    return metrics


def _one_hot(y: list[str], classes: list[str]) -> np.ndarray:
    idx = {c: i for i, c in enumerate(classes)}
    out = np.zeros((len(y), len(classes)), dtype=np.int8)
    for i, lab in enumerate(y):
        out[i, idx[lab]] = 1
    return out


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------
def plot_confusion_matrix(y_true: list[str], y_pred: list[str],
                          classes: list[str], out_path: str,
                          highlight_pairs: list[tuple] | None = None,
                          normalize: str = "true") -> str:
    """Confusion matrix heatmap; highlights the given (true, pred) confusion
    pairs with a warm outline + annotation. Returns the saved path."""
    highlight_pairs = highlight_pairs if highlight_pairs is not None else CONFUSION_PAIRS
    _apply_style()

    cm = confusion_matrix(y_true, y_pred, labels=classes)
    if normalize == "true":
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1e-9)
    elif normalize == "pred":
        cm_norm = cm.astype(float) / cm.sum(axis=0, keepdims=True).clip(min=1e-9)
    else:
        cm_norm = cm.astype(float)

    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")

    # highlight confusion pairs
    idx = {c: i for i, c in enumerate(classes)}
    for a, b in highlight_pairs:
        if a in idx and b in idx:
            for (ti, pi) in [(idx[a], idx[b]), (idx[b], idx[a])]:
                ax.add_patch(plt.Rectangle((pi - 0.5, ti - 0.5), 1, 1,
                                           fill=False, edgecolor=HIGHLIGHT,
                                           linewidth=2.2, zorder=3))

    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Emotion confusion matrix (row-normalised)")

    # annotate counts + fraction
    thresh = cm_norm.max() / 2.0
    for i in range(len(classes)):
        for j in range(len(classes)):
            label = f"{cm[i, j]}\n{cm_norm[i, j]:.2f}"
            ax.text(j, i, label, ha="center", va="center",
                    color="white" if cm_norm[i, j] > thresh else "#1a1a1a",
                    fontsize=8)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# ROC curves
# ---------------------------------------------------------------------------
def plot_roc_curves(y_true: list[str], y_scores: np.ndarray,
                    classes: list[str], out_path: str) -> str:
    """One-vs-rest ROC curves + macro/micro average. Returns saved path."""
    _apply_style()
    y_bin = _one_hot(y_true, classes)
    n = len(classes)
    colors = [NAVY, GRAY, HIGHLIGHT, "#5B7A9D", "#A6B0BC", "#8C5A3B"]

    fig, ax = plt.subplots(figsize=(6.6, 6.0))
    tprs, aucs = [], []
    mean_fpr = np.linspace(0, 1, 100)
    for i, c in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_scores[:, i])
        a = auc(fpr, tpr)
        aucs.append(a)
        ax.plot(fpr, tpr, color=colors[i], lw=1.8,
                label=f"{c} (AUC={a:.2f})")
        interp = np.interp(mean_fpr, fpr, tpr)
        interp[0] = 0.0
        tprs.append(interp)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    macro_auc = auc(mean_fpr, mean_tpr)
    ax.plot(mean_fpr, mean_tpr, color=NAVY, linestyle="--", lw=2.2,
            label=f"Macro-avg (AUC={macro_auc:.2f})")

    ax.plot([0, 1], [0, 1], color="#bbb", linestyle=":", lw=1.2, label="Chance")
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("One-vs-rest ROC curves (emotion states)")
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Subtype report
# ---------------------------------------------------------------------------
def subtype_report(y_true: list[str], y_pred: list[str],
                   classes: list[str] | None = None) -> dict:
    """Accuracy, macro-F1 and Cohen's kappa for the 3-way subtype task."""
    classes = classes or SUBTYPE_CLASSES
    y_true = list(y_true)
    y_pred = list(y_pred)
    acc = accuracy_score(y_true, y_pred)
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=classes,
                                                 average="macro", zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred, labels=classes)
    report_str = classification_report(y_true, y_pred, labels=classes,
                                       zero_division=0, digits=3)
    return {
        "accuracy": float(acc),
        "macro_f1": float(f),
        "cohen_kappa": float(kappa),
        "n_samples": len(y_true),
        "report": report_str,
    }


# ---------------------------------------------------------------------------
# Convenience: full evaluation bundle
# ---------------------------------------------------------------------------
def run_evaluation(y_emo_true, y_emo_pred, y_emo_score,
                   y_sub_true, y_sub_pred, out_dir: str) -> dict:
    """Run the whole evaluation and return a dict of metrics + figure paths."""
    os.makedirs(out_dir, exist_ok=True)

    emo_metrics = per_class_metrics(y_emo_true, y_emo_pred, EMOTION_CLASSES)
    emo_metrics = attach_auc(emo_metrics, y_emo_true, np.asarray(y_emo_score),
                             EMOTION_CLASSES)

    cm_path = plot_confusion_matrix(y_emo_true, y_emo_pred, EMOTION_CLASSES,
                                    os.path.join(out_dir, "confusion_matrix.png"))
    roc_path = plot_roc_curves(y_emo_true, np.asarray(y_emo_score),
                               EMOTION_CLASSES,
                               os.path.join(out_dir, "roc_curves.png"))
    sub = subtype_report(y_sub_true, y_sub_pred)

    return {
        "emotion_metrics": emo_metrics,
        "subtype_metrics": sub,
        "figures": {"confusion_matrix": cm_path, "roc_curves": roc_path},
    }
