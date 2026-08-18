"""
generate_diagrams.py — Render the explanatory diagrams for the docs/ folder.

Style: "11 Build" luxury-minimalism + Fathom scientific restraint —
off-white background, one navy accent, thin lines, generous whitespace,
no gradients, no heavy shadows, no neon.

Outputs 5 PNGs into docs/figures/ at 200 dpi.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "docs", "figures")
os.makedirs(OUT, exist_ok=True)

# palette
NAVY = "#1F3A5F"
INK = "#1B1B1B"
GRAY = "#8A919C"
FAINT = "#ECEFF3"
PAPER = "#FBFAF8"
ACCENT = "#C75B39"
GREEN = "#3E7D5A"

plt.rcParams.update({
    "figure.facecolor": PAPER, "axes.facecolor": PAPER,
    "font.family": "DejaVu Sans", "text.color": INK,
    "axes.edgecolor": "#444", "savefig.bbox": "tight", "figure.dpi": 200,
})


def box(ax, x, y, w, h, text, fc="#FFFFFF", ec=NAVY, tc=INK, fs=9, lw=1.4,
        bold=False, rounded=0.06):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.02,rounding_size={rounded}",
                       fc=fc, ec=ec, lw=lw, zorder=2)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tc, zorder=3, fontweight="bold" if bold else "normal")


def arrow(ax, x1, y1, x2, y2, color=GRAY, lw=1.4):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13,
                        color=color, lw=lw, zorder=1)
    ax.add_patch(a)


def blank_axes(w=10, h=6):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, w); ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=200)
    plt.close(fig)
    print("wrote", p)


# ---------------------------------------------------------------- 1. Pipeline
fig, ax = blank_axes(12, 5)
stages = [
    ("DATA", "Public datasets\nCK+ · AffectNet · RAF-DB\n(DISFA · BP4D on request)"),
    ("FEATURES", "py-feat / OpenFace\n28-dim frame vector"),
    ("MODEL", "Late-fusion MLP\n4 inputs → 2 heads"),
    ("TRAIN", "AMP · early stop\nDrive checkpoints"),
    ("EVALUATE", "P/R/F1 · ROC\nconfusion · tables"),
    ("EXPLAIN", "SHAP\nGrad-CAM"),
]
n = len(stages)
bw, bh, gap = 1.75, 1.6, 0.28
x = 0.5
for i, (title, sub) in enumerate(stages):
    box(ax, x, 1.7, bw, bh, "", fc="#FFFFFF", ec=NAVY)
    ax.text(x + bw / 2, 2.6, title, ha="center", va="center", fontsize=10,
            color=NAVY, fontweight="bold")
    ax.text(x + bw / 2, 2.1, sub, ha="center", va="center", fontsize=7.2, color=INK)
    if i < n - 1:
        arrow(ax, x + bw, 2.5, x + bw + gap, 2.5)
    x += bw + gap
ax.text(6, 0.9, "Data → Features → Model → Train → Evaluate → Explain",
        ha="center", fontsize=10, color=GRAY, style="italic")
save(fig, "01_pipeline_overview.png")


# ---------------------------------------------------------------- 2. Model arch
fig, ax = blank_axes(12, 7.2)
inputs = [("Action Units\n(17)", 0.7), ("Region scores\n(4)", 3.7),
          ("Head pose\n(3)", 6.7), ("Temporal\n(4)", 9.7)]
for label, x in inputs:
    box(ax, x, 5.6, 1.6, 1.0, label, ec=GRAY, fs=8.5)
    arrow(ax, x + 0.8, 5.6, x + 0.8, 4.9, color=GRAY)
# branch MLPs
for label, x in inputs:
    box(ax, x, 3.9, 1.6, 1.0, "Branch\nMLP", ec=NAVY, fs=8.5)
    arrow(ax, x + 0.8, 3.9, 5.95, 2.7, color=GRAY)
# fusion
box(ax, 4.4, 2.3, 3.2, 0.9, "Late fusion · concat (128)", ec=NAVY, bold=True, fs=9)
arrow(ax, 6.0, 2.3, 6.0, 1.6, color=GRAY)
box(ax, 4.4, 0.7, 3.2, 0.9, "Shared MLP (128 → 64)\nDropout + BatchNorm", ec=NAVY, fs=8.5)
# heads
arrow(ax, 4.4, 1.15, 2.2, 1.15, color=GRAY)
arrow(ax, 7.6, 1.15, 9.8, 1.15, color=GRAY)
box(ax, 0.6, 0.7, 1.6, 0.9, "Emotion head\n(6)", ec=ACCENT, fs=8.5, bold=True)
box(ax, 9.8, 0.7, 1.6, 0.9, "Subtype head\n(3)", ec=ACCENT, fs=8.5, bold=True)
ax.text(6, 6.8, "MultiInputPTSDAffectModel", ha="center", fontsize=12,
        color=NAVY, fontweight="bold")
ax.text(6, 6.35, "27,017 trainable parameters", ha="center", fontsize=9, color=GRAY)
save(fig, "02_model_architecture.png")


# ---------------------------------------------------------------- 3. Feature schema
fig, ax = blank_axes(12, 7)
groups = [
    ("Action Units", "17", ["AU01–AU45 intensities (0–5)", "FACS convention"], NAVY),
    ("Region weights", "4", ["eye · mouth", "upper_face · lower_face"], NAVY),
    ("Head pose", "3", ["yaw · pitch · roll", "(degrees)"], NAVY),
    ("Temporal", "4", ["dwell time", "transition rate", "entropy", "AU variability"], NAVY),
]
bw, bh, gap = 2.6, 3.4, 0.35
x = 0.6
for i, (title, dim, items, color) in enumerate(groups):
    box(ax, x, 2.6, bw, bh, "", fc="#FFFFFF", ec=color)
    ax.text(x + bw / 2, 5.45, title, ha="center", va="center", fontsize=10,
            color=color, fontweight="bold")
    ax.text(x + bw / 2, 4.95, dim, ha="center", va="center", fontsize=15,
            color=color, fontweight="bold")
    body = "\n".join(items)
    ax.text(x + bw / 2, 3.6, body, ha="center", va="center", fontsize=7.6, color=INK)
    x += bw + gap
ax.text(6, 1.6, "28-dim feature vector  =  17 + 4 + 3 + 4",
        ha="center", fontsize=11, color=ACCENT, fontweight="bold")
ax.text(6, 1.15, "one row per face frame · temporal features use a causal 15-frame window per sequence",
        ha="center", fontsize=8.5, color=GRAY, style="italic")
save(fig, "03_feature_schema.png")


# ---------------------------------------------------------------- 4. GPU training flow
fig, ax = blank_axes(12, 6.4)
flow = [
    ("1 · Mount Drive", "checkpoints survive\nsession timeout"),
    ("2 · Load features", "frame CSV →\nAffectFeatureDataset"),
    ("3 · Subject split", "train / val\n(no leakage)"),
    ("4 · Train (AMP)", "mixed precision ·\nclass-weighted loss"),
    ("5 · Early stop", "patience = 10\nbest val loss"),
    ("6 · Resume", "restore model +\noptimizer + scaler"),
]
n = len(flow)
bw, bh, gap = 1.7, 1.9, 0.22
x = 0.5
for i, (title, sub) in enumerate(flow):
    box(ax, x, 2.3, bw, bh, "", fc="#FFFFFF", ec=NAVY)
    ax.text(x + bw / 2, 3.5, title, ha="center", va="center", fontsize=8.8,
            color=NAVY, fontweight="bold")
    ax.text(x + bw / 2, 2.7, sub, ha="center", va="center", fontsize=7.0, color=INK)
    if i < n - 1:
        arrow(ax, x + bw, 3.25, x + bw + gap, 3.25)
    x += bw + gap
ax.text(6, 1.5, "T4 (16 GB) → A100 (40 GB): same code, larger batch_size",
        ha="center", fontsize=9.5, color=GRAY, style="italic")
ax.text(6, 0.95, "OOM? halve batch_size → ensure use_amp + use_gradient_checkpointing = True",
        ha="center", fontsize=8.5, color=ACCENT)
save(fig, "04_gpu_training_flow.png")


# ---------------------------------------------------------------- 5. Research roadmap
fig, ax = blank_axes(12, 7)
rows = [
    ("SHIPPED", GREEN, ["Pipeline + model code", "Evaluation + explainability", "Dataset report + manifest", "Smoke tests 15/15 + 11/11"]),
    ("BLOCKERS", ACCENT, ["Real dataset licenses (DISFA/BP4D forms)", "PTSD subtype labels (study cohort)", "Ethics / IRB approval"]),
    ("HIGH-VALUE", NAVY, ["Full T4/A100 training run", "Subject-level split", "Wire py-feat / OpenFace output"]),
    ("POLISH", GRAY, ["Diagnosticity weight tuning", "Ablation study", "AUC significance (DeLong / bootstrap)"]),
]
y = 5.9
for label, color, items in rows:
    box(ax, 0.6, y - 0.9, 2.2, 1.3, label, fc=color, ec=color, tc="white", bold=True, fs=10)
    txt = " · ".join(items)
    ax.text(3.2, y - 0.25, txt, ha="left", va="center", fontsize=8.6, color=INK)
    y -= 1.7
ax.text(6, 0.35, "Code is done — the rest is data + labels + ethics + one GPU run.",
        ha="center", fontsize=10, color=ACCENT, fontweight="bold")
save(fig, "05_research_roadmap.png")

print("ALL_DIAGRAMS_DONE")
