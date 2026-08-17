"""
explain.py — SHAP and Grad-CAM explainability for the paper.

SHAP   : which AU / region / head-pose / temporal features drive each
         PTSD-subtype prediction. Output is a Fathom-style horizontal bar chart.
Grad-CAM: overlay class-activation maps on face images to show which facial
         region the auxiliary image pathway attends to per emotion class.

Both run on Colab T4. SHAP uses a small background set to keep runtime low;
Grad-CAM uses the lightweight CNN backbone from model.py.
"""
from __future__ import annotations

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import FEATURE_NAMES, EMOTION_CLASSES, SUBTYPE_CLASSES

NAVY = "#1F3A5F"
GRAY = "#8A919C"
HIGHLIGHT = "#C75B39"
FACE = "#F7F6F3"


# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------
class _SHAPWrapper(nn.Module):
    """Wraps the multi-input model so SHAP sees one flat (N, 28) input and one
    output (subtype probabilities)."""

    def __init__(self, model, output_head="subtype"):
        super().__init__()
        self.model = model
        self.output_head = output_head

    def forward(self, x_flat):
        emo, sub = self.model.forward_flat(x_flat)
        logits = sub if self.output_head == "subtype" else emo
        return torch.softmax(logits, dim=1)


def shap_feature_importance(model, background_X: np.ndarray, sample_X: np.ndarray,
                            feature_names: list[str] | None = None,
                            output_head: str = "subtype",
                            class_names: list[str] | None = None,
                            out_dir: str = "/content/figures",
                            n_background: int = 100) -> dict:
    """Compute SHAP values and save one bar chart per class.

    Returns a dict: {class_name: path_to_bar_chart}.
    """
    import shap
    feature_names = feature_names or FEATURE_NAMES
    class_names = class_names or (SUBTYPE_CLASSES if output_head == "subtype"
                                  else EMOTION_CLASSES)
    os.makedirs(out_dir, exist_ok=True)

    model.eval()
    wrapper = _SHAPWrapper(model, output_head)

    bg = torch.tensor(background_X[:n_background], dtype=torch.float32)
    sample = torch.tensor(sample_X[:n_background], dtype=torch.float32)

    explainer = shap.GradientExplainer(wrapper, bg)
    shap_values = explainer.shap_values(sample)

    # shap_values shape: (n_classes, n_samples, n_features)
    paths = {}
    for ci, cls_name in enumerate(class_names):
        vals = shap_values[ci] if isinstance(shap_values, list) else shap_values
        mean_abs = np.abs(vals).mean(axis=0)
        order = np.argsort(mean_abs)[::-1][:15]  # top-15 features
        top_names = [feature_names[i] for i in order]
        top_vals = mean_abs[order]

        fig, ax = plt.subplots(figsize=(6.4, 5.0))
        y = np.arange(len(top_names))
        ax.barh(y, top_vals, color=NAVY, height=0.6)
        ax.set_yticks(y)
        ax.set_yticklabels(top_names, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title(f"{cls_name} — feature importance", fontsize=12,
                     fontweight="bold", color="#222")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x", color="#ECEFF3", linewidth=0.6)
        fig.tight_layout()
        path = os.path.join(out_dir, f"shap_{cls_name.replace(' ', '_').replace('/', '_')}.png")
        fig.savefig(path, dpi=300)
        plt.close(fig)
        paths[cls_name] = path

    return paths


# ---------------------------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------------------------
def grad_cam(backbone, image_tensor, target_class: int, device="cpu"):
    """Compute a Grad-CAM heatmap (numpy, HxW) for `target_class`."""
    backbone.eval()
    image_tensor = image_tensor.to(device).unsqueeze(0)
    image_tensor.requires_grad_(True)

    out = backbone(image_tensor)
    score = out[0, target_class]
    backbone.zero_grad()
    score.backward()

    grads = backbone.gradients  # (1, C, H', W')
    acts = backbone.activations  # (1, C, H', W')
    weights = grads.mean(dim=(2, 3), keepdim=True)  # global-average-pool grad
    cam = (weights * acts).sum(dim=1, keepdim=True)
    cam = torch.relu(cam)
    cam = torch.nn.functional.interpolate(cam, size=image_tensor.shape[-2:],
                                          mode="bilinear", align_corners=False)
    cam = cam.squeeze().detach().cpu().numpy()
    if cam.max() > 0:
        cam = cam / cam.max()
    return cam


def overlay_gradcam(image_np: np.ndarray, cam: np.ndarray, alpha: float = 0.45):
    """Overlay a heatmap on an (H, W, 3) uint8 image using matplotlib."""
    import matplotlib.cm as cm
    if image_np.dtype != np.uint8:
        image_np = (image_np * 255).astype(np.uint8)
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.imshow(image_np)
    ax.imshow(cam, cmap=cm.jet, alpha=alpha)
    ax.axis("off")
    fig.tight_layout()
    return fig


def save_gradcam_grid(backbone, images: list[np.ndarray], class_names: list[str],
                      out_path: str, device="cpu", target_per_image: list[int] | None = None):
    """Render a grid of face images with Grad-CAM overlays per class."""
    n = len(images)
    cols = min(n, 4)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.0, rows * 3.0))
    if rows * cols == 1:
        axes = np.array([[axes]])
    axes = np.atleast_2d(axes)

    for i, img in enumerate(images):
        r, c = divmod(i, cols)
        ax = axes[r][c]
        tc = target_per_image[i] if target_per_image else i % len(class_names)
        img_t = torch.tensor(img.transpose(2, 0, 1), dtype=torch.float32) / 255.0
        cam = grad_cam(backbone, img_t, tc, device)
        ax.imshow(img)
        ax.imshow(cam, cmap="jet", alpha=0.45)
        ax.set_title(class_names[tc], fontsize=9)
        ax.axis("off")

    for j in range(n, rows * cols):
        r, c = divmod(j, cols)
        axes[r][c].axis("off")

    fig.suptitle("Grad-CAM: attended facial regions per emotion class",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path
