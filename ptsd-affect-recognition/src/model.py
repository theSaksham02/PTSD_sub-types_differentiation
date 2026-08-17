"""
model.py — Multi-input late-fusion classifier (PyTorch).

Accepts FOUR separate inputs:
  1. AU intensity vector        (17)
  2. Region-weight vector       (4)
  3. Head-pose vector           (3)
  4. Temporal feature vector    (4)

Each branch is a small MLP (Linear -> BatchNorm -> ReLU -> Dropout). Branch
embeddings are concatenated (late fusion) and passed through a shared MLP,
then two heads predict:
  - 6 emotion/affect states: Fear, Anger, Sadness, Neutral, Surprise, Flat/Blunted
  - 3 PTSD subtypes: Classic PTSD, D-PTSD, Control

Dropout + BatchNorm are used throughout. An optional `use_gradient_checkpointing`
flag wraps the shared MLP with torch.utils.checkpoint to reduce VRAM.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from config import (
    N_AU, N_REGION, N_POSE, N_TEMPORAL, N_EMOTION, N_SUBTYPE,
    EMOTION_CLASSES, SUBTYPE_CLASSES, FEATURE_NAMES,
)


class BranchMLP(nn.Module):
    """Single-input branch encoder."""

    def __init__(self, in_dim: int, hidden: list[int], dropout: float = 0.3):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            prev = h
        self.net = nn.Sequential(*layers)
        self.out_dim = prev

    def forward(self, x):
        return self.net(x)


class FusionMLP(nn.Module):
    """Shared late-fusion MLP with optional gradient checkpointing."""

    def __init__(self, in_dim: int, hidden: list[int], dropout: float = 0.35,
                 use_checkpoint: bool = False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList()
        prev = in_dim
        for h in hidden:
            self.blocks.append(nn.Sequential(
                nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ))
            prev = h
        self.out_dim = prev

    def forward(self, x):
        for block in self.blocks:
            if self.use_checkpoint and self.training and x.requires_grad:
                x = torch.utils.checkpoint.checkpoint(block, x,
                                                      use_reentrant=False)
            else:
                x = block(x)
        return x


class MultiInputPTSDAffectModel(nn.Module):
    def __init__(
        self,
        au_dim: int = N_AU,
        region_dim: int = N_REGION,
        pose_dim: int = N_POSE,
        temporal_dim: int = N_TEMPORAL,
        n_emotion: int = N_EMOTION,
        n_subtype: int = N_SUBTYPE,
        branch_hidden: list[int] | None = None,
        fusion_hidden: list[int] | None = None,
        dropout: float = 0.35,
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()
        branch_hidden = branch_hidden or [32]
        fusion_hidden = fusion_hidden or [128, 64]

        self.au_branch = BranchMLP(au_dim, branch_hidden, dropout)
        self.region_branch = BranchMLP(region_dim, branch_hidden, dropout)
        self.pose_branch = BranchMLP(pose_dim, branch_hidden, dropout)
        self.temporal_branch = BranchMLP(temporal_dim, branch_hidden, dropout)

        fusion_in = (self.au_branch.out_dim + self.region_branch.out_dim
                     + self.pose_branch.out_dim + self.temporal_branch.out_dim)
        self.fusion = FusionMLP(fusion_in, fusion_hidden, dropout,
                                use_gradient_checkpointing)

        self.emotion_head = nn.Linear(self.fusion.out_dim, n_emotion)
        self.subtype_head = nn.Linear(self.fusion.out_dim, n_subtype)

        self.n_emotion = n_emotion
        self.n_subtype = n_subtype

    def forward(self, au, region, pose, temporal):
        a = self.au_branch(au)
        r = self.region_branch(region)
        p = self.pose_branch(pose)
        t = self.temporal_branch(temporal)
        fused = torch.cat([a, r, p, t], dim=1)
        z = self.fusion(fused)
        return self.emotion_head(z), self.subtype_head(z)

    def forward_flat(self, x_flat):
        """Convenience for SHAP: split a flat (N, 28) vector into the 4 branches."""
        au = x_flat[:, :N_AU]
        region = x_flat[:, N_AU:N_AU + N_REGION]
        pose = x_flat[:, N_AU + N_REGION:N_AU + N_REGION + N_POSE]
        temporal = x_flat[:, N_AU + N_REGION + N_POSE:]
        return self.forward(au, region, pose, temporal)

    def summary(self) -> str:
        """Return a human-readable parameter summary table."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        lines = [
            "=" * 62,
            "MultiInputPTSDAffectModel",
            "=" * 62,
            f"{'Input branch':<24}{'in_dim':>10}{'out_dim':>10}",
            f"{'AU':<24}{N_AU:>10}{self.au_branch.out_dim:>10}",
            f"{'Region':<24}{N_REGION:>10}{self.region_branch.out_dim:>10}",
            f"{'Head-pose':<24}{N_POSE:>10}{self.pose_branch.out_dim:>10}",
            f"{'Temporal':<24}{N_TEMPORAL:>10}{self.temporal_branch.out_dim:>10}",
            "-" * 62,
            f"Fusion input dim : {self.au_branch.out_dim + self.region_branch.out_dim + self.pose_branch.out_dim + self.temporal_branch.out_dim}",
            f"Fusion output dim: {self.fusion.out_dim}",
            f"Emotion head     : {self.n_emotion} classes {EMOTION_CLASSES}",
            f"Subtype head     : {self.n_subtype} classes {SUBTYPE_CLASSES}",
            "-" * 62,
            f"Total params     : {total:,}",
            f"Trainable params : {trainable:,}",
            "=" * 62,
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Lightweight CNN backbone for Grad-CAM (visualising attended facial regions)
# ---------------------------------------------------------------------------
class GradCAMBackbone(nn.Module):
    """Small CNN whose final conv feature map feeds Grad-CAM.

    Used only for the *visualisation* of which facial regions the auxiliary
    image pathway attends to. The primary model remains the feature-fusion MLP.
    """

    def __init__(self, in_channels: int = 3, n_emotion: int = N_EMOTION):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.BatchNorm2d(32),
            nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64),
            nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128),
            nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(128, n_emotion)
        self.gradients = None
        self.activations = None
        self._hook()

    def _hook(self):
        def _forward(module, inp, out):
            self.activations = out

        def _backward(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        self.features.register_forward_hook(_forward)
        self.features.register_full_backward_hook(_backward)

    def forward(self, x):
        f = self.features(x)
        pooled = self.pool(f).flatten(1)
        return self.classifier(pooled)


def build_model(**kwargs) -> MultiInputPTSDAffectModel:
    """Factory: build the multi-input model with sensible defaults."""
    return MultiInputPTSDAffectModel(**kwargs)


if __name__ == "__main__":
    from config import seed_everything
    seed_everything(0)
    model = build_model()
    print(model.summary())
    b = 8
    emo, sub = model(
        torch.randn(b, N_AU), torch.randn(b, N_REGION),
        torch.randn(b, N_POSE), torch.randn(b, N_TEMPORAL),
    )
    print("emotion logits:", tuple(emo.shape), "subtype logits:", tuple(sub.shape))
