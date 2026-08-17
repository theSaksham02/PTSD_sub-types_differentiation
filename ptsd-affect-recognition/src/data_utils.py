"""
data_utils.py — Colab dataset acquisition, structuring and integrity checks.

Strategies: Kaggle API / kagglehub, gdown, and direct URL. Includes helpers to
unzip, organise into train/val/test folders, and display a sample grid of face
images with their labels (and AU labels where available) as an integrity check.

NOTE: DISFA and BP4D are the gold-standard AU datasets but are *request-gated*
(no public auto-download). This module documents the request workflow and only
auto-downloads the freely accessible top-3: CK+, AffectNet, RAF-DB.
"""
from __future__ import annotations

import os
import zipfile
import shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Low-level download helpers
# ---------------------------------------------------------------------------
def download_kaggle(dataset_slug: str, out_dir: str, use_kagglehub: bool = True) -> str:
    """Download a Kaggle dataset. Prefers kagglehub (no manual key upload)."""
    os.makedirs(out_dir, exist_ok=True)
    if use_kagglehub:
        import kagglehub
        path = kagglehub.dataset_download(dataset_slug)
        return path
    else:
        # requires ~/.kaggle/kaggle.json
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(dataset_slug, path=out_dir, unzip=True)
        return out_dir


def download_gdown(file_id: str, out_path: str) -> str:
    """Download a file from Google Drive by file id."""
    import gdown
    gdown.download(id=file_id, output=out_path, quiet=False)
    return out_path


def download_url(url: str, out_path: str) -> str:
    """Download a file from a direct URL with a progress bar."""
    import urllib.request
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    urllib.request.urlretrieve(url, out_path)
    return out_path


def unzip_all(src_zip: str, dest_dir: str) -> str:
    """Extract a zip archive into dest_dir."""
    os.makedirs(dest_dir, exist_ok=True)
    with zipfile.ZipFile(src_zip, "r") as z:
        z.extractall(dest_dir)
    print(f"[unzip] {src_zip} -> {dest_dir} ({len(z.namelist())} entries)")
    return dest_dir


# ---------------------------------------------------------------------------
# Train / val / test split
# ---------------------------------------------------------------------------
def split_image_folders(class_root: str, out_root: str,
                        val_ratio: float = 0.15, test_ratio: float = 0.15,
                        seed: int = 42, copy: bool = True):
    """Reorganise class subfolders into train/val/test splits by subject-agnostic
    random sampling (subject-level split must be added by the study for clinical
    datasets to avoid leakage)."""
    from sklearn.model_selection import train_test_split
    os.makedirs(out_root, exist_ok=True)
    classes = sorted(d for d in os.listdir(class_root)
                     if os.path.isdir(os.path.join(class_root, d)))

    for split in ("train", "val", "test"):
        for c in classes:
            os.makedirs(os.path.join(out_root, split, c), exist_ok=True)

    counts = {"train": 0, "val": 0, "test": 0}
    for c in classes:
        src = os.path.join(class_root, c)
        files = sorted(os.listdir(src))
        tr, rest = train_test_split(files, test_size=val_ratio + test_ratio,
                                    random_state=seed)
        va, te = train_test_split(rest, test_size=test_ratio / (val_ratio + test_ratio),
                                  random_state=seed)
        for f, split in [(f, "train") for f in tr] + [(f, "val") for f in va] + [(f, "test") for f in te]:
            op = shutil.copy2 if copy else shutil.move
            op(os.path.join(src, f), os.path.join(out_root, split, c, f))
            counts[split] += 1
    print("[split] train/val/test counts:", counts)
    return out_root


# ---------------------------------------------------------------------------
# Integrity check: display sample images + labels
# ---------------------------------------------------------------------------
def display_sample_grid(image_paths: list[str], labels: list[str],
                        au_labels: list[str] | None = None,
                        title: str = "Dataset integrity check",
                        n_rows: int = 2, n_cols: int = 4):
    """Show a grid of face images with their emotion (and optional AU) labels."""
    from PIL import Image
    n = min(len(image_paths), n_rows * n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.6, n_rows * 2.8))
    axes = np.atleast_2d(axes)
    for i in range(n):
        r, c = divmod(i, n_cols)
        img = np.asarray(Image.open(image_paths[i]).convert("RGB"))
        axes[r][c].imshow(img)
        cap = labels[i]
        if au_labels is not None and au_labels[i]:
            cap += f"\nAU: {au_labels[i]}"
        axes[r][c].set_title(cap, fontsize=8)
        axes[r][c].axis("off")
    for j in range(n, n_rows * n_cols):
        r, c = divmod(j, n_cols)
        axes[r][c].axis("off")
    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    plt.show()
    return fig


# ---------------------------------------------------------------------------
# Request-form workflow (DISFA / BP4D)
# ---------------------------------------------------------------------------
REQUEST_GATED = {
    "DISFA": {
        "url": "https://mohammadmahoor.com/pages/databases/disfa/",
        "note": "Email request to the University of Denver team; free for academic "
                "use under a signed agreement. 27 subjects, 12 AUs @ 0-5 intensity, "
                "spontaneous (20 fps video). Gold standard for spontaneous AU intensity.",
    },
    "BP4D": {
        "url": "https://binghamton.technologypublisher.com/tech/BP4D",
        "note": "Request via Binghamton University technology transfer; free academic. "
                "41 subjects (BP4D+) / 140 (BP4D+), 2D+3D video, FACS-coded AUs with "
                "intensity. Best AU + video for dynamic affect.",
    },
}


def print_request_instructions(dataset: str) -> None:
    """Print the manual request workflow for a request-gated dataset."""
    info = REQUEST_GATED[dataset]
    print(f"\n=== {dataset} (request-gated, not auto-downloadable) ===\n"
          f"URL : {info['url']}\n"
          f"How : {info['note']}\n"
          f"Once received, place the data under /content/data/{dataset.lower()}/ "
          f"and run the feature pipeline as usual.\n")
