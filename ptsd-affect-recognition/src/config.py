"""
config.py — Central configuration for the PTSD facial-affect recognition pipeline.

Everything here is imported by the other modules and by the integrated Colab
notebook so that constants (AU list, class labels, feature dimensions, seed)
live in exactly one place.
"""
import os
import random

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42


def seed_everything(seed: int = SEED) -> None:
    """Fix Python, NumPy and PyTorch seeds for reproducible runs."""
    import numpy as np
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Label sets
# ---------------------------------------------------------------------------
# Six emotion / affect states (primary classification head)
EMOTION_CLASSES = ["Fear", "Anger", "Sadness", "Neutral", "Surprise", "Flat/Blunted Affect"]
N_EMOTION = len(EMOTION_CLASSES)

# Secondary head: PTSD subtype / group
SUBTYPE_CLASSES = ["Classic PTSD", "D-PTSD", "Control"]
N_SUBTYPE = len(SUBTYPE_CLASSES)

# ---------------------------------------------------------------------------
# Action Unit schema (canonical 17-AU set produced by OpenFace / py-feat)
# ---------------------------------------------------------------------------
AU_NAMES = [
    "AU01", "AU02", "AU04", "AU05", "AU06", "AU07", "AU09", "AU10",
    "AU12", "AU14", "AU15", "AU17", "AU20", "AU23", "AU25", "AU26", "AU45",
]
N_AU = len(AU_NAMES)

# Region definitions (indices into AU_NAMES). Four regions as requested:
# eyes / mouth (fine diagnosticity) and upper-face / lower-face (broad).
REGION_AU_MAP = {
    "eye":        ["AU01", "AU02", "AU04", "AU05", "AU06", "AU07", "AU45"],
    "mouth":      ["AU09", "AU10", "AU12", "AU14", "AU15", "AU17", "AU20", "AU23", "AU25", "AU26"],
    "upper_face": ["AU01", "AU02", "AU04", "AU05", "AU06", "AU07"],
    "lower_face": ["AU09", "AU10", "AU12", "AU14", "AU15", "AU17", "AU20", "AU23", "AU25", "AU26"],
}
REGION_NAMES = list(REGION_AU_MAP.keys())          # eye, mouth, upper_face, lower_face
N_REGION = len(REGION_NAMES)

# Diagnosticity weights for region scores. These are *literature-informed
# priors* for PTSD / blunted-affect work and are fully tunable: brow lowering
# (AU04), lid raising (AU05), cheek raising (AU06) and lip-corner pulling
# (AU12) carry the most signal for reduced / dysregulated expressivity.
# Defaults are intentionally transparent and overridable in the notebook.
REGION_DIAGNOSTIC_WEIGHTS = {
    "AU01": 1.0, "AU02": 1.0, "AU04": 1.5, "AU05": 1.5, "AU06": 1.5,
    "AU07": 1.2, "AU09": 1.0, "AU10": 1.0, "AU12": 1.5, "AU14": 1.0,
    "AU15": 1.0, "AU17": 1.0, "AU20": 1.1, "AU23": 1.0, "AU25": 1.0,
    "AU26": 1.0, "AU45": 1.0,
}

# ---------------------------------------------------------------------------
# Head pose + temporal feature names
# ---------------------------------------------------------------------------
POSE_NAMES = ["yaw", "pitch", "roll"]
N_POSE = len(POSE_NAMES)

TEMPORAL_NAMES = ["dwell_time", "transition_rate", "entropy", "au_variability"]
N_TEMPORAL = len(TEMPORAL_NAMES)

# Full flat feature vector (28 dims) = AU(17) + region(4) + pose(3) + temporal(4)
FEATURE_NAMES = (
    [f"au_{a}" for a in AU_NAMES]
    + [f"region_{r}" for r in REGION_NAMES]
    + [f"pose_{p}" for p in POSE_NAMES]
    + [f"temp_{t}" for t in TEMPORAL_NAMES]
)
N_FEATURES = N_AU + N_REGION + N_POSE + N_TEMPORAL

# Branch dimensions fed as *separate* model inputs
INPUT_DIMS = {
    "au": N_AU,
    "region": N_REGION,
    "pose": N_POSE,
    "temporal": N_TEMPORAL,
}

# ---------------------------------------------------------------------------
# Paths (Colab defaults; override at runtime)
# ---------------------------------------------------------------------------
DATA_ROOT = "/content/data"
DRIVE_CHECKPOINT_DIR = "/content/drive/MyDrive/ptsd_affect/checkpoints"
LOCAL_CHECKPOINT_DIR = "/content/checkpoints"
FIG_DIR = "/content/figures"

# Temporal sliding window (frames) for sequence-level features
TEMPORAL_WINDOW = 15

# AU activation threshold for binarised dwell / entropy / transition features
AU_ACTIVATION_THRESHOLD = 1.0

# Emotion classes that are "confusion pairs" to highlight in the paper figure
CONFUSION_PAIRS = [("Fear", "Surprise"), ("Anger", "Disgust")]
