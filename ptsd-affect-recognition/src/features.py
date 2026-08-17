"""
features.py — Frame-level feature extraction.

Produces, per face frame, a structured DataFrame with:
  1. Action Unit (AU) intensities            (17 canonical AUs)
  2. Region-weighted facial scores           (eye, mouth, upper_face, lower_face)
  3. Head-pose angles                        (yaw, pitch, roll)
  4. Temporal sequence features              (dwell time, transition rate,
                                               entropy, AU variability)

The `build_feature_frame` and `add_temporal_features` functions are pure
NumPy/Pandas and are the part of the pipeline covered by the local CPU smoke
test. The AU + pose values are assumed to come from OpenFace / py-feat; a
synthetic generator is provided for end-to-end demonstration and the smoke test.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    AU_NAMES, REGION_AU_MAP, REGION_NAMES, POSE_NAMES,
    TEMPORAL_NAMES, FEATURE_NAMES, REGION_DIAGNOSTIC_WEIGHTS,
    TEMPORAL_WINDOW, AU_ACTIVATION_THRESHOLD, N_FEATURES,
)


# ---------------------------------------------------------------------------
# 1. Action Unit intensities
# ---------------------------------------------------------------------------
def validate_au_matrix(au: np.ndarray) -> np.ndarray:
    """Normalise and clip an (n_frames, N_AU) AU-intensity matrix."""
    au = np.asarray(au, dtype=np.float32)
    if au.ndim == 1:
        au = au[None, :]
    assert au.shape[1] == len(AU_NAMES), f"Expected {len(AU_NAMES)} AUs, got {au.shape[1]}"
    au = np.clip(au, 0.0, 5.0)  # AU intensity is conventionally 0-5
    return au


# ---------------------------------------------------------------------------
# 2. Region-weighted facial scores
# ---------------------------------------------------------------------------
def region_scores(au: np.ndarray) -> np.ndarray:
    """Region-weighted scores for eye/mouth/upper_face/lower_face.

    Each region score is a diagnosticity-weighted mean of its member AUs:
        score = sum(w_i * au_i) / sum(w_i)
    Returns (n_frames, N_REGION).
    """
    au = validate_au_matrix(au)
    out = np.zeros((au.shape[0], len(REGION_NAMES)), dtype=np.float32)
    for j, region in enumerate(REGION_NAMES):
        members = REGION_AU_MAP[region]
        idx = [AU_NAMES.index(a) for a in members]
        w = np.array([REGION_DIAGNOSTIC_WEIGHTS[a] for a in members], dtype=np.float32)
        out[:, j] = (au[:, idx] * w).sum(axis=1) / w.sum()
    return out


# ---------------------------------------------------------------------------
# 3. Head-pose angles
# ---------------------------------------------------------------------------
def validate_pose(pose: np.ndarray) -> np.ndarray:
    """Normalise an (n_frames, 3) pose matrix of [yaw, pitch, roll] in degrees."""
    pose = np.asarray(pose, dtype=np.float32)
    if pose.ndim == 1:
        pose = pose[None, :]
    assert pose.shape[1] == len(POSE_NAMES), f"Expected {len(POSE_NAMES)} pose angles"
    return pose


# ---------------------------------------------------------------------------
# 4. Temporal sequence features (causal sliding window)
# ---------------------------------------------------------------------------
def _binary_active(au: np.ndarray, thr: float = AU_ACTIVATION_THRESHOLD) -> np.ndarray:
    """Binarise AU activations above threshold."""
    return (au >= thr).astype(np.int8)


def dwell_time(au: np.ndarray, window: int = TEMPORAL_WINDOW,
               thr: float = AU_ACTIVATION_THRESHOLD) -> np.ndarray:
    """Fraction of frames in the trailing window where >=1 diagnostic AU is active.

    Captures how long a face 'holds' an expression (blunted affect -> low dwell).
    """
    act = _binary_active(au, thr)
    any_active = (act.sum(axis=1) > 0).astype(np.float32)
    return _rolling_mean(any_active, window)


def transition_rate(au: np.ndarray, window: int = TEMPORAL_WINDOW,
                    thr: float = AU_ACTIVATION_THRESHOLD) -> np.ndarray:
    """Mean per-frame count of AU on/off transitions in the trailing window.

    A transition is any AU changing active-state between consecutive frames.
    """
    act = _binary_active(au, thr)
    # per-frame number of AUs that flipped state
    flips = np.abs(np.diff(act, axis=0)).sum(axis=1).astype(np.float32)
    flips = np.concatenate([np.zeros(1, dtype=np.float32), flips])
    return _rolling_mean(flips, window)


def au_entropy(au: np.ndarray, window: int = TEMPORAL_WINDOW,
               thr: float = AU_ACTIVATION_THRESHOLD) -> np.ndarray:
    """Shannon entropy of the binarised AU activation vector (per frame),
    averaged over the trailing window. High entropy -> more complex / mixed
    facial behaviour; low entropy -> restricted, flat repertoire.
    """
    act = _binary_active(au, thr)
    # entropy over the N_AU binary channels per frame
    p_active = act.mean(axis=1)
    p_active = np.clip(p_active, 1e-9, 1.0 - 1e-9)
    ent = -(p_active * np.log2(p_active) + (1 - p_active) * np.log2(1 - p_active))
    return _rolling_mean(ent.astype(np.float32), window)


def au_variability(au: np.ndarray, window: int = TEMPORAL_WINDOW) -> np.ndarray:
    """Mean per-AU standard deviation over the trailing window."""
    out = np.zeros(au.shape[0], dtype=np.float32)
    for t in range(au.shape[0]):
        lo = max(0, t - window + 1)
        out[t] = au[lo:t + 1].std(axis=0).mean()
    return out


def _rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    """Causal rolling mean (uses only past + current frames)."""
    x = np.asarray(x, dtype=np.float32)
    out = np.empty_like(x)
    running = 0.0
    for t in range(x.shape[0]):
        running += x[t]
        if t >= window:
            running -= x[t - window]
        out[t] = running / min(t + 1, window)
    return out


def temporal_features(au: np.ndarray, window: int = TEMPORAL_WINDOW) -> np.ndarray:
    """Stack the four temporal features into an (n_frames, N_TEMPORAL) matrix."""
    return np.column_stack([
        dwell_time(au, window),
        transition_rate(au, window),
        au_entropy(au, window),
        au_variability(au, window),
    ]).astype(np.float32)


# ---------------------------------------------------------------------------
# Full frame-level DataFrame
# ---------------------------------------------------------------------------
def build_feature_frame(au: np.ndarray, pose: np.ndarray,
                        sequence_ids: np.ndarray | list[str] | None = None,
                        frame_idx: np.ndarray | None = None,
                        emotion_labels: list[str] | np.ndarray | None = None,
                        subtype_labels: list[str] | np.ndarray | None = None,
                        window: int = TEMPORAL_WINDOW) -> pd.DataFrame:
    """Assemble one row per frame with all AU / region / pose / temporal columns.

    Temporal features are computed per-sequence (grouped by `sequence_ids`) so
    that windows never bleed across sequence boundaries.
    """
    au = validate_au_matrix(au)
    pose = validate_pose(pose)
    n = au.shape[0]
    assert pose.shape[0] == n, "AU and pose must have the same number of frames"

    if sequence_ids is None:
        sequence_ids = np.zeros(n, dtype=int)
    sequence_ids = np.asarray(sequence_ids)

    if frame_idx is None:
        frame_idx = np.arange(n)
    frame_idx = np.asarray(frame_idx)

    regions = region_scores(au)

    # temporal features per sequence
    temporal = np.zeros((n, len(TEMPORAL_NAMES)), dtype=np.float32)
    for sid in np.unique(sequence_ids):
        mask = sequence_ids == sid
        temporal[mask] = temporal_features(au[mask], window)

    cols = {}
    for j, a in enumerate(AU_NAMES):
        cols[f"au_{a}"] = au[:, j]
    for j, r in enumerate(REGION_NAMES):
        cols[f"region_{r}"] = regions[:, j]
    for j, p in enumerate(POSE_NAMES):
        cols[f"pose_{p}"] = pose[:, j]
    for j, t in enumerate(TEMPORAL_NAMES):
        cols[f"temp_{t}"] = temporal[:, j]

    df = pd.DataFrame(cols, columns=FEATURE_NAMES)
    df.insert(0, "frame_idx", frame_idx)
    df.insert(0, "sequence_id", sequence_ids)
    if emotion_labels is not None:
        df["emotion"] = list(emotion_labels)
    if subtype_labels is not None:
        df["subtype"] = list(subtype_labels)
    return df


# ---------------------------------------------------------------------------
# Synthetic demo generator (for notebook demo + local smoke test)
# ---------------------------------------------------------------------------
def generate_synthetic_dataset(n_sequences: int = 6, frames_per_seq: int = 60,
                               n_au: int = None, seed: int = 0):
    """Create plausible synthetic AU + pose + labels for end-to-end testing.

    Returns (df, flat_X, y_emotion, y_subtype). Real runs replace this with
    OpenFace/py-feat output. Emphasises that the pipeline can be exercised end
    to end before any real (possibly request-gated) data arrives.
    """
    from config import EMOTION_CLASSES, SUBTYPE_CLASSES
    rng = np.random.default_rng(seed)
    n_au = n_au or len(AU_NAMES)
    aus, poses, seq_ids, frames, emo, sub = [], [], [], [], [], []

    # emotion -> characteristic AU 'activation pattern' (mean intensity)
    emo_au_prior = {
        "Fear":     {"AU01": 2.5, "AU02": 2.5, "AU05": 3.0, "AU20": 2.0, "AU26": 2.5},
        "Anger":    {"AU04": 3.0, "AU05": 2.0, "AU07": 2.5, "AU23": 2.0},
        "Sadness":  {"AU01": 2.5, "AU04": 2.0, "AU15": 2.5, "AU17": 2.0},
        "Neutral":  {},
        "Surprise": {"AU01": 2.5, "AU02": 3.0, "AU05": 3.0, "AU25": 2.0, "AU26": 3.0},
        "Flat/Blunted Affect": {},
    }

    for s in range(n_sequences):
        e = EMOTION_CLASSES[s % len(EMOTION_CLASSES)]
        prior = emo_au_prior[e]
        seq_au = np.zeros((frames_per_seq, n_au), dtype=np.float32)
        for j, a in enumerate(AU_NAMES[:n_au]):
            base = prior.get(a, rng.uniform(0.0, 0.4))
            seq_au[:, j] = np.clip(rng.normal(base, 0.5, frames_per_seq), 0, 5)
        # pose: small smooth wander
        t = np.arange(frames_per_seq)
        pose = np.column_stack([
            15 * np.sin(2 * np.pi * t / 60) + rng.normal(0, 2, frames_per_seq),
            8 * np.cos(2 * np.pi * t / 45) + rng.normal(0, 2, frames_per_seq),
            6 * np.sin(2 * np.pi * t / 30) + rng.normal(0, 1.5, frames_per_seq),
        ])
        aus.append(seq_au)
        poses.append(pose)
        seq_ids += [s] * frames_per_seq
        frames += list(range(frames_per_seq))
        emo += [e] * frames_per_seq
        sub += [SUBTYPE_CLASSES[s % len(SUBTYPE_CLASSES)]] * frames_per_seq

    au = np.concatenate(aus)
    pose = np.concatenate(poses)
    df = build_feature_frame(au, pose, sequence_ids=seq_ids, frame_idx=frames,
                             emotion_labels=emo, subtype_labels=sub)
    X = df[FEATURE_NAMES].to_numpy(dtype=np.float32)
    return df, X, emo, sub


if __name__ == "__main__":
    df, X, y_emo, y_sub = generate_synthetic_dataset(seed=0)
    print("Feature frame shape:", df.shape)
    print("Feature vector dim :", X.shape[1], "(expected", N_FEATURES, ")")
    print(df[["sequence_id", "frame_idx", "au_AU04", "region_mouth",
              "pose_yaw", "temp_entropy", "emotion", "subtype"]].head(8).to_string(index=False))
