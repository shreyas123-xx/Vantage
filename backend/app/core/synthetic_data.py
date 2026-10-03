"""
Generates production-like labelled tabular datasets with controllable,
parametrized drift so the rest of the system (drift metrics, impact model,
longitudinal simulation) can be validated against a known ground truth.

Design note: reference and "current" batches must be independent draws from
the *same* underlying generative world when magnitude=0, or the synthetic
benchmark itself would manufacture fake drift. sklearn's make_classification
uses its random_state to pick the whole geometry (cluster centroids, feature
rotation) as well as the per-row noise, so calling it twice with two
different seeds produces two genuinely different distributions, not two
samples of the same one. To avoid that, we draw one big pool from a single
make_classification call and slice rows into reference / current / (for the
longitudinal simulation) successive time steps, then apply drift transforms
only to the non-reference slices.

Two drift mechanisms are modelled, covering most real-world production drift:
  * covariate shift  - P(X) moves (sensor recalibration, seasonality, new
                        user segments) but P(y|X) is unchanged.
  * concept drift     - P(y|X) itself changes (fraud patterns evolve), here
                        by flipping a magnitude-proportional fraction of labels.

`magnitude` in [0, 1]: 0 = no injected drift (still has natural sampling
noise, as a real "quiet" production week would), 1 = severe drift.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.datasets import make_classification

NUMERIC_FEATURES = 6
CATEGORICAL_FEATURES = 2
CATEGORY_LEVELS = ["A", "B", "C", "D"]


@dataclass
class SyntheticBatch:
    X: pd.DataFrame
    y: np.ndarray


def _world_pool(seed: int, n_total: int, class_sep: float = 1.5):
    """One shared generative process. Slice rows out of this for reference /
    current / longitudinal time steps so they come from the same distribution."""
    return make_classification(
        n_samples=n_total,
        n_features=NUMERIC_FEATURES,
        n_informative=4,
        n_redundant=1,
        n_clusters_per_class=2,
        class_sep=class_sep,
        flip_y=0.02,
        random_state=seed,
    )


def _attach_categoricals(X: np.ndarray, y: np.ndarray, rng: np.random.Generator, cat_drift: float = 0.0) -> pd.DataFrame:
    df = pd.DataFrame(X, columns=[f"num_{i}" for i in range(X.shape[1])])
    for j in range(CATEGORICAL_FEATURES):
        base_probs = np.array([0.4, 0.3, 0.2, 0.1]) if j == 0 else np.array([0.25, 0.25, 0.25, 0.25])
        probs = base_probs.copy()
        if cat_drift > 0:
            skew = np.array([1, -1, 1, -1]) * cat_drift
            probs = np.clip(probs + skew * 0.3, 0.01, None)
            probs = probs / probs.sum()
        df[f"cat_{j}"] = rng.choice(CATEGORY_LEVELS, size=len(df), p=probs)
    return df


def _apply_drift_transform(X: np.ndarray, y: np.ndarray, magnitude: float, drift_type: str, rng: np.random.Generator):
    magnitude = float(np.clip(magnitude, 0.0, 1.0))
    X = X.copy()
    y = y.copy()
    if drift_type in ("covariate", "mixed") and magnitude > 0:
        shift = rng.normal(loc=1.0, scale=0.3, size=X.shape[1]) * magnitude * 1.8
        scale = 1.0 + magnitude * rng.uniform(0.3, 0.9, size=X.shape[1])
        X = X * scale + shift
    if drift_type in ("concept", "mixed") and magnitude > 0:
        flip_mask = rng.random(len(y)) < (0.35 * magnitude)
        y = np.where(flip_mask, 1 - y, y)
    return X, y


def generate_reference_batch(n_samples: int = 2000, seed: int = 42) -> SyntheticBatch:
    X, y = _world_pool(seed, n_samples)
    rng = np.random.default_rng(seed)
    df = _attach_categoricals(X, y, rng, cat_drift=0.0)
    return SyntheticBatch(X=df, y=y)


def generate_matched_pair(
    n_reference: int,
    n_current: int,
    seed: int,
    magnitude: float,
    drift_type: str = "covariate",
) -> tuple[SyntheticBatch, SyntheticBatch]:
    """The core generator: reference and current are independent slices of one
    shared world, so magnitude=0 yields two samples of the identical
    distribution (only natural sampling noise), and increasing magnitude
    smoothly increases genuine, measurable drift in the current slice only."""
    total = n_reference + n_current
    X, y = _world_pool(seed, total)
    X_ref, y_ref = X[:n_reference], y[:n_reference]
    X_cur, y_cur = X[n_reference:], y[n_reference:]

    drift_rng = np.random.default_rng(seed + 1)
    X_cur, y_cur = _apply_drift_transform(X_cur, y_cur, magnitude, drift_type, drift_rng)

    ref_df = _attach_categoricals(X_ref, y_ref, np.random.default_rng(seed + 2), cat_drift=0.0)
    cat_drift = magnitude if drift_type in ("covariate", "mixed") else 0.0
    cur_df = _attach_categoricals(X_cur, y_cur, np.random.default_rng(seed + 3), cat_drift=cat_drift)

    return SyntheticBatch(X=ref_df, y=y_ref), SyntheticBatch(X=cur_df, y=y_cur)


def generate_longitudinal_world(
    seed: int,
    n_reference: int,
    n_per_step: int,
    n_steps: int,
    magnitudes: list[float],
    drift_type: str = "covariate",
) -> tuple[SyntheticBatch, list[SyntheticBatch]]:
    """Reference batch plus n_steps successive production batches, all drawn
    from the same shared world, each with its own drift magnitude (so a
    trajectory like a gradual ramp or a sudden jump can be simulated)."""
    total = n_reference + n_per_step * n_steps
    X, y = _world_pool(seed, total)
    X_ref, y_ref = X[:n_reference], y[:n_reference]
    ref_df = _attach_categoricals(X_ref, y_ref, np.random.default_rng(seed + 2), cat_drift=0.0)
    reference = SyntheticBatch(X=ref_df, y=y_ref)

    steps: list[SyntheticBatch] = []
    idx = n_reference
    for t in range(n_steps):
        X_t, y_t = X[idx: idx + n_per_step], y[idx: idx + n_per_step]
        idx += n_per_step
        rng = np.random.default_rng(seed + 1000 + t)
        mag = magnitudes[t]
        X_t, y_t = _apply_drift_transform(X_t, y_t, mag, drift_type, rng)
        cat_drift = mag if drift_type in ("covariate", "mixed") else 0.0
        df = _attach_categoricals(X_t, y_t, rng, cat_drift=cat_drift)
        steps.append(SyntheticBatch(X=df, y=y_t))

    return reference, steps


def batch_to_records(batch: SyntheticBatch, n_preview: int | None = 25) -> list[dict]:
    df = batch.X.copy()
    df["label"] = batch.y
    if n_preview is not None:
        df = df.head(n_preview)
    return df.round(4).to_dict(orient="records")
