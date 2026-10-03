"""
Longitudinal validation experiment.

Deploys a real classifier trained on a reference batch, then steps it through
n_weeks of production-like data drawn from a shared synthetic "world" that
drifts according to a chosen trajectory (gradual ramp, sudden shock, or
recurring seasonal pattern). At each week we:

  1. Compute drift metrics between the (current) training reference and that
     week's unlabeled-in-spirit production batch.
  2. Ask the impact model to predict performance degradation from those
     drift metrics alone (no labels used).
  3. Because this is a simulation we *do* have that week's true labels, so we
     also measure the live model's true performance - this is exactly the
     ground truth a real deployment only gets after a label delay, and it's
     what lets us validate the impact model honestly (predicted vs true).
  4. Apply a retraining strategy (none / fixed periodic / adaptive) - when it
     fires, retrain the live model on the most recent window of
     (now-labelled) production data and reset the drift baseline to it.

run_longitudinal_experiment needs both a detailed week-by-week result for
whichever strategy the caller asked about *and* a side-by-side comparison of
all three strategies. Both are built from a single _simulate_strategy() call
per strategy (three calls total, run once each) rather than simulating the
requested strategy once for the detail view and then again inside the
comparison, which is what an earlier version of this file did.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from .synthetic_data import generate_longitudinal_world, SyntheticBatch
from .drift_metrics import analyze_drift, drift_report_to_feature_vector
from .impact_model import predict_degradation

STRATEGIES = ("none", "fixed", "adaptive")


def _encode(X: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    enc = pd.get_dummies(X, dummy_na=True)
    if columns is None:
        return enc
    return enc.reindex(columns=columns, fill_value=0)


def _eval_model(model, X, y) -> dict:
    pred = model.predict(X)
    out = {"accuracy": accuracy_score(y, pred), "f1": f1_score(y, pred, zero_division=0)}
    try:
        out["roc_auc"] = roc_auc_score(y, model.predict_proba(X)[:, 1])
    except Exception:
        out["roc_auc"] = 0.5
    return out


def _train_live_model(X: pd.DataFrame, y: np.ndarray, seed: int):
    cols = sorted(_encode(X).columns)
    model = RandomForestClassifier(n_estimators=150, max_depth=6, random_state=seed)
    model.fit(_encode(X, cols), y)
    return model, cols


def _drift_pattern_magnitudes(pattern: str, n_steps: int, peak_magnitude: float) -> list[float]:
    t = np.arange(n_steps)
    if pattern == "sudden":
        shock_at = n_steps // 3
        return [0.02 if i < shock_at else peak_magnitude for i in t]
    if pattern == "recurring":
        period = max(4, n_steps // 4)
        return list(0.02 + (peak_magnitude - 0.02) * (0.5 + 0.5 * np.sin(2 * np.pi * t / period)))
    # gradual (default): linear ramp from ~0 to peak_magnitude
    return list(np.linspace(0.02, peak_magnitude, n_steps))


@dataclass
class LongitudinalResult:
    weeks: list[dict]
    summary: dict
    strategy_comparison: dict


def _simulate_strategy(
    reference: SyntheticBatch,
    weekly_batches: list[SyntheticBatch],
    metric: str,
    strategy: str,
    degradation_threshold: float,
    fixed_interval_weeks: int,
    seed: int,
) -> dict:
    """Runs one full deployment under one retraining strategy, once. Returns
    the per-week series plus bookkeeping needed for both the detailed view
    and the cross-strategy comparison table."""
    live_ref_X, live_ref_y = reference.X.copy(), reference.y.copy()
    model, model_cols = _train_live_model(live_ref_X, live_ref_y, seed)
    baseline_scores = _eval_model(model, _encode(live_ref_X, model_cols), live_ref_y)

    weeks_out = []
    n_retrains = 0
    last_retrain_week = 0

    for t, batch in enumerate(weekly_batches):
        Xenc = _encode(batch.X, model_cols)
        true_scores = _eval_model(model, Xenc, batch.y)
        true_degradation = max(0.0, baseline_scores[metric] - true_scores[metric])

        report = analyze_drift(live_ref_X, batch.X)
        pred = predict_degradation(drift_report_to_feature_vector(report), metric=metric)
        predicted_degradation = max(0.0, pred["predicted_degradation"])

        weeks_since_retrain = t - last_retrain_week
        should_retrain = False
        if strategy == "fixed" and weeks_since_retrain >= fixed_interval_weeks:
            should_retrain = True
        elif strategy == "adaptive" and predicted_degradation >= degradation_threshold:
            should_retrain = True
        # "none" never retrains

        if should_retrain:
            live_ref_X, live_ref_y = batch.X.copy(), batch.y.copy()
            model, model_cols = _train_live_model(live_ref_X, live_ref_y, seed)
            baseline_scores = _eval_model(model, _encode(live_ref_X, model_cols), live_ref_y)
            n_retrains += 1
            last_retrain_week = t

        weeks_out.append({
            "week": t,
            "drift_score": round(report.overall_drift_score, 4),
            "c2st_auc": report.c2st_auc,
            "predicted_degradation": round(predicted_degradation, 4),
            "true_degradation": round(true_degradation, 4),
            "true_performance": round(true_scores[metric], 4),
            "retrained": should_retrain,
        })

    return {"weeks": weeks_out, "baseline_performance": baseline_scores[metric], "n_retrains": n_retrains}


def run_longitudinal_experiment(
    n_weeks: int = 26,
    drift_pattern: str = "gradual",
    peak_magnitude: float = 0.7,
    drift_type: str = "covariate",
    retrain_strategy: str = "adaptive",
    fixed_interval_weeks: int = 8,
    degradation_threshold: float = 0.15,
    metric: str = "accuracy",
    seed: int = 123,
) -> LongitudinalResult:
    if retrain_strategy not in STRATEGIES:
        raise ValueError(f"retrain_strategy must be one of {STRATEGIES}")

    magnitudes = _drift_pattern_magnitudes(drift_pattern, n_weeks, peak_magnitude)
    reference, weekly_batches = generate_longitudinal_world(
        seed=seed, n_reference=1800, n_per_step=350, n_steps=n_weeks,
        magnitudes=magnitudes, drift_type=drift_type,
    )

    # Simulate every strategy exactly once. The featured (requested) strategy's
    # result drives the detailed week-by-week view below; all three feed the
    # comparison table. This avoids re-simulating the featured strategy twice.
    all_results = {
        strategy: _simulate_strategy(
            reference, weekly_batches, metric, strategy,
            degradation_threshold, fixed_interval_weeks, seed,
        )
        for strategy in STRATEGIES
    }

    featured = all_results[retrain_strategy]
    weeks_out = featured["weeks"]

    pred_arr = np.array([w["predicted_degradation"] for w in weeks_out])
    true_arr = np.array([w["true_degradation"] for w in weeks_out])
    mae = float(np.mean(np.abs(pred_arr - true_arr)))
    if np.std(pred_arr) > 1e-9 and np.std(true_arr) > 1e-9:
        corr = float(np.corrcoef(pred_arr, true_arr)[0, 1])
    else:
        corr = None

    summary = {
        "mae_predicted_vs_true": round(mae, 4),
        "correlation_predicted_vs_true": round(corr, 4) if corr is not None else None,
        "n_retrains": featured["n_retrains"],
        "mean_true_performance": round(float(np.mean([w["true_performance"] for w in weeks_out])), 4),
        "final_true_performance": weeks_out[-1]["true_performance"],
        "baseline_performance": round(featured["baseline_performance"], 4),
        "drift_type": drift_type,
        "drift_pattern": drift_pattern,
        "retrain_strategy": retrain_strategy,
        "note": (
            "Predicted-vs-true correlation reflects a real, structural limit: pure concept "
            "drift (label relationship changes without the input distribution changing) is "
            "not observable from input-distribution drift metrics alone. Expect strong "
            "correlation for 'covariate'/'mixed' drift_type and weak correlation for 'concept'."
        ),
    }

    strategy_comparison = {
        name: {
            "n_retrains": r["n_retrains"],
            "mean_true_performance": round(float(np.mean([w["true_performance"] for w in r["weeks"]])), 4),
            "min_true_performance": round(float(np.min([w["true_performance"] for w in r["weeks"]])), 4),
            "final_true_performance": round(r["weeks"][-1]["true_performance"], 4),
        }
        for name, r in all_results.items()
    }

    return LongitudinalResult(weeks=weeks_out, summary=summary, strategy_comparison=strategy_comparison)
