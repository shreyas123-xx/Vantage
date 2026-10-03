"""
The impact model is the novel core of this project: instead of stopping at
"here is how much the data has drifted", it answers "here is how much your
model's performance will degrade because of it".

Because we don't have oracle access to every production model's true
performance drop for an arbitrary drift signature, we build the training set
ourselves by *simulation*: repeatedly (a) generating a reference/current
dataset pair with a randomly chosen drift mechanism and magnitude, (b)
training one of several production-like classifiers (logistic regression,
random forest, gradient boosting) on the reference data, (c) measuring its
*true* performance drop on the current data, and (d) computing the same
drift-metric feature vector the live system would compute. Fitting a
regressor from (drift feature vector) -> (true performance drop) across
hundreds of such scenarios gives a model that generalizes to new, unseen
drift signatures at inference time.
"""
from __future__ import annotations

import time
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from .synthetic_data import generate_matched_pair
from .drift_metrics import analyze_drift, drift_report_to_feature_vector

FEATURE_ORDER = [
    "mean_psi", "max_psi", "mean_js_divergence", "max_js_divergence",
    "frac_significant", "c2st_auc", "drift_fraction", "n_features",
    "overall_drift_score",
]
METRICS = ["accuracy", "f1", "roc_auc"]
DRIFT_TYPES = ["none", "covariate", "concept", "mixed"]
BASE_MODELS = ["logreg", "random_forest", "gradient_boosting"]

ARTIFACT_DIR = Path(__file__).resolve().parents[2] / "artifacts"


def _make_base_model(kind: str, seed: int):
    if kind == "logreg":
        return LogisticRegression(max_iter=500, random_state=seed)
    if kind == "random_forest":
        return RandomForestClassifier(n_estimators=120, max_depth=6, random_state=seed)
    return GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=seed)


def _encode(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    enc = pd.get_dummies(df, dummy_na=True)
    return enc.reindex(columns=columns, fill_value=0)


def _eval(model, X, y) -> dict:
    pred = model.predict(X)
    out = {"accuracy": accuracy_score(y, pred), "f1": f1_score(y, pred, zero_division=0)}
    try:
        proba = model.predict_proba(X)[:, 1]
        out["roc_auc"] = roc_auc_score(y, proba)
    except Exception:
        out["roc_auc"] = 0.5
    return out


def generate_training_dataset(n_scenarios: int = 320, seed: int = 7, start_index: int = 0, verbose: bool = False) -> pd.DataFrame:
    """Runs the full simulate-train-evaluate-measure-drift loop many times.
    Returns one row per scenario: drift feature vector + true performance drop
    per metric, plus bookkeeping columns (not used as model inputs).

    start_index offsets the drift_type/base_model cycling only, so this can be
    called repeatedly in chunks (e.g. from a resumable CLI) with results
    concatenated afterwards, without every chunk being biased toward the same
    drift type or base model.
    """
    rng = np.random.default_rng(seed)
    rows = []
    t0 = time.time()

    for local_i in range(n_scenarios):
        i = start_index + local_i
        scenario_seed = int(rng.integers(0, 1_000_000))
        drift_type = DRIFT_TYPES[i % len(DRIFT_TYPES)]
        # bias sampling towards a good spread including near-zero drift so the
        # model learns what "no meaningful degradation" looks like too.
        magnitude = 0.0 if drift_type == "none" else float(rng.uniform(0.02, 1.0))
        base_kind = BASE_MODELS[i % len(BASE_MODELS)]

        ref, cur = generate_matched_pair(
            n_reference=1800, n_current=1300, seed=scenario_seed, magnitude=magnitude,
            drift_type=drift_type if drift_type != "none" else "covariate",
        )

        columns = sorted(set(pd.get_dummies(ref.X, dummy_na=True).columns) |
                          set(pd.get_dummies(cur.X, dummy_na=True).columns))
        ref_enc = _encode(ref.X, columns)
        cur_enc = _encode(cur.X, columns)

        Xtr, Xho, ytr, yho = train_test_split(ref_enc, ref.y, test_size=0.25, random_state=scenario_seed, stratify=ref.y)
        model = _make_base_model(base_kind, scenario_seed)
        model.fit(Xtr, ytr)

        ref_scores = _eval(model, Xho, yho)
        cur_scores = _eval(model, cur_enc, cur.y)

        report = analyze_drift(ref.X, cur.X)
        feat_vec = drift_report_to_feature_vector(report)

        row = {**feat_vec}
        for m in METRICS:
            row[f"{m}_drop"] = ref_scores[m] - cur_scores[m]
        row["drift_type"] = drift_type
        row["magnitude"] = magnitude
        row["base_model"] = base_kind
        rows.append(row)

        if verbose and (i + 1) % 25 == 0:
            print(f"  scenario {i+1}/{n_scenarios}  ({time.time()-t0:.1f}s elapsed)")

    return pd.DataFrame(rows)


@dataclass
class MetricModelBundle:
    mean_model: GradientBoostingRegressor
    lo_model: GradientBoostingRegressor
    hi_model: GradientBoostingRegressor
    validation: dict


def train_impact_models(df: pd.DataFrame, seed: int = 0) -> dict[str, MetricModelBundle]:
    """Trains one mean model + two quantile models (10th/90th percentile, for
    a confidence band) per metric.

    IMPORTANT, and reported honestly rather than hidden: pure concept drift
    (P(y|X) changes with P(X) unchanged) is invisible to every drift metric
    here, because they all operate on the *input* feature distributions.
    That's not a bug in this implementation, it's a structural limit shared
    by every input-distribution drift detector (PSI, KS, MMD, C2ST, etc.) -
    detecting it requires ground-truth or proxy labels, which is exactly why
    production systems pair drift monitoring with a delayed-label feedback
    loop. We report validation broken out by drift type below so this
    limitation is visible rather than averaged away.
    """
    X = df[FEATURE_ORDER].to_numpy(dtype=float)
    bundles: dict[str, MetricModelBundle] = {}

    idx = np.arange(len(df))
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    split = int(len(idx) * 0.8)
    train_idx, test_idx = idx[:split], idx[split:]
    test_drift_types = df["drift_type"].to_numpy()[test_idx]

    for metric in METRICS:
        y = df[f"{metric}_drop"].to_numpy(dtype=float)
        Xtr, Xte = X[train_idx], X[test_idx]
        ytr, yte = y[train_idx], y[test_idx]

        mean_model = GradientBoostingRegressor(n_estimators=150, max_depth=3, learning_rate=0.05, random_state=seed)
        mean_model.fit(Xtr, ytr)

        lo_model = GradientBoostingRegressor(loss="quantile", alpha=0.1, n_estimators=150, max_depth=3, learning_rate=0.05, random_state=seed)
        lo_model.fit(Xtr, ytr)

        hi_model = GradientBoostingRegressor(loss="quantile", alpha=0.9, n_estimators=150, max_depth=3, learning_rate=0.05, random_state=seed)
        hi_model.fit(Xtr, ytr)

        pred_te = mean_model.predict(Xte)
        ss_res = float(np.sum((yte - pred_te) ** 2))
        ss_tot = float(np.sum((yte - yte.mean()) ** 2)) or 1e-9
        r2 = 1 - ss_res / ss_tot
        mae = float(np.mean(np.abs(yte - pred_te)))

        mae_by_type = {}
        for dt in np.unique(test_drift_types):
            mask = test_drift_types == dt
            if mask.sum() > 0:
                mae_by_type[str(dt)] = round(float(np.mean(np.abs(yte[mask] - pred_te[mask]))), 4)

        bundles[metric] = MetricModelBundle(
            mean_model=mean_model,
            lo_model=lo_model,
            hi_model=hi_model,
            validation={
                "r2": round(r2, 4),
                "mae": round(mae, 4),
                "mae_by_drift_type": mae_by_type,
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
            },
        )
    return bundles


def save_models(bundles: dict[str, MetricModelBundle], path: Path | None = None) -> Path:
    path = path or ARTIFACT_DIR
    path.mkdir(parents=True, exist_ok=True)
    out = path / "impact_models.joblib"
    joblib.dump(bundles, out)
    return out


_CACHE: dict[str, MetricModelBundle] | None = None


def load_models(path: Path | None = None) -> dict[str, MetricModelBundle]:
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    path = path or ARTIFACT_DIR
    fp = path / "impact_models.joblib"
    if not fp.exists():
        raise FileNotFoundError(
            f"No trained impact model found at {fp}. Run train_impact_model.py first."
        )
    _CACHE = joblib.load(fp)
    return _CACHE


def predict_degradation(drift_features: dict, metric: str = "accuracy") -> dict:
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {METRICS}")
    bundles = load_models()
    bundle = bundles[metric]
    x = np.array([[drift_features.get(f, 0.0) for f in FEATURE_ORDER]])

    mean_pred = float(bundle.mean_model.predict(x)[0])
    lo_pred = float(bundle.lo_model.predict(x)[0])
    hi_pred = float(bundle.hi_model.predict(x)[0])
    lo_pred, hi_pred = min(lo_pred, hi_pred), max(lo_pred, hi_pred)

    importances = bundle.mean_model.feature_importances_
    feature_importance = sorted(
        [{"feature": f, "importance": round(float(imp), 4)} for f, imp in zip(FEATURE_ORDER, importances)],
        key=lambda r: -r["importance"],
    )

    return {
        "metric": metric,
        "predicted_degradation": round(mean_pred, 4),
        "confidence_interval": [round(lo_pred, 4), round(hi_pred, 4)],
        "feature_importance": feature_importance,
        "model_validation": bundle.validation,
    }
