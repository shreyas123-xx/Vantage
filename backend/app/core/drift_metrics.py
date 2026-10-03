"""
Drift detection engine.

Computes per-feature univariate drift statistics (PSI, KS test, Wasserstein
distance, Jensen-Shannon divergence) for numeric features and (PSI, chi-square,
JS divergence) for categorical features, plus a multivariate summary using a
classifier two-sample test (C2ST): a classifier is trained to tell reference
and current rows apart, and its held-out AUC is a model-free measure of how
separable the two distributions are (0.5 = indistinguishable, 1.0 = trivially
separable).

These per-feature + multivariate numbers are exactly the feature vector that
the impact model (core/impact_model.py) is trained to map onto predicted
downstream performance degradation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sstats
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from dataclasses import dataclass
from typing import Literal

PSI_MODERATE = 0.1
PSI_MAJOR = 0.25
EPS = 1e-6

# overall_drift_score (see analyze_drift) is a blend of mean PSI and C2ST
# separability with a max of ~1.5; these thresholds are calibrated against
# that combined scale, not the raw per-feature PSI thresholds above.
OVERALL_MODERATE = 0.3
OVERALL_MAJOR = 0.75


def _psi(reference: np.ndarray, current: np.ndarray, bins: np.ndarray) -> float:
    ref_counts, _ = np.histogram(reference, bins=bins)
    cur_counts, _ = np.histogram(current, bins=bins)
    ref_pct = ref_counts / max(len(reference), 1) + EPS
    cur_pct = cur_counts / max(len(current), 1) + EPS
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = p / (p.sum() + EPS) + EPS
    q = q / (q.sum() + EPS) + EPS
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))
    # normalize to [0, 1] (natural-log JS divergence has max ln(2))
    return float(0.5 * (kl_pm + kl_qm) / np.log(2))


def _severity(psi: float) -> Literal["none", "moderate", "major"]:
    if psi >= PSI_MAJOR:
        return "major"
    if psi >= PSI_MODERATE:
        return "moderate"
    return "none"


def _overall_severity(overall: float) -> Literal["none", "moderate", "major"]:
    if overall >= OVERALL_MAJOR:
        return "major"
    if overall >= OVERALL_MODERATE:
        return "moderate"
    return "none"


@dataclass
class FeatureDrift:
    feature: str
    dtype: Literal["numeric", "categorical"]
    psi: float
    js_divergence: float
    test_name: str
    test_statistic: float
    p_value: float | None
    drifted: bool
    severity: Literal["none", "moderate", "major"]
    reference_mean: float | None = None
    current_mean: float | None = None
    reference_std: float | None = None
    current_std: float | None = None

    def to_dict(self):
        return self.__dict__


@dataclass
class DriftReport:
    features: list[FeatureDrift]
    c2st_auc: float
    overall_drift_score: float
    n_drifted_features: int
    n_features: int
    drift_fraction: float
    severity: Literal["none", "moderate", "major"]
    n_reference: int
    n_current: int

    def to_dict(self):
        d = self.__dict__.copy()
        d["features"] = [f.to_dict() for f in self.features]
        return d


def _numeric_feature_drift(name: str, ref: pd.Series, cur: pd.Series) -> FeatureDrift:
    ref_v = ref.dropna().to_numpy(dtype=float)
    cur_v = cur.dropna().to_numpy(dtype=float)

    if len(ref_v) == 0 or len(cur_v) == 0:
        # A column that's all-NaN on one side carries no distributional
        # information to compare - report it as unmeasurable rather than
        # crashing on .min()/.max() of an empty array.
        return FeatureDrift(
            feature=name, dtype="numeric", psi=0.0, js_divergence=0.0,
            test_name="ks_2samp", test_statistic=0.0, p_value=None,
            drifted=False, severity="none",
        ), 0.0

    lo = min(ref_v.min(), cur_v.min())
    hi = max(ref_v.max(), cur_v.max())
    if hi - lo < EPS:
        hi = lo + 1.0
    bins = np.linspace(lo, hi, 11)  # decile-style bins shared across both samples

    psi = _psi(ref_v, cur_v, bins)
    ref_hist, _ = np.histogram(ref_v, bins=bins)
    cur_hist, _ = np.histogram(cur_v, bins=bins)
    js = _js_divergence(ref_hist.astype(float), cur_hist.astype(float))

    ks_stat, ks_p = sstats.ks_2samp(ref_v, cur_v)

    ref_std = float(ref_v.std()) or 1.0
    wasserstein = sstats.wasserstein_distance(ref_v, cur_v) / ref_std

    drifted = psi >= PSI_MODERATE or ks_p < 0.05
    return FeatureDrift(
        feature=name,
        dtype="numeric",
        psi=round(psi, 4),
        js_divergence=round(js, 4),
        test_name="ks_2samp",
        test_statistic=round(float(ks_stat), 4),
        p_value=round(float(ks_p), 6),
        drifted=bool(drifted),
        severity=_severity(max(psi, 0)),
        reference_mean=round(float(ref_v.mean()), 4),
        current_mean=round(float(cur_v.mean()), 4),
        reference_std=round(ref_std, 4),
        current_std=round(float(cur_v.std()), 4),
    ), wasserstein


def _categorical_feature_drift(name: str, ref: pd.Series, cur: pd.Series) -> FeatureDrift:
    categories = sorted(set(ref.dropna().unique()) | set(cur.dropna().unique()))
    ref_counts = ref.value_counts().reindex(categories, fill_value=0).to_numpy(dtype=float)
    cur_counts = cur.value_counts().reindex(categories, fill_value=0).to_numpy(dtype=float)

    ref_pct = ref_counts / max(ref_counts.sum(), 1) + EPS
    cur_pct = cur_counts / max(cur_counts.sum(), 1) + EPS
    psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
    js = _js_divergence(ref_counts, cur_counts)

    try:
        chi2, p_value, _, _ = sstats.chi2_contingency(
            np.vstack([ref_counts, cur_counts]) + EPS
        )
    except ValueError:
        chi2, p_value = 0.0, 1.0

    drifted = psi >= PSI_MODERATE or p_value < 0.05
    return FeatureDrift(
        feature=name,
        dtype="categorical",
        psi=round(psi, 4),
        js_divergence=round(js, 4),
        test_name="chi_square",
        test_statistic=round(float(chi2), 4),
        p_value=round(float(p_value), 6),
        drifted=bool(drifted),
        severity=_severity(max(psi, 0)),
    )


def _classifier_two_sample_test(reference: pd.DataFrame, current: pd.DataFrame) -> float:
    """Train a classifier to distinguish reference (0) vs current (1) rows.
    Cross-validated AUC away from 0.5 is a distribution-free multivariate
    drift signal that captures interactions PSI/KS miss feature-by-feature.
    """
    ref_enc = pd.get_dummies(reference, dummy_na=True)
    cur_enc = pd.get_dummies(current, dummy_na=True)
    cols = sorted(set(ref_enc.columns) | set(cur_enc.columns))
    ref_enc = ref_enc.reindex(columns=cols, fill_value=0)
    cur_enc = cur_enc.reindex(columns=cols, fill_value=0)

    X = pd.concat([ref_enc, cur_enc], axis=0).to_numpy(dtype=float)
    y = np.concatenate([np.zeros(len(ref_enc)), np.ones(len(cur_enc))])

    if len(np.unique(y)) < 2 or len(y) < 20:
        return 0.5

    clf = GradientBoostingClassifier(n_estimators=60, max_depth=2, random_state=0)
    try:
        scores = cross_val_score(clf, X, y, cv=4, scoring="roc_auc")
        return float(np.mean(scores))
    except ValueError:
        return 0.5


def analyze_drift(reference: pd.DataFrame, current: pd.DataFrame) -> DriftReport:
    """Compare a reference (baseline/training) dataframe to a current
    (production) dataframe, column by column, plus a multivariate summary."""
    common_cols = [c for c in reference.columns if c in current.columns]
    features: list[FeatureDrift] = []
    wasserstein_scores = []

    for col in common_cols:
        ref_col, cur_col = reference[col], current[col]
        is_numeric = pd.api.types.is_numeric_dtype(ref_col) and pd.api.types.is_numeric_dtype(cur_col)
        if is_numeric and ref_col.nunique() > 10:
            fd, wasserstein = _numeric_feature_drift(col, ref_col, cur_col)
            wasserstein_scores.append(wasserstein)
        else:
            fd = _categorical_feature_drift(col, ref_col.astype(str), cur_col.astype(str))
        features.append(fd)

    c2st_auc = _classifier_two_sample_test(reference[common_cols], current[common_cols])

    n_drifted = sum(f.drifted for f in features)
    mean_psi = float(np.mean([f.psi for f in features])) if features else 0.0
    # Overall drift score blends mean per-feature PSI with multivariate separability,
    # scaled to roughly [0, 1+] so downstream thresholds are stable across dataset sizes.
    overall = 0.5 * min(mean_psi / PSI_MAJOR, 2.0) + 0.5 * max(0.0, (c2st_auc - 0.5) * 2)

    report = DriftReport(
        features=features,
        c2st_auc=round(c2st_auc, 4),
        overall_drift_score=round(float(overall), 4),
        n_drifted_features=n_drifted,
        n_features=len(features),
        drift_fraction=round(n_drifted / len(features), 4) if features else 0.0,
        severity=_overall_severity(overall),
        n_reference=len(reference),
        n_current=len(current),
    )
    return report


def drift_report_to_feature_vector(report: DriftReport) -> dict:
    """Collapse a DriftReport into the fixed-width numeric feature vector
    consumed by the impact model. Keep this in sync with impact_model.py's
    FEATURE_ORDER — training and inference must build the vector identically.
    """
    psis = [f.psi for f in report.features] or [0.0]
    js = [f.js_divergence for f in report.features] or [0.0]
    pvals = [f.p_value for f in report.features if f.p_value is not None] or [1.0]
    return {
        "mean_psi": float(np.mean(psis)),
        "max_psi": float(np.max(psis)),
        "mean_js_divergence": float(np.mean(js)),
        "max_js_divergence": float(np.max(js)),
        "frac_significant": float(np.mean([p < 0.05 for p in pvals])),
        "c2st_auc": report.c2st_auc,
        "drift_fraction": report.drift_fraction,
        "n_features": float(report.n_features),
        "overall_drift_score": report.overall_drift_score,
    }
