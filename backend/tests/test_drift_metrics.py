import numpy as np
import pandas as pd
from app.core.drift_metrics import (
    analyze_drift, drift_report_to_feature_vector, _overall_severity,
    OVERALL_MODERATE, OVERALL_MAJOR,
)
from app.core.synthetic_data import generate_matched_pair


def test_overall_severity_thresholds_are_consistent():
    """Regression test: severity used to be computed on mean_psi below 0.5
    overall and on overall-score thresholds above it, a confusing two-branch
    rule. It should now be a single monotonic function of overall score."""
    assert _overall_severity(0.0) == "none"
    assert _overall_severity(OVERALL_MODERATE - 0.01) == "none"
    assert _overall_severity(OVERALL_MODERATE) == "moderate"
    assert _overall_severity(OVERALL_MAJOR - 0.01) == "moderate"
    assert _overall_severity(OVERALL_MAJOR) == "major"
    assert _overall_severity(1.5) == "major"


def test_severity_matches_overall_score_in_full_report():
    ref, cur = generate_matched_pair(n_reference=800, n_current=500, seed=5, magnitude=0.6, drift_type="covariate")
    report = analyze_drift(ref.X, cur.X)
    assert report.severity == _overall_severity(report.overall_drift_score)


def test_symmetric_all_nan_column_routes_to_categorical_and_does_not_crash():
    """Both sides all-NaN: nunique()==0 on the reference side routes this to
    the categorical path (which already tolerates empty/degenerate inputs via
    epsilon smoothing), not the numeric path. It shouldn't crash either way."""
    ref, cur = generate_matched_pair(n_reference=300, n_current=300, seed=1, magnitude=0.3, drift_type="covariate")
    ref.X["broken_col"] = np.nan
    cur.X["broken_col"] = np.nan
    report = analyze_drift(ref.X, cur.X)  # should not raise
    broken = next(f for f in report.features if f.feature == "broken_col")
    assert broken.severity == "none"
    assert broken.dtype == "categorical"


def test_asymmetric_all_nan_numeric_column_does_not_crash():
    """Reference has normal high-cardinality numeric data (so it's routed to
    the numeric path via nunique() > 10), but current is entirely NaN for
    that column - e.g. a field a new data source doesn't populate yet. This
    is what the defensive guard in _numeric_feature_drift protects against:
    calling .min()/.max() on an empty array used to crash here."""
    ref, cur = generate_matched_pair(n_reference=300, n_current=300, seed=1, magnitude=0.3, drift_type="covariate")
    cur.X["num_0"] = np.nan  # reference's num_0 still has >10 unique values
    report = analyze_drift(ref.X, cur.X)  # should not raise
    broken = next(f for f in report.features if f.feature == "num_0")
    assert broken.dtype == "numeric"
    assert broken.severity == "none"
    assert broken.p_value is None


def test_feature_vector_matches_expected_keys():
    ref, cur = generate_matched_pair(n_reference=400, n_current=300, seed=2, magnitude=0.4, drift_type="covariate")
    report = analyze_drift(ref.X, cur.X)
    vec = drift_report_to_feature_vector(report)
    expected_keys = {
        "mean_psi", "max_psi", "mean_js_divergence", "max_js_divergence",
        "frac_significant", "c2st_auc", "drift_fraction", "n_features",
        "overall_drift_score",
    }
    assert set(vec.keys()) == expected_keys
    assert all(isinstance(v, float) for v in vec.values())


def test_categorical_column_routed_correctly():
    ref, cur = generate_matched_pair(n_reference=500, n_current=300, seed=7, magnitude=0.5, drift_type="covariate")
    report = analyze_drift(ref.X, cur.X)
    cat_features = [f for f in report.features if f.dtype == "categorical"]
    assert len(cat_features) == 2  # cat_0, cat_1 from synthetic_data.py
    assert all(f.test_name == "chi_square" for f in cat_features)
