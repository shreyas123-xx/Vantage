"""
synthetic_data.py's whole job is to be a trustworthy benchmark: reference and
current must be statistically identical at magnitude=0 and diverge smoothly
as magnitude increases. An earlier version of this generator called
make_classification separately for reference and current, which randomizes
the feature-to-signal geometry itself and manufactured fake drift even at
"zero drift". These tests pin that behavior down so it can't regress.
"""
import numpy as np
from app.core.synthetic_data import generate_matched_pair, generate_longitudinal_world
from app.core.drift_metrics import analyze_drift


def test_zero_magnitude_shows_minimal_drift():
    ref, cur = generate_matched_pair(n_reference=1000, n_current=600, seed=1, magnitude=0.0, drift_type="covariate")
    report = analyze_drift(ref.X, cur.X)
    # Should reflect only natural sampling noise, not manufactured drift.
    assert report.overall_drift_score < 0.2
    assert report.c2st_auc < 0.6


def test_drift_score_increases_with_magnitude():
    scores = []
    for mag in (0.0, 0.3, 0.6, 1.0):
        ref, cur = generate_matched_pair(n_reference=800, n_current=500, seed=5, magnitude=mag, drift_type="covariate")
        report = analyze_drift(ref.X, cur.X)
        scores.append(report.overall_drift_score)
    # Monotonically non-decreasing as magnitude rises.
    assert all(a <= b + 1e-9 for a, b in zip(scores, scores[1:])), scores


def test_concept_drift_does_not_move_input_distribution():
    """Concept drift flips labels without touching X - drift metrics (which
    only see X) should stay low even at high magnitude. This is the
    documented structural blind spot, not a bug - pin it down explicitly."""
    ref, cur = generate_matched_pair(n_reference=1000, n_current=600, seed=3, magnitude=0.9, drift_type="concept")
    report = analyze_drift(ref.X, cur.X)
    assert report.overall_drift_score < 0.3


def test_reference_and_current_same_shape_different_seeds_independent():
    """Different seeds should give different (but still low, since magnitude=0)
    drift scores - i.e. the world pool is seed-dependent but internally consistent."""
    ref1, cur1 = generate_matched_pair(n_reference=500, n_current=300, seed=10, magnitude=0.0, drift_type="covariate")
    ref2, cur2 = generate_matched_pair(n_reference=500, n_current=300, seed=20, magnitude=0.0, drift_type="covariate")
    assert not ref1.X.equals(ref2.X)  # different seeds -> different data
    for ref, cur in [(ref1, cur1), (ref2, cur2)]:
        report = analyze_drift(ref.X, cur.X)
        assert report.overall_drift_score < 0.2


def test_longitudinal_world_reference_matches_first_step_distribution_at_low_magnitude():
    reference, steps = generate_longitudinal_world(
        seed=42, n_reference=1000, n_per_step=300, n_steps=3,
        magnitudes=[0.01, 0.01, 0.01], drift_type="covariate",
    )
    report = analyze_drift(reference.X, steps[0].X)
    assert report.overall_drift_score < 0.2
    assert len(steps) == 3
