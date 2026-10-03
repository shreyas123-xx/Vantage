from app.core.scheduler import recommend_schedule

REFERENCE_VECTOR = {
    "mean_psi": 0.3, "max_psi": 0.5, "mean_js_divergence": 0.1,
    "max_js_divergence": 0.2, "frac_significant": 0.5, "c2st_auc": 0.75,
    "drift_fraction": 0.5, "n_features": 8.0, "overall_drift_score": 0.6,
}


def test_immediate_threshold_crossing_means_retrain_now_not_never():
    """Regression test: an earlier version treated 'degradation already at
    threshold on day 0' the same as 'never crosses the threshold', which is
    backwards - it should mean retrain immediately (or as often as daily),
    not never."""
    history = [{"day": 0, "drift_score": 0.05}, {"day": 10, "drift_score": 0.3}]
    result = recommend_schedule(
        history, REFERENCE_VECTOR, metric="accuracy",
        horizon_days=60, degradation_threshold=0.0001,  # trivially low threshold
    )
    assert result.floor_warning is not None
    assert result.strategy_costs["adaptive_threshold"]["trigger_after_days"] <= 1
    assert result.strategy_costs["adaptive_threshold"]["n_retrains"] > 0


def test_small_horizon_still_produces_a_periodic_candidate():
    """Regression test: candidate_intervals used to be a fixed list starting
    at 7, so horizon_days=7 (the schema minimum) excluded every candidate and
    best_fixed_periodic silently disappeared from the response."""
    history = [{"day": 0, "drift_score": 0.05}, {"day": 5, "drift_score": 0.2}]
    result = recommend_schedule(history, REFERENCE_VECTOR, metric="accuracy", horizon_days=7)
    assert "best_fixed_periodic" in result.strategy_costs
    assert result.strategy_costs["best_fixed_periodic"]["interval_days"] < 7


def test_flat_drift_history_recommends_never_retrain():
    # Threshold chosen comfortably above the impact model's zero-drift floor
    # (~0.12 for this reference vector - see test_floor_warning_fires_below_model_floor
    # below) so this test isolates "flat history" from "threshold too strict".
    history = [{"day": 0, "drift_score": 0.01}, {"day": 10, "drift_score": 0.012}, {"day": 20, "drift_score": 0.011}]
    result = recommend_schedule(history, REFERENCE_VECTOR, metric="accuracy", horizon_days=60, degradation_threshold=0.2)
    assert result.recommended_strategy == "never_retrain"
    assert result.floor_warning is None


def test_floor_warning_fires_below_model_floor_regardless_of_history():
    """floor_warning depends only on the model's zero-drift prediction vs the
    threshold, not on the drift history - a threshold below the floor should
    fire the warning even for a perfectly flat, low history."""
    history = [{"day": 0, "drift_score": 0.01}, {"day": 10, "drift_score": 0.01}]
    result = recommend_schedule(history, REFERENCE_VECTOR, metric="accuracy", horizon_days=60, degradation_threshold=0.01)
    assert result.floor_warning is not None


def test_single_history_point_does_not_crash():
    history = [{"day": 0, "drift_score": 0.1}]
    result = recommend_schedule(history, REFERENCE_VECTOR, metric="accuracy", horizon_days=30)
    assert result.trend_slope_per_day == 0.0  # can't fit a trend from one point


def test_zero_retrain_adaptive_costs_exactly_the_same_as_never_retrain():
    """Regression test: cost_for_strategy used to assume day 0 was always
    'right after a fresh retrain' (the since-retrain curve), even for a
    strategy that triggers zero actual retrains. That made a strategy with
    n_retrains=0 cost something different from the literal never-retrain
    baseline, even though both describe the exact same thing: nothing ever
    reset. A flat, low-drift history with a high threshold should produce
    an adaptive strategy that never fires, and its cost must match
    never_retrain to the cent."""
    history = [{"day": 0, "drift_score": 0.01}, {"day": 10, "drift_score": 0.011}]
    result = recommend_schedule(history, REFERENCE_VECTOR, metric="accuracy", horizon_days=60, degradation_threshold=0.49)
    assert result.strategy_costs["adaptive_threshold"]["n_retrains"] == 0
    assert result.strategy_costs["adaptive_threshold"]["total_cost"] == result.strategy_costs["never_retrain"]["total_cost"]
