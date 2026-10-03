"""
Turns a drift trajectory + the impact model's degradation predictions into a
concrete, minimal retraining schedule.

Two different curves matter here and must not be conflated:

  * "never retrain" forecast - starting from today's actual drift level,
    project the historical trend forward untouched. This is "what happens if
    we do nothing", used for the never-retrain baseline cost and for the
    chart the user sees.
  * "since-last-retrain" curve - starting from ~zero drift (a fresh retrain
    matches the current data by definition) and growing at the same trend
    slope. This is what periodic/adaptive strategies reset to *after* an
    actual retrain happens.

The important bit, found by a failing test: before a strategy's *first*
retrain, nothing has reset yet, so that period must still be costed with the
never-retrain curve (today's actual accumulated drift), not the since-retrain
curve. Using the since-retrain curve from day zero regardless of whether a
retrain had actually happened made a strategy with zero retrains compute a
different cost than the literal "never retrain" baseline, which is
incoherent - they describe the same thing and must match exactly.

Three strategies are evaluated over the horizon:

  * never retrain           - baseline cost of doing nothing
  * fixed periodic retrain  - retrain every N days, sweep N to find the best
  * adaptive threshold      - retrain once the current trajectory first
                               crosses a business-defined threshold, then
                               keep retraining every time the post-reset
                               curve crosses it again (the "minimal
                               necessary" schedule)

Total cost for a strategy = (number of retrains * retrain_cost) + (integral
of degradation_cost_per_point_per_day * degradation over the horizon). The
recommended schedule is whichever strategy minimizes total projected cost.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from .impact_model import predict_degradation


@dataclass
class ScheduleRecommendation:
    strategy_costs: dict
    recommended_strategy: str
    recommended_interval_days: int | None
    next_retrain_in_days: int
    projected_trajectory: list[dict]
    trend_slope_per_day: float
    floor_warning: str | None = None


def _fit_trend(days: list[float], scores: list[float]) -> tuple[float, float]:
    """Simple linear trend fit through the drift-score history."""
    x = np.array(days, dtype=float)
    y = np.array(scores, dtype=float)
    if len(x) < 2:
        return 0.0, float(y[-1]) if len(y) else 0.0
    x0 = x - x[0]
    slope, intercept = np.polyfit(x0, y, 1)
    return float(slope), float(intercept)


def _drift_score_to_features(drift_score: float, reference_feature_vector: dict) -> dict:
    """Scales a reference drift-feature vector's magnitude-sensitive fields by
    the ratio of a target drift score to the reference analysis's own overall
    score, so an arbitrary point on a projected trajectory can be fed through
    the same impact model. Scale-invariant proportions/counts are held fixed."""
    base_score = max(reference_feature_vector.get("overall_drift_score", 0.0), 1e-6)
    ratio = max(0.0, drift_score) / base_score
    scaled = dict(reference_feature_vector)
    for key in ("mean_psi", "max_psi", "mean_js_divergence", "max_js_divergence"):
        scaled[key] = reference_feature_vector.get(key, 0.0) * ratio
    scaled["c2st_auc"] = float(np.clip(0.5 + (reference_feature_vector.get("c2st_auc", 0.5) - 0.5) * ratio, 0.5, 1.0))
    scaled["overall_drift_score"] = max(0.0, drift_score)
    return scaled


def _degradation_curve(drift_scores: list[float], reference_feature_vector: dict, metric: str) -> list[float]:
    out = []
    for score in drift_scores:
        feats = _drift_score_to_features(score, reference_feature_vector)
        pred = predict_degradation(feats, metric=metric)
        out.append(max(0.0, pred["predicted_degradation"]))
    return out


def recommend_schedule(
    drift_history: list[dict],
    reference_feature_vector: dict,
    metric: str = "accuracy",
    retrain_cost: float = 500.0,
    degradation_cost_per_point_per_day: float = 50.0,
    horizon_days: int = 60,
    degradation_threshold: float = 0.15,
    candidate_intervals: list[int] | None = None,
) -> ScheduleRecommendation:
    days = [h["day"] for h in drift_history]
    scores = [h["drift_score"] for h in drift_history]
    slope, _intercept = _fit_trend(days, scores)
    last_observed = scores[-1]

    # "do nothing from today" forecast, anchored at the latest observed point.
    # This is the ground truth for "no retrain has happened yet" at any point
    # in any strategy, not just the never-retrain strategy.
    never_retrain_scores = [max(0.0, last_observed + slope * d) for d in range(horizon_days)]
    never_retrain_degradation = _degradation_curve(never_retrain_scores, reference_feature_vector, metric)

    # "since a fresh retrain" curve, anchored at ~zero drift and growing at the
    # same slope. Only valid for time elapsed *after* an actual retrain.
    since_retrain_scores = [max(0.0, slope * k) for k in range(horizon_days)]
    since_retrain_degradation = _degradation_curve(since_retrain_scores, reference_feature_vector, metric)

    floor_warning = None
    if since_retrain_degradation[0] >= degradation_threshold:
        floor_warning = (
            "The impact model's predicted degradation right after a hypothetical retrain "
            "already meets the threshold. This usually means the threshold is set below the "
            "model's irreducible prediction floor (partly driven by the concept-drift blind "
            "spot - see the note on the Impact tab) rather than that retraining daily is truly "
            "necessary. Consider raising degradation_threshold."
        )

    if candidate_intervals is None:
        # Generated relative to horizon_days rather than a fixed list, so a
        # short horizon (schema minimum is 7 days) still gets at least one
        # periodic candidate instead of silently dropping best_fixed_periodic.
        base = [1, 2, 3, 5, 7, 10, 14, 21, 30, 45, 60, 90, 120, 180]
        candidate_intervals = [d for d in base if d < horizon_days]
        if not candidate_intervals:
            candidate_intervals = [max(1, horizon_days // 2)]

    def cost_for_strategy(retrain_days: set[int]) -> float:
        """retrain_days may be empty - in which case this must reduce to
        exactly the never-retrain cost, since 'zero retrains' describes one
        single scenario regardless of which strategy produced it."""
        total = 0.0
        last_retrain_day = None  # None until this strategy's first retrain
        for d in range(horizon_days):
            if d in retrain_days:
                last_retrain_day = d
                total += retrain_cost
            if last_retrain_day is None:
                deg = never_retrain_degradation[min(d, len(never_retrain_degradation) - 1)]
            else:
                idx = min(d - last_retrain_day, len(since_retrain_degradation) - 1)
                deg = since_retrain_degradation[idx]
            total += degradation_cost_per_point_per_day * deg
        return total

    strategy_costs = {}

    never_cost = cost_for_strategy(set())
    strategy_costs["never_retrain"] = {"total_cost": round(never_cost, 2), "n_retrains": 0}

    best_periodic = None
    for interval in candidate_intervals:
        if interval >= horizon_days:
            continue
        retrain_days = set(range(interval, horizon_days, interval))
        cost = cost_for_strategy(retrain_days)
        entry = {"interval_days": interval, "total_cost": round(cost, 2), "n_retrains": len(retrain_days)}
        if best_periodic is None or cost < best_periodic["total_cost"]:
            best_periodic = entry
    if best_periodic:
        strategy_costs["best_fixed_periodic"] = best_periodic

    # Adaptive, two phases:
    #  1. first retrain fires once TODAY's actual trajectory (never_retrain
    #     curve) first crosses the threshold - it may be very soon if we're
    #     already close, or never, if the trend is flat enough.
    #  2. every retrain after that fires once the POST-RESET trajectory
    #     (since_retrain curve) crosses the threshold again, repeating.
    first_retrain_day = next(
        (d for d, deg in enumerate(never_retrain_degradation) if deg >= degradation_threshold), None
    )
    subsequent_interval = next(
        (k for k, deg in enumerate(since_retrain_degradation) if k >= 1 and deg >= degradation_threshold), None
    )
    if subsequent_interval is None and floor_warning:
        subsequent_interval = 1  # already at/above threshold immediately after a reset

    adaptive_days: set[int] = set()
    if first_retrain_day is not None:
        d = first_retrain_day
        while d < horizon_days:
            adaptive_days.add(d)
            if subsequent_interval is None:
                break  # post-reset trajectory never re-crosses the threshold
            d += subsequent_interval

    adaptive_cost = cost_for_strategy(adaptive_days)
    strategy_costs["adaptive_threshold"] = {
        "trigger_after_days": first_retrain_day,
        "repeat_every_days": subsequent_interval,
        "total_cost": round(adaptive_cost, 2),
        "n_retrains": len(adaptive_days),
        "degradation_threshold": degradation_threshold,
    }

    candidates = {"never_retrain": never_cost}
    if best_periodic:
        candidates[f"fixed_every_{best_periodic['interval_days']}_days"] = best_periodic["total_cost"]
    candidates["adaptive_threshold"] = adaptive_cost
    recommended_strategy = min(candidates, key=candidates.get)

    recommended_interval = None
    next_retrain_in = horizon_days
    if recommended_strategy == "adaptive_threshold":
        recommended_interval = subsequent_interval
        next_retrain_in = first_retrain_day if first_retrain_day is not None else horizon_days
    elif recommended_strategy.startswith("fixed_every"):
        recommended_interval = best_periodic["interval_days"]
        next_retrain_in = best_periodic["interval_days"]

    trajectory = [
        {
            "day": d,
            "projected_drift_score": round(never_retrain_scores[d], 4),
            "projected_degradation": round(never_retrain_degradation[d], 4),
        }
        for d in range(horizon_days)
    ]

    return ScheduleRecommendation(
        strategy_costs=strategy_costs,
        recommended_strategy=recommended_strategy,
        recommended_interval_days=recommended_interval,
        next_retrain_in_days=next_retrain_in,
        projected_trajectory=trajectory,
        trend_slope_per_day=round(slope, 5),
        floor_warning=floor_warning,
    )
