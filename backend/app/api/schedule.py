from __future__ import annotations
from fastapi import APIRouter, HTTPException

from ..models.schemas import ScheduleRequest, ScheduleResponse
from ..core.scheduler import recommend_schedule
from .. import storage

router = APIRouter(prefix="/api/schedule", tags=["schedule"])


@router.post("/recommend", response_model=ScheduleResponse)
def recommend(req: ScheduleRequest):
    if req.analysis_id:
        try:
            feature_vector = storage.get_analysis(req.analysis_id)["feature_vector"]
        except KeyError as e:
            raise HTTPException(404, str(e))
    else:
        # a reasonable generic reference vector if no analysis was run yet
        feature_vector = {
            "mean_psi": 0.1, "max_psi": 0.2, "mean_js_divergence": 0.05,
            "max_js_divergence": 0.1, "frac_significant": 0.3, "c2st_auc": 0.6,
            "drift_fraction": 0.3, "n_features": 8.0, "overall_drift_score": 0.2,
        }

    if not req.drift_history:
        raise HTTPException(400, "drift_history must contain at least one point")

    try:
        result = recommend_schedule(
            drift_history=[h.model_dump() for h in req.drift_history],
            reference_feature_vector=feature_vector,
            metric=req.metric,
            retrain_cost=req.retrain_cost,
            degradation_cost_per_point_per_day=req.degradation_cost_per_point_per_day,
            horizon_days=req.horizon_days,
            degradation_threshold=req.degradation_threshold,
        )
    except FileNotFoundError as e:
        raise HTTPException(503, str(e))

    return {
        "strategy_costs": result.strategy_costs,
        "recommended_strategy": result.recommended_strategy,
        "recommended_interval_days": result.recommended_interval_days,
        "next_retrain_in_days": result.next_retrain_in_days,
        "projected_trajectory": result.projected_trajectory,
        "trend_slope_per_day": result.trend_slope_per_day,
        "floor_warning": result.floor_warning,
    }
