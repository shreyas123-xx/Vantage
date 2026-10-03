from __future__ import annotations
from fastapi import APIRouter, HTTPException

from ..models.schemas import LongitudinalRequest, LongitudinalResponse
from ..core.longitudinal import run_longitudinal_experiment

router = APIRouter(prefix="/api/experiments", tags=["experiments"])


@router.post("/longitudinal", response_model=LongitudinalResponse)
def longitudinal(req: LongitudinalRequest):
    try:
        result = run_longitudinal_experiment(
            n_weeks=req.n_weeks,
            drift_pattern=req.drift_pattern,
            peak_magnitude=req.peak_magnitude,
            drift_type=req.drift_type,
            retrain_strategy=req.retrain_strategy,
            fixed_interval_weeks=req.fixed_interval_weeks,
            degradation_threshold=req.degradation_threshold,
            metric=req.metric,
            seed=req.seed,
        )
    except FileNotFoundError as e:
        raise HTTPException(503, str(e))

    return {
        "weeks": result.weeks,
        "summary": result.summary,
        "strategy_comparison": result.strategy_comparison,
    }
