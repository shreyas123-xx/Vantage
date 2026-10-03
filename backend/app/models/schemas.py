from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Literal, Optional


# ---------- Datasets ----------

class SyntheticDatasetRequest(BaseModel):
    n_reference: int = Field(1500, ge=200, le=5000)
    n_current: int = Field(800, ge=200, le=5000)
    magnitude: float = Field(0.4, ge=0.0, le=1.0)
    drift_type: Literal["covariate", "concept", "mixed"] = "covariate"
    seed: int = 42


class SyntheticDatasetResponse(BaseModel):
    reference_preview: list[dict]
    current_preview: list[dict]
    reference_id: str
    current_id: str
    n_reference: int
    n_current: int


# ---------- Drift ----------

class DriftAnalyzeRequest(BaseModel):
    reference_id: Optional[str] = None
    current_id: Optional[str] = None


class FeatureDriftOut(BaseModel):
    feature: str
    dtype: str
    psi: float
    js_divergence: float
    test_name: str
    test_statistic: float
    p_value: Optional[float]
    drifted: bool
    severity: str
    reference_mean: Optional[float] = None
    current_mean: Optional[float] = None
    reference_std: Optional[float] = None
    current_std: Optional[float] = None


class DriftAnalyzeResponse(BaseModel):
    features: list[FeatureDriftOut]
    c2st_auc: float
    overall_drift_score: float
    n_drifted_features: int
    n_features: int
    drift_fraction: float
    severity: str
    n_reference: int
    n_current: int
    drift_feature_vector: dict
    analysis_id: str


# ---------- Impact ----------

class ImpactPredictRequest(BaseModel):
    analysis_id: Optional[str] = None
    drift_feature_vector: Optional[dict] = None
    metric: Literal["accuracy", "f1", "roc_auc"] = "accuracy"


class ImpactPredictResponse(BaseModel):
    metric: str
    predicted_degradation: float
    confidence_interval: list[float]
    feature_importance: list[dict]
    model_validation: dict


# ---------- Schedule ----------

class DriftHistoryPoint(BaseModel):
    day: float
    drift_score: float


class ScheduleRequest(BaseModel):
    drift_history: list[DriftHistoryPoint]
    analysis_id: Optional[str] = None
    metric: Literal["accuracy", "f1", "roc_auc"] = "accuracy"
    retrain_cost: float = 500.0
    degradation_cost_per_point_per_day: float = 50.0
    horizon_days: int = Field(60, ge=7, le=365)
    # 0.15 default clears the impact model's typical zero-drift prediction
    # floor (~0.12 for accuracy; see backend/tests/test_scheduler.py), so a
    # healthy/flat drift history doesn't spuriously trip the floor_warning.
    degradation_threshold: float = Field(0.15, ge=0.001, le=0.5)


class ScheduleResponse(BaseModel):
    strategy_costs: dict
    recommended_strategy: str
    recommended_interval_days: Optional[int]
    next_retrain_in_days: int
    projected_trajectory: list[dict]
    trend_slope_per_day: float
    floor_warning: Optional[str] = None


# ---------- Longitudinal ----------

class LongitudinalRequest(BaseModel):
    n_weeks: int = Field(26, ge=6, le=104)
    drift_pattern: Literal["gradual", "sudden", "recurring"] = "gradual"
    peak_magnitude: float = Field(0.7, ge=0.05, le=1.0)
    drift_type: Literal["covariate", "concept", "mixed"] = "covariate"
    retrain_strategy: Literal["none", "fixed", "adaptive"] = "adaptive"
    fixed_interval_weeks: int = Field(8, ge=1, le=52)
    degradation_threshold: float = Field(0.15, ge=0.001, le=0.5)  # see note on ScheduleRequest above
    metric: Literal["accuracy", "f1", "roc_auc"] = "accuracy"
    seed: int = 123


class LongitudinalResponse(BaseModel):
    weeks: list[dict]
    summary: dict
    strategy_comparison: dict
