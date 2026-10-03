from __future__ import annotations
from fastapi import APIRouter, HTTPException

from ..models.schemas import ImpactPredictRequest, ImpactPredictResponse
from ..core.impact_model import predict_degradation
from .. import storage

router = APIRouter(prefix="/api/impact", tags=["impact"])


@router.post("/predict", response_model=ImpactPredictResponse)
def predict(req: ImpactPredictRequest):
    if req.drift_feature_vector:
        feature_vector = req.drift_feature_vector
    elif req.analysis_id:
        try:
            feature_vector = storage.get_analysis(req.analysis_id)["feature_vector"]
        except KeyError as e:
            raise HTTPException(404, str(e))
    else:
        raise HTTPException(400, "Provide either analysis_id or drift_feature_vector")

    try:
        result = predict_degradation(feature_vector, metric=req.metric)
    except FileNotFoundError as e:
        raise HTTPException(503, str(e))
    return result
