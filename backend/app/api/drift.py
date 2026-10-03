from __future__ import annotations
from fastapi import APIRouter, HTTPException

from ..models.schemas import DriftAnalyzeRequest, DriftAnalyzeResponse
from ..core.drift_metrics import analyze_drift, drift_report_to_feature_vector
from .. import storage

router = APIRouter(prefix="/api/drift", tags=["drift"])


@router.post("/analyze", response_model=DriftAnalyzeResponse)
def analyze(req: DriftAnalyzeRequest):
    if not req.reference_id or not req.current_id:
        raise HTTPException(400, "reference_id and current_id are required")
    try:
        reference = storage.get_dataset(req.reference_id)
        current = storage.get_dataset(req.current_id)
    except KeyError as e:
        raise HTTPException(404, str(e))

    report = analyze_drift(reference.X, current.X)
    feature_vector = drift_report_to_feature_vector(report)
    analysis_id = storage.store_analysis({"report": report, "feature_vector": feature_vector})

    payload = report.to_dict()
    payload["drift_feature_vector"] = feature_vector
    payload["analysis_id"] = analysis_id
    return payload
