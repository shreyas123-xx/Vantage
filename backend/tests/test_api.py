import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_model_status_trained():
    r = client.get("/api/model-status")
    assert r.status_code == 200
    body = r.json()
    assert body["trained"] is True
    assert set(body["metrics_available"]) == {"accuracy", "f1", "roc_auc"}


@pytest.fixture
def analysis():
    ds = client.post("/api/datasets/synthetic", json={
        "n_reference": 600, "n_current": 400, "magnitude": 0.5,
        "drift_type": "covariate", "seed": 3,
    }).json()
    return client.post("/api/drift/analyze", json={
        "reference_id": ds["reference_id"], "current_id": ds["current_id"],
    }).json()


def test_full_pipeline_dataset_to_schedule(analysis):
    assert analysis["n_features"] == 8
    assert 0 <= analysis["c2st_auc"] <= 1

    impact = client.post("/api/impact/predict", json={
        "analysis_id": analysis["analysis_id"], "metric": "accuracy",
    }).json()
    assert "predicted_degradation" in impact
    assert len(impact["confidence_interval"]) == 2
    assert impact["confidence_interval"][0] <= impact["confidence_interval"][1]

    schedule = client.post("/api/schedule/recommend", json={
        "analysis_id": analysis["analysis_id"],
        "drift_history": [{"day": 0, "drift_score": 0.05}, {"day": 10, "drift_score": 0.2}],
        "metric": "accuracy", "horizon_days": 30,
    }).json()
    assert schedule["recommended_strategy"] in schedule["strategy_costs"] or \
        schedule["recommended_strategy"].startswith("fixed_every")


def test_drift_analyze_missing_ids_returns_400():
    r = client.post("/api/drift/analyze", json={})
    assert r.status_code == 400


def test_drift_analyze_unknown_id_returns_404():
    r = client.post("/api/drift/analyze", json={"reference_id": "nope", "current_id": "also-nope"})
    assert r.status_code == 404


def test_impact_predict_requires_source():
    r = client.post("/api/impact/predict", json={"metric": "accuracy"})
    assert r.status_code == 400


def test_longitudinal_smoke():
    r = client.post("/api/experiments/longitudinal", json={
        "n_weeks": 8, "drift_pattern": "gradual", "peak_magnitude": 0.5,
        "drift_type": "covariate", "retrain_strategy": "adaptive",
    })
    assert r.status_code == 200
    body = r.json()
    assert len(body["weeks"]) == 8
    assert set(body["strategy_comparison"].keys()) == {"none", "fixed", "adaptive"}


def test_longitudinal_concept_drift_has_weak_correlation():
    """Documents the structural blind spot end-to-end through the API."""
    r = client.post("/api/experiments/longitudinal", json={
        "n_weeks": 10, "drift_pattern": "gradual", "peak_magnitude": 0.9,
        "drift_type": "concept", "retrain_strategy": "none",
    })
    assert r.status_code == 200
    corr = r.json()["summary"]["correlation_predicted_vs_true"]
    # Correlation may be None (near-zero variance) or low; either is consistent
    # with concept drift being invisible to input-distribution drift metrics.
    assert corr is None or corr < 0.7


def test_upload_rejects_non_csv():
    r = client.post("/api/datasets/upload", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert r.status_code == 400
