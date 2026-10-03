from __future__ import annotations
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api import datasets, drift, impact, schedule, experiments
from .core.impact_model import load_models

app = FastAPI(
    title="Drift Impact Estimator",
    description="Detects dataset drift, predicts downstream model performance "
                "degradation, and recommends minimal retraining schedules.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # demo project - lock this down to your frontend origin in production
    allow_credentials=False,  # no cookie/session auth here, so no need to allow credentialed cross-origin requests
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(datasets.router)
app.include_router(drift.router)
app.include_router(impact.router)
app.include_router(schedule.router)
app.include_router(experiments.router)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/model-status")
def model_status():
    try:
        bundles = load_models()
        return {
            "trained": True,
            "metrics_available": list(bundles.keys()),
            "validation": {m: b.validation for m, b in bundles.items()},
        }
    except FileNotFoundError:
        return {
            "trained": False,
            "metrics_available": [],
            "message": "Run `python3 train_impact_model.py` in backend/ before starting the API.",
        }
