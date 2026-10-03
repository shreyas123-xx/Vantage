from __future__ import annotations
import pandas as pd
from fastapi import APIRouter, UploadFile, File, HTTPException
from io import StringIO

from ..models.schemas import SyntheticDatasetRequest, SyntheticDatasetResponse
from ..core.synthetic_data import generate_matched_pair, batch_to_records
from .. import storage

router = APIRouter(prefix="/api/datasets", tags=["datasets"])


@router.post("/synthetic", response_model=SyntheticDatasetResponse)
def create_synthetic_dataset(req: SyntheticDatasetRequest):
    ref, cur = generate_matched_pair(
        n_reference=req.n_reference,
        n_current=req.n_current,
        seed=req.seed,
        magnitude=req.magnitude,
        drift_type=req.drift_type,
    )
    reference_id = storage.store_dataset(ref)
    current_id = storage.store_dataset(cur)
    return SyntheticDatasetResponse(
        reference_preview=batch_to_records(ref, n_preview=15),
        current_preview=batch_to_records(cur, n_preview=15),
        reference_id=reference_id,
        current_id=current_id,
        n_reference=len(ref.X),
        n_current=len(cur.X),
    )


@router.post("/upload")
async def upload_dataset(file: UploadFile = File(...), label_column: str | None = None):
    """Accepts a CSV. If label_column is provided and present, it's split out
    as y; otherwise the whole file is treated as feature-only (drift analysis
    still works fine without labels - only the impact model's *training*
    simulation needs labels, not live drift analysis)."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "Only .csv files are supported")
    content = (await file.read()).decode("utf-8", errors="replace")
    try:
        df = pd.read_csv(StringIO(content))
    except Exception as e:
        raise HTTPException(400, f"Could not parse CSV: {e}")
    if df.empty:
        raise HTTPException(400, "Uploaded CSV has no rows")

    y = None
    if label_column and label_column in df.columns:
        y = df[label_column].to_numpy()
        df = df.drop(columns=[label_column])

    from ..core.synthetic_data import SyntheticBatch
    import numpy as np
    batch = SyntheticBatch(X=df, y=y if y is not None else np.zeros(len(df)))
    dataset_id = storage.store_dataset(batch)
    return {
        "dataset_id": dataset_id,
        "n_rows": len(df),
        "columns": list(df.columns),
        "preview": df.head(15).round(4).to_dict(orient="records"),
    }
