# src/haven/api/http.py
from typing import Any

from fastapi import FastAPI, HTTPException

from haven.adapters.sql_repo import DealRow, SqlDealRepository
from haven.services.deal_analyzer import analyze_deal_with_defaults
from .schemas import AnalyzeRequest, AnalyzeResponse

app = FastAPI()
_repo = SqlDealRepository()

@app.post("/analyze")
def analyze_endpoint(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        return analyze_deal_with_defaults(raw_payload=payload)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e  # B904

@app.post("/analyze2", response_model=AnalyzeResponse)
def analyze_endpoint2(payload: AnalyzeRequest) -> AnalyzeResponse:
    from haven.services.deal_analyzer import _default_estimator, analyze_deal
    try:
        return AnalyzeResponse(
            **analyze_deal(payload.model_dump(), rent_estimator=_default_estimator, repo=_repo)
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e  # B904

@app.get("/deals", response_model=list[dict])
def list_deals(limit: int = 50) -> list[dict]:
    rows: list[DealRow] = _repo.list_recent(limit=limit)
    return [r.result | {"deal_id": r.id, "ts": r.ts.isoformat()} for r in rows]

@app.get("/deals/{deal_id}", response_model=dict)
def get_deal(deal_id: int) -> dict:
    row = _repo.get(deal_id)
    if not row:
        raise HTTPException(404, "deal not found")
    return row.result | {"deal_id": row.id, "ts": row.ts.isoformat()}
