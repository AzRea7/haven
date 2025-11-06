from fastapi import FastAPI, HTTPException

from haven.adapters.sql_repo import DealRow, SqlDealRepository
from haven.api.schemas import AnalyzeRequest, AnalyzeResponse
from haven.services.deal_analyzer import analyze_deal_with_defaults

app = FastAPI(title="Haven Deal Analysis API")
_repo = SqlDealRepository("sqlite:///haven.db")

@app.post("/analyze")
def analyze_endpoint(payload: dict):
    try:
        # use the same default estimator but pass our SQL repo
        return analyze_deal_with_defaults.__wrapped__(  # call underlying fn so we can pass repo
            raw_payload=payload,  # type: ignore
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

# add dependency-friendly variant so we can use our repo instance
@app.post("/analyze2", response_model=AnalyzeResponse)
def analyze_endpoint2(payload: AnalyzeRequest):
    from haven.services.deal_analyzer import _default_estimator, analyze_deal
    try:
        return analyze_deal(payload.model_dump(), rent_estimator=_default_estimator, repo=_repo)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

@app.get("/deals", response_model=list[dict])
def list_deals(limit: int = 50):
    rows: list[DealRow] = _repo.list_recent(limit=limit)
    return [r.result | {"deal_id": r.id, "ts": r.ts.isoformat()} for r in rows]

@app.get("/deals/{deal_id}", response_model=dict)
def get_deal(deal_id: int):
    row = _repo.get(deal_id)
    if not row:
        raise HTTPException(404, "deal not found")
    return row.result | {"deal_id": row.id, "ts": row.ts.isoformat()}
