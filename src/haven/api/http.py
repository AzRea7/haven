from fastapi import FastAPI, HTTPException
from haven.services.deal_analyzer import analyze_deal_with_defaults

app = FastAPI(title="Haven Deal Analysis API")

@app.post("/analyze")
def analyze_endpoint(payload: dict):
    try:
        return analyze_deal_with_defaults(payload)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
