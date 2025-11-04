from __future__ import annotations
from typing import Any, Dict, Optional, List
from sqlmodel import SQLModel, Field, create_engine, Session, select, Column, JSON
from datetime import datetime

class DealRow(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    ts: datetime = Field(default_factory=datetime.utcnow, index=True)
    address: str
    city: str
    state: str
    zipcode: str
    property_type: str
    payload: Dict[str, Any] = Field(sa_column=Column(JSON))   # request
    result: Dict[str, Any]  = Field(sa_column=Column(JSON))   # analysis out

class SqlDealRepository:
    def __init__(self, uri: str = "sqlite:///haven.db"):
        self.engine = create_engine(uri, echo=False)
        SQLModel.metadata.create_all(self.engine)

    def save_analysis(self, analysis: Dict[str, Any], request_payload: Dict[str, Any]) -> int:
        addr = analysis.get("address", {})
        row = DealRow(
            address=addr.get("address",""),
            city=addr.get("city",""),
            state=addr.get("state",""),
            zipcode=addr.get("zipcode",""),
            property_type=analysis.get("property_type",""),
            payload=request_payload,
            result=analysis,
        )
        with Session(self.engine) as s:
            s.add(row)
            s.commit()
            s.refresh(row)
            return row.id

    def get(self, deal_id: int) -> Optional[DealRow]:
        with Session(self.engine) as s:
            return s.get(DealRow, deal_id)

    def list_recent(self, limit: int = 50) -> List[DealRow]:
        with Session(self.engine) as s:
            stmt = select(DealRow).order_by(DealRow.ts.desc()).limit(limit)
            return list(s.exec(stmt))
