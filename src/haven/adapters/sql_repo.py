from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Any, cast

from sqlmodel import JSON, Column, Field, Session, SQLModel, create_engine, desc, select

from haven.domain.ports import DealRowLike


class DealRow(SQLModel, table=True): # type: ignore[call-arg]
    id: int | None = Field(default=None, primary_key=True)
    ts: datetime = Field(default_factory=datetime.utcnow, index=True)
    address: str
    city: str
    state: str
    zipcode: str
    property_type: str
    payload: dict[str, Any] = Field(sa_column=Column(JSON))   # request
    result: dict[str, Any]  = Field(sa_column=Column(JSON))   # analysis out
    request_payload: dict[str, Any] | None = Field(default=None, sa_column=Column(JSON))

class SqlDealRepository:
    def __init__(self, uri: str = "sqlite:///haven.db"):
        self.engine = create_engine(uri, echo=False)
        SQLModel.metadata.create_all(self.engine)

    def save_analysis(
        self,
        analysis: dict[str, Any],
        request_payload: dict[str, Any] | None = None,
    ) -> int:
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
            assert row.id is not None
            return int(row.id)

    def list_recent(self, limit: int = 50) -> Sequence[DealRowLike]:
        with Session(self.engine) as s:
            stmt = select(DealRow).order_by(desc(DealRow.ts)).limit(limit)
            rows = list(s.exec(stmt))
            return cast(Sequence[DealRowLike], rows)

    def get(self, deal_id: int) -> DealRowLike | None:
        with Session(self.engine) as s:
            row = s.get(DealRow, deal_id)
            return cast(DealRowLike | None, row)
