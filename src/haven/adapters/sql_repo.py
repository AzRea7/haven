# src/haven/adapters/sql_repo.py
from __future__ import annotations

from datetime import datetime
from typing import Any, Iterable, Sequence

from sqlmodel import JSON, Column, Field, Session, SQLModel, create_engine, select


# ---------- Deals (existing behavior) ----------

class DealRow(SQLModel, table=True):
    __tablename__ = "deals"

    id: int | None = Field(default=None, primary_key=True)
    ts: datetime = Field(default_factory=datetime.utcnow, index=True)

    address: str
    city: str
    state: str
    zipcode: str
    property_type: str

    payload: dict[str, Any] = Field(sa_column=Column(JSON))
    result: dict[str, Any] = Field(sa_column=Column(JSON))


class SqlDealRepository:
    def __init__(self, uri: str = "sqlite:///haven.db"):
        self.engine = create_engine(uri, echo=False)
        SQLModel.metadata.create_all(self.engine)

    def save_analysis(self, analysis: dict[str, Any], request_payload: dict[str, Any]) -> int:
        addr = analysis.get("address", {})
        row = DealRow(
            address=addr.get("address", ""),
            city=addr.get("city", ""),
            state=addr.get("state", ""),
            zipcode=addr.get("zipcode", ""),
            property_type=analysis.get("property_type", ""),
            payload=request_payload,
            result=analysis,
        )
        with Session(self.engine) as session:
            session.add(row)
            session.commit()
            session.refresh(row)
            return int(row.id)  # type: ignore[arg-type]

    def get(self, deal_id: int) -> DealRow | None:
        with Session(self.engine) as session:
            return session.get(DealRow, deal_id)

    def list_recent(self, limit: int = 50) -> list[DealRow]:
        with Session(self.engine) as session:
            stmt = select(DealRow).order_by(DealRow.ts.desc()).limit(limit)
            return list(session.exec(stmt))


# ---------- Property storage ----------

class PropertyRow(SQLModel, table=True):
    __tablename__ = "properties"

    id: int | None = Field(default=None, primary_key=True)
    ts: datetime = Field(default_factory=datetime.utcnow, index=True)

    source: str = Field(index=True)
    external_id: str | None = Field(default=None, index=True)

    address: str
    city: str
    state: str
    zipcode: str = Field(index=True)

    lat: float | None = None
    lon: float | None = None

    beds: float | None = None
    baths: float | None = None
    sqft: float | None = None
    year_built: int | None = None

    list_price: float | None = Field(default=None, index=True)
    list_date: datetime | None = Field(default=None, index=True)

    property_type: str | None = Field(default=None, index=True)

    raw: dict[str, Any] = Field(sa_column=Column(JSON))


class SqlPropertyRepository:
    def __init__(self, uri: str = "sqlite:///haven.db"):
        self.engine = create_engine(uri, echo=False)
        SQLModel.metadata.create_all(self.engine)

    def upsert_many(self, items: Iterable[dict[str, Any]]) -> int:
        written = 0
        with Session(self.engine) as session:
            for item in items:
                if not item:
                    continue

                source = (item.get("source") or "unknown").strip()
                external_id = (item.get("external_id") or "").strip() or None

                row: PropertyRow | None = None
                if external_id:
                    stmt = select(PropertyRow).where(
                        PropertyRow.source == source,
                        PropertyRow.external_id == external_id,
                    )
                    row = session.exec(stmt).first()

                if row:
                    for field in [
                        "address", "city", "state", "zipcode",
                        "lat", "lon",
                        "beds", "baths", "sqft", "year_built",
                        "list_price", "property_type",
                    ]:
                        if field in item and item[field] is not None:
                            setattr(row, field, item[field])  # type: ignore[index]

                    raw = item.get("raw")
                    if isinstance(raw, dict) and raw:
                        row.raw = raw
                else:
                    row = PropertyRow(
                        source=source,
                        external_id=external_id,
                        address=str(item.get("address", "")),
                        city=str(item.get("city", "")),
                        state=str(item.get("state", "")),
                        zipcode=str(item.get("zipcode", "")),
                        lat=float(item.get("lat")) if item.get("lat") is not None else None,
                        lon=float(item.get("lon")) if item.get("lon") is not None else None,
                        beds=float(item.get("beds")) if item.get("beds") is not None else None,
                        baths=float(item.get("baths")) if item.get("baths") is not None else None,
                        sqft=float(item.get("sqft")) if item.get("sqft") is not None else None,
                        year_built=int(item.get("year_built")) if item.get("year_built") else None,
                        list_price=float(item.get("list_price")) if item.get("list_price") else None,
                        property_type=str(item.get("property_type", "")) or None,
                        raw=item.get("raw") or {},
                    )
                    session.add(row)

                written += 1
            session.commit()
        return written

    def search(self, zipcode: str, max_price: float | None = None, limit: int = 200) -> list[dict[str, Any]]:
        with Session(self.engine) as session:
            stmt = select(PropertyRow).where(PropertyRow.zipcode == zipcode)
            if max_price is not None:
                stmt = stmt.where(PropertyRow.list_price <= max_price)
            stmt = stmt.order_by(PropertyRow.list_price).limit(limit)
            rows = list(session.exec(stmt))

        out: list[dict[str, Any]] = []
        for r in rows:
            out.append(
                dict(
                    external_id=r.external_id or "",
                    source=r.source,
                    address=r.address,
                    city=r.city,
                    state=r.state,
                    zipcode=r.zipcode,
                    lat=r.lat,
                    lon=r.lon,
                    beds=r.beds,
                    baths=r.baths,
                    sqft=r.sqft,
                    year_built=r.year_built,
                    list_price=r.list_price,
                    list_date=r.list_date.isoformat() if r.list_date else None,
                    property_type=r.property_type or "",
                    raw=r.raw or {},
                )
            )
        return out


# ---------- Leads + Lead Events ----------

class LeadRow(SQLModel, table=True):
    __tablename__ = "leads"

    lead_id: int | None = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    updated_at: datetime = Field(default_factory=datetime.utcnow, index=True)

    # Identity from upstream (same pattern as properties)
    source: str = Field(index=True)
    external_id: str | None = Field(default=None, index=True)

    address: str
    city: str
    state: str
    zipcode: str = Field(index=True)

    lat: float | None = None
    lon: float | None = None

    beds: float | None = None
    baths: float | None = None
    sqft: float | None = None
    property_type: str | None = Field(default=None, index=True)
    list_price: float | None = Field(default=None, index=True)

    # Lead funnel fields
    stage: str = Field(default="new", index=True)
    owner: str | None = Field(default=None, index=True)

    touches: int = Field(default=0)
    last_contacted_at: datetime | None = Field(default=None, index=True)

    # Ranking target (0..100)
    lead_score: float = Field(default=0.0, index=True)

    # Optional underwriting preview
    dscr: float | None = None
    cash_on_cash_return: float | None = None
    rank_score: float | None = None
    label: str | None = None
    reason: str | None = None  # ✅ ADD THIS

    # Full snapshot of property record + raw payload
    snapshot: dict[str, Any] = Field(sa_column=Column(JSON))


class LeadEventRow(SQLModel, table=True):
    __tablename__ = "lead_events"

    event_id: int | None = Field(default=None, primary_key=True)
    ts: datetime = Field(default_factory=datetime.utcnow, index=True)

    lead_id: int = Field(index=True)

    event_type: str
    note: str | None = None

    meta: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))


class SqlLeadRepository:
    def __init__(self, uri: str = "sqlite:///haven.db"):
        self.engine = create_engine(uri, echo=False)
        SQLModel.metadata.create_all(self.engine)

    def _find_existing(
        self,
        session: Session,
        *,
        source: str,
        external_id: str | None,
        address: str,
        zipcode: str,
    ) -> LeadRow | None:
        if external_id:
            stmt = select(LeadRow).where(
                LeadRow.source == source,
                LeadRow.external_id == external_id,
            )
            row = session.exec(stmt).first()
            if row:
                return row

        stmt2 = select(LeadRow).where(
            LeadRow.source == source,
            LeadRow.address == address,
            LeadRow.zipcode == zipcode,
        )
        return session.exec(stmt2).first()

    def upsert_from_properties(
        self,
        *,
        properties: Sequence[dict[str, Any]],
        compute_preview_fn,  # callable(prop_rec)->dict
    ) -> dict[str, int]:
        created = 0
        updated = 0

        with Session(self.engine) as session:
            for p in properties:
                source = str(p.get("source") or "unknown")
                external_id = (p.get("external_id") or "").strip() or None
                address = str(p.get("address") or "")
                zipcode = str(p.get("zipcode") or "")

                if not address or not zipcode:
                    continue

                existing = self._find_existing(
                    session,
                    source=source,
                    external_id=external_id,
                    address=address,
                    zipcode=zipcode,
                )

                preview = compute_preview_fn(p) or {}

                def _set_preview_fields(obj: LeadRow) -> None:
                    # Always set lead_score if provided
                    if preview.get("lead_score") is not None:
                        obj.lead_score = float(preview.get("lead_score") or 0.0)

                    # Persist preview columns
                    for f in ["dscr", "cash_on_cash_return", "rank_score", "label", "reason"]:
                        if preview.get(f) is not None:
                            setattr(obj, f, preview.get(f))

                if existing:
                    existing.updated_at = datetime.utcnow()
                    existing.snapshot = p

                    for field in ["city", "state", "lat", "lon", "beds", "baths", "sqft", "property_type", "list_price"]:
                        if p.get(field) is not None:
                            setattr(existing, field, p.get(field))

                    _set_preview_fields(existing)
                    updated += 1
                else:
                    row = LeadRow(
                        source=source,
                        external_id=external_id,
                        address=address,
                        city=str(p.get("city") or ""),
                        state=str(p.get("state") or ""),
                        zipcode=zipcode,
                        lat=p.get("lat"),
                        lon=p.get("lon"),
                        beds=p.get("beds"),
                        baths=p.get("baths"),
                        sqft=p.get("sqft"),
                        property_type=p.get("property_type") or None,
                        list_price=p.get("list_price"),
                        stage="new",
                        touches=0,
                        snapshot=p,
                    )
                    _set_preview_fields(row)

                    session.add(row)
                    created += 1

            session.commit()

        return {"created": created, "updated": updated}

    def list_top_leads(
        self,
        *,
        zipcode: str,
        limit: int = 200,
        stage: str | None = None,
    ) -> list[LeadRow]:
        with Session(self.engine) as session:
            stmt = select(LeadRow).where(LeadRow.zipcode == zipcode)

            if stage:
                stmt = stmt.where(LeadRow.stage == stage)

            stmt = stmt.order_by(LeadRow.lead_score.desc()).limit(limit)
            return list(session.exec(stmt))

    def add_event(
        self,
        *,
        lead_id: int,
        event_type: str,
        note: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> LeadEventRow:
        meta = meta or {}
        now = datetime.utcnow()

        with Session(self.engine) as session:
            lead = session.get(LeadRow, lead_id)
            if not lead:
                raise ValueError("lead not found")

            if event_type in {"contacted", "appointment", "contract", "closed_won", "closed_lost", "dead"}:
                lead.stage = event_type
                lead.updated_at = now

            if event_type in {"attempt", "contacted"}:
                lead.touches = int(lead.touches or 0) + 1
                lead.last_contacted_at = now
                lead.updated_at = now

            ev = LeadEventRow(
                lead_id=lead_id,
                event_type=event_type,
                note=note,
                meta=meta,
            )
            session.add(ev)
            session.add(lead)
            session.commit()
            session.refresh(ev)
            return ev
