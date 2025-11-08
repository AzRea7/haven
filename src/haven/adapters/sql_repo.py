# src/haven/adapters/sql_repo.py
from __future__ import annotations

from datetime import datetime
from typing import Any, Iterable

from sqlmodel import JSON, Column, Field, Session, SQLModel, create_engine, select

from haven.domain.ports import DealRepository, PropertyRecord, PropertyRepository


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

    # Original request (inputs / scenario)
    payload: dict[str, Any] = Field(sa_column=Column(JSON))
    # Result of analysis (outputs / metrics)
    result: dict[str, Any] = Field(sa_column=Column(JSON))


class SqlDealRepository(DealRepository):
    """
    SQL-backed DealRepository for persisting analyses.
    """

    def __init__(self, uri: str = "sqlite:///haven.db"):
        self.engine = create_engine(uri, echo=False)
        SQLModel.metadata.create_all(self.engine)

    def save_analysis(
        self,
        analysis: dict[str, Any],
        request_payload: dict[str, Any],
    ) -> int:
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
            return row.id

    # convenience helpers (not strictly required for the pipeline)

    def get(self, deal_id: int) -> DealRow | None:
        with Session(self.engine) as session:
            return session.get(DealRow, deal_id)

    def list_recent(self, limit: int = 50) -> list[DealRow]:
        with Session(self.engine) as session:
            stmt = select(DealRow).order_by(DealRow.ts.desc()).limit(limit)
            return list(session.exec(stmt))


# ---------- Property storage (new) ----------


class PropertyRow(SQLModel, table=True):
    __tablename__ = "properties"

    id: int | None = Field(default=None, primary_key=True)
    ts: datetime = Field(default_factory=datetime.utcnow, index=True)

    # Stable identity from upstream
    source: str = Field(index=True)
    external_id: str | None = Field(default=None, index=True)

    # Address
    address: str
    city: str
    state: str
    zipcode: str = Field(index=True)

    # Geo
    lat: float | None = None
    lon: float | None = None

    # Physical
    beds: float | None = None
    baths: float | None = None
    sqft: float | None = None
    year_built: int | None = None

    # Pricing
    list_price: float | None = Field(default=None, index=True)
    list_date: datetime | None = Field(default=None, index=True)

    property_type: str | None = Field(default=None, index=True)

    # Full original payload for debugging / feature expansion
    raw: dict[str, Any] = Field(sa_column=Column(JSON))


class SqlPropertyRepository(PropertyRepository):
    """
    SQL-backed PropertyRepository.

    Holds curated listing records to:
      - feed the ML feature builder
      - back /properties/search and /top-deals APIs
    """

    def __init__(self, uri: str = "sqlite:///haven.db"):
        self.engine = create_engine(uri, echo=False)
        SQLModel.metadata.create_all(self.engine)

    def upsert_many(self, items: Iterable[PropertyRecord]) -> int:
        written = 0

        with Session(self.engine) as session:
            for item in items:
                if not item:
                    continue

                source = (item.get("source") or "zillow_hasdata").strip()
                external_id = (item.get("external_id") or "").strip() or None

                # Try to find an existing row by stable identity
                row = None
                if external_id:
                    stmt = select(PropertyRow).where(
                        PropertyRow.source == source,
                        PropertyRow.external_id == external_id,
                    )
                    row = session.exec(stmt).first()

                if row:
                    # Update key fields if new values are present
                    for field in [
                        "address",
                        "city",
                        "state",
                        "zipcode",
                        "lat",
                        "lon",
                        "beds",
                        "baths",
                        "sqft",
                        "year_built",
                        "list_price",
                        "property_type",
                    ]:
                        if field in item and item[field] is not None:
                            setattr(row, field, item[field])  # type: ignore[index]

                    # Always refresh raw snapshot if provided
                    raw = item.get("raw")
                    if isinstance(raw, dict) and raw:
                        row.raw = raw
                else:
                    # New row
                    row = PropertyRow(
                        source=source,
                        external_id=external_id,
                        address=str(item.get("address", "")),
                        city=str(item.get("city", "")),
                        state=str(item.get("state", "")),
                        zipcode=str(item.get("zipcode", "")),
                        lat=float(item.get("lat", 0.0)) if item.get("lat") is not None else None,
                        lon=float(item.get("lon", 0.0)) if item.get("lon") is not None else None,
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

    def search(
        self,
        zipcode: str,
        max_price: float | None = None,
        limit: int = 200,
    ) -> list[PropertyRecord]:
        with Session(self.engine) as session:
            stmt = select(PropertyRow).where(PropertyRow.zipcode == zipcode)

            if max_price is not None:
                stmt = stmt.where(PropertyRow.list_price <= max_price)

            stmt = stmt.order_by(PropertyRow.list_price).limit(limit)
            rows = list(session.exec(stmt))

        out: list[PropertyRecord] = []
        for r in rows:
            out.append(
                PropertyRecord(
                    external_id=r.external_id or "",
                    source=r.source,
                    address=r.address,
                    city=r.city,
                    state=r.state,
                    zipcode=r.zipcode,
                    lat=float(r.lat) if r.lat is not None else 0.0,
                    lon=float(r.lon) if r.lon is not None else 0.0,
                    beds=float(r.beds) if r.beds is not None else 0.0,
                    baths=float(r.baths) if r.baths is not None else 0.0,
                    sqft=float(r.sqft) if r.sqft is not None else 0.0,
                    year_built=int(r.year_built) if r.year_built is not None else 0,
                    list_price=float(r.list_price) if r.list_price is not None else 0.0,
                    list_date=r.list_date.isoformat() if r.list_date else "",
                    property_type=r.property_type or "",
                    raw=r.raw or {},
                )
            )

        return out
