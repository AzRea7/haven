from __future__ import annotations

from typing import Any, Optional
import sqlite3


class LeadsRepo:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def upsert_lead(
        self,
        *,
        address: str,
        city: str,
        state: str,
        zipcode: str,
        source: str,
        external_id: Optional[str],
        lat: Optional[float],
        lon: Optional[float],
        list_price: Optional[float],

        # preview/scoring fields:
        dscr: Optional[float],
        cash_on_cash_return: Optional[float],
        rank_score: Optional[float],
        label: Optional[str],
        reason: Optional[str],
        lead_score: float,
    ) -> tuple[int, str]:
        """
        Returns: (lead_id, action) where action is "created" or "updated"
        """
        cur = self.conn.cursor()

        # Define uniqueness: (source, external_id) if external_id present, else (address, zipcode)
        # This matches your current behavior where rentcast and zillow can both exist.
        if external_id:
            cur.execute(
                """
                SELECT lead_id FROM leads
                WHERE source = ? AND external_id = ?
                """,
                (source, external_id),
            )
        else:
            cur.execute(
                """
                SELECT lead_id FROM leads
                WHERE address = ? AND zipcode = ?
                """,
                (address, zipcode),
            )

        row = cur.fetchone()
        if row:
            lead_id = int(row[0])
            cur.execute(
                """
                UPDATE leads SET
                  address=?,
                  city=?,
                  state=?,
                  zipcode=?,
                  lat=?,
                  lon=?,
                  list_price=?,
                  dscr=?,
                  cash_on_cash_return=?,
                  rank_score=?,
                  label=?,
                  reason=?,
                  lead_score=?,
                  updated_at=datetime('now')
                WHERE lead_id=?
                """,
                (
                    address,
                    city,
                    state,
                    zipcode,
                    lat,
                    lon,
                    list_price,
                    dscr,
                    cash_on_cash_return,
                    rank_score,
                    label,
                    reason,
                    lead_score,
                    lead_id,
                ),
            )
            self.conn.commit()
            return lead_id, "updated"

        cur.execute(
            """
            INSERT INTO leads (
              address, city, state, zipcode,
              lat, lon,
              source, external_id,
              stage,
              created_at, updated_at,
              touches,
              owner,
              list_price,
              dscr, cash_on_cash_return, rank_score, label, reason,
              lead_score
            ) VALUES (
              ?,?,?,?,
              ?,?,
              ?,?,
              'new',
              datetime('now'), datetime('now'),
              0,
              NULL,
              ?,
              ?,?,?,?,?,
              ?
            )
            """,
            (
                address,
                city,
                state,
                zipcode,
                lat,
                lon,
                source,
                external_id,
                list_price,
                dscr,
                cash_on_cash_return,
                rank_score,
                label,
                reason,
                lead_score,
            ),
        )
        self.conn.commit()
        return int(cur.lastrowid), "created"
