import React from "react";

export type TopDealItem = {
  id?: string; // may be undefined if backend doesn't send it
  external_id?: string;

  address: string;
  city: string;
  state: string;
  zipcode: string;

  list_price: number;

  dscr: number;
  cash_on_cash_return: number;
  breakeven_occupancy_pct: number;
  rank_score: number;

  label: "buy" | "maybe" | "pass";
  reason: string;
  source: string;

  lat?: number | null;
  lon?: number | null;

  dom?: number | null;
};

type Props = {
  deals: TopDealItem[];
  isLoading?: boolean;
  error?: string | null;
  selectedDealId?: string | null;
  onSelectDeal?: (id: string | null) => void;
};

/* ---------- helpers ---------- */

function formatMoney(value: number): string {
  if (!Number.isFinite(value)) return "-";
  return value.toLocaleString("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0,
  });
}

function formatPct(value: number): string {
  if (!Number.isFinite(value)) return "-";
  return `${(value * 100).toFixed(1)}%`;
}

function formatDscr(value: number): string {
  if (!Number.isFinite(value)) return "-";
  return value.toFixed(2);
}

function formatRankScore(value: number): string {
  if (!Number.isFinite(value)) return "-";
  return value.toFixed(1);
}

function labelChipClass(label: TopDealItem["label"]): string {
  switch (label) {
    case "buy":
      return "label-chip label-chip-buy";
    case "maybe":
      return "label-chip label-chip-maybe";
    case "pass":
    default:
      return "label-chip label-chip-pass";
  }
}

function labelText(label: TopDealItem["label"]): string {
  switch (label) {
    case "buy":
      return "BUY";
    case "maybe":
      return "MAYBE";
    case "pass":
    default:
      return "PASS";
  }
}

/**
 * Stable unique key for React + selection.
 * Uses:
 *   1) id if present
 *   2) external_id if present
 *   3) address+zip+price as a fallback
 */
function dealKey(deal: TopDealItem): string {
  return (
    deal.id ||
    deal.external_id ||
    `${deal.address}-${deal.zipcode}-${deal.list_price}`
  );
}

/**
 * Build a Zillow listing URL from the deal.
 *
 * We assume:
 *  - external_id is the Zillow zpid (from HasData)
 *  - source contains "zillow" for Zillow-backed data
 *
 * Zillow will resolve URLs of the form:
 *   https://www.zillow.com/homedetails/<ZPID>_zpid/
 */
function buildZillowUrl(deal: TopDealItem): string | null {
  if (!deal.external_id) return null;

  const source = (deal.source || "").toLowerCase();
  if (!source.includes("zillow")) {
    // Unknown or non-Zillow source – don't try to build a URL
    return null;
  }

  return `https://www.zillow.com/homedetails/${deal.external_id}_zpid/`;
}

/* ---------- main component ---------- */

export function TopDealsTable({
  deals,
  isLoading,
  error,
  selectedDealId,
  onSelectDeal,
}: Props) {
  // loading skeleton
  if (isLoading) {
    return (
      <>
        <div className="panel-header">
          <h2>Top Deals</h2>
          <div className="muted">
            Running analysis and ranking by cashflow and DSCR…
          </div>
        </div>
        <div className="skeleton-list">
          <div className="skeleton-card" />
          <div className="skeleton-card" />
          <div className="skeleton-card" />
        </div>
      </>
    );
  }

  // error banner
  if (error) {
    return (
      <>
        <div className="panel-header">
          <h2>Top Deals</h2>
        </div>
        <div className="error-banner">{error}</div>
      </>
    );
  }

  // empty state
  if (!deals || deals.length === 0) {
    return (
      <>
        <div className="panel-header">
          <h2>Top Deals</h2>
          <div className="muted">
            No properties found for this search. Try a different ZIP or widen
            your price range.
          </div>
        </div>
      </>
    );
  }

  return (
    <>
      <div className="panel-header">
        <h2>Top Deals</h2>
        <div className="muted">
          {deals.length} properties ranked by cashflow and DSCR.
        </div>
      </div>

      <div className="deal-list">
        {deals.map((deal) => {
          const key = dealKey(deal);
          const isSelected = selectedDealId === key;

          const handleClick = () => {
            // keep selection behavior for map highlighting
            onSelectDeal?.(isSelected ? null : key);

            // open Zillow listing in new tab if we can build a URL
            const url = buildZillowUrl(deal);
            if (!isSelected && url) {
              window.open(url, "_blank", "noopener,noreferrer");
            }
          };

          return (
            <div
              key={key}
              className="deal-card"
              style={{
                ...(isSelected
                  ? {
                      borderColor: "rgba(37,99,235,0.6)",
                      boxShadow: "0 16px 40px rgba(37,99,235,0.18)",
                    }
                  : {}),
                cursor: buildZillowUrl(deal) ? "pointer" : "default",
              }}
              onClick={handleClick}
            >
              {/* price + label row */}
              <div className="deal-card-price-row">
                <div>
                  <div className="deal-card-price">
                    {formatMoney(deal.list_price)}
                  </div>
                  <div className="deal-card-address">{deal.address}</div>
                  <span className="deal-card-city">
                    {deal.city}, {deal.state} {deal.zipcode}
                  </span>
                </div>

                <div>
                  <span className={labelChipClass(deal.label)}>
                    {labelText(deal.label)}
                  </span>
                </div>
              </div>

              {/* metrics row */}
              <div className="deal-card-metrics">
                <div className="metric">
                  <div className="metric-label">DSCR</div>
                  <div className="metric-value">{formatDscr(deal.dscr)}</div>
                </div>
                <div className="metric">
                  <div className="metric-label">Cash-on-Cash</div>
                  <div className="metric-value">
                    {formatPct(deal.cash_on_cash_return)}
                  </div>
                </div>
                <div className="metric">
                  <div className="metric-label">Breakeven Occ.</div>
                  <div className="metric-value">
                    {formatPct(deal.breakeven_occupancy_pct)}
                  </div>
                </div>
                <div className="metric">
                  <div className="metric-label">Rank Score</div>
                  <div className="metric-value">
                    {formatRankScore(deal.rank_score)}
                  </div>
                </div>
              </div>

              {/* narrative + footer */}
              <p className="deal-card-reason">
                {deal.reason ||
                  (deal.label === "buy"
                    ? "High risk-adjusted score with strong coverage and returns."
                    : deal.label === "maybe"
                    ? "Workable, but requires deeper underwriting or better terms."
                    : "Negative cashflow in base case.")}
              </p>

              <div className="deal-card-footer">
                {typeof deal.dom === "number" && (
                  <span>{deal.dom} days on market · </span>
                )}
                <span className="deal-card-source">
                  Source: {deal.source || "ZILLOW_HASDATA"}
                </span>
              </div>
            </div>
          );
        })}
      </div>
    </>
  );
}
