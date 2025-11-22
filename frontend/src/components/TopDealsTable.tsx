// frontend/src/components/TopDealsTable.tsx

export type TopDealItem = {
  external_id: string;
  source: string;
  address: string;
  city: string;
  state: string;
  zipcode: string;
  list_price: number;
  dscr: number;
  cash_on_cash_return: number;
  breakeven_occupancy_pct: number;
  rank_score: number;
  label: string;
  reason: string;
  lat?: number | null;
  lon?: number | null;
  dom?: number | null;
};

type Props = {
  zip: string;
  deals: TopDealItem[];
  loading: boolean;
  error: string | null;
};

function formatMoney(value: number): string {
  return `$${value.toLocaleString(undefined, {
    maximumFractionDigits: 0,
  })}`;
}

function formatPercent(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

function formatDSCR(value: number): string {
  return value.toFixed(2);
}

function labelClass(label: string): string {
  const lower = label.toLowerCase();
  if (lower === "buy") return "label-chip label-chip-buy";
  if (lower === "maybe") return "label-chip label-chip-maybe";
  return "label-chip label-chip-pass";
}

export function TopDealsTable({ zip, deals, loading, error }: Props) {
  if (!zip) {
    return (
      <div className="panel">
        <p className="muted">Enter a ZIP code to view ranked opportunities.</p>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="panel">
        <div className="panel-header">
          <h2>Top Deals</h2>
          <span className="muted">Loading market insights…</span>
        </div>
        <div className="skeleton-list">
          <div className="skeleton-card" />
          <div className="skeleton-card" />
          <div className="skeleton-card" />
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="panel">
        <div className="panel-header">
          <h2>Top Deals</h2>
        </div>
        <div className="error-banner">{error}</div>
      </div>
    );
  }

  if (!deals.length) {
    return (
      <div className="panel">
        <div className="panel-header">
          <h2>Top Deals</h2>
        </div>
        <p className="muted">
          No results for <strong>{zip}</strong> under the current criteria. Try
          adjusting your max price or ZIP code.
        </p>
      </div>
    );
  }

  return (
    <div className="panel">
      <div className="panel-header">
        <h2>Top Deals</h2>
        <span className="muted">
          {deals.length} properties ranked by cashflow and DSCR
        </span>
      </div>

      <div className="deal-list">
        {deals.map((deal) => (
          <article
            key={`${deal.source}-${deal.external_id}`}
            className="deal-card"
          >
            <div className="deal-card-main">
              <div className="deal-card-price-row">
                <div className="deal-card-price">
                  {formatMoney(deal.list_price)}
                </div>
                <span className={labelClass(deal.label)}>
                  {deal.label.toUpperCase()}
                </span>
              </div>

              <div className="deal-card-address">
                {deal.address}
                <span className="deal-card-city">
                  {deal.city}, {deal.state} {deal.zipcode}
                </span>
              </div>

              <div className="deal-card-metrics">
                <div className="metric">
                  <div className="metric-label">DSCR</div>
                  <div className="metric-value">{formatDSCR(deal.dscr)}</div>
                </div>
                <div className="metric">
                  <div className="metric-label">Cash-on-Cash</div>
                  <div className="metric-value">
                    {formatPercent(deal.cash_on_cash_return)}
                  </div>
                </div>
                <div className="metric">
                  <div className="metric-label">Breakeven Occ.</div>
                  <div className="metric-value">
                    {deal.breakeven_occupancy_pct.toFixed(1)}%
                  </div>
                </div>
                <div className="metric">
                  <div className="metric-label">Rank Score</div>
                  <div className="metric-value">
                    {deal.rank_score.toFixed(1)}
                  </div>
                </div>
                {deal.dom != null && (
                  <div className="metric">
                    <div className="metric-label">Days on Market</div>
                    <div className="metric-value">{deal.dom}</div>
                  </div>
                )}
              </div>

              <p className="deal-card-reason">{deal.reason}</p>
            </div>

            <div className="deal-card-footer">
              <span className="deal-card-source">
                Source: {deal.source.toUpperCase()}
              </span>
            </div>
          </article>
        ))}
      </div>
    </div>
  );
}
