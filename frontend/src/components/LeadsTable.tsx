import React from "react";

export type LeadStage =
  | "new"
  | "contacted"
  | "appointment"
  | "contract"
  | "closed_won"
  | "closed_lost"
  | "dead";

export type LeadItem = {
  lead_id: number;

  address: string;
  city: string;
  state: string;
  zipcode: string;

  lat?: number | null;
  lon?: number | null;

  source: string;
  external_id?: string | null;

  lead_score: number; // 0..100
  stage: LeadStage;

  created_at: string;
  updated_at: string;

  last_contacted_at?: string | null;
  touches: number;

  // optional preview
  list_price?: number | null;
  dscr?: number | null;
  cash_on_cash_return?: number | null;
  rank_score?: number | null;
  label?: string | null;
};

type Props = {
  leads: LeadItem[];
  isLoading?: boolean;
  error?: string | null;
  onEvent?: (leadId: number, eventType: string) => void;
};

function formatMoney(v?: number | null): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return "-";
  return v.toLocaleString("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0,
  });
}

function formatPct(v?: number | null): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return "-";
  return `${(v * 100).toFixed(1)}%`;
}

function formatDate(s?: string | null): string {
  if (!s) return "-";
  const d = new Date(s);
  if (Number.isNaN(d.getTime())) return "-";
  return d.toLocaleString();
}

function stageChip(stage: LeadStage): { cls: string; text: string } {
  switch (stage) {
    case "new":
      return { cls: "label-chip label-chip-maybe", text: "NEW" };
    case "contacted":
      return { cls: "label-chip label-chip-buy", text: "CONTACTED" };
    case "appointment":
      return { cls: "label-chip label-chip-buy", text: "APPOINTMENT" };
    case "contract":
      return { cls: "label-chip label-chip-buy", text: "CONTRACT" };
    case "closed_won":
      return { cls: "label-chip label-chip-buy", text: "CLOSED WON" };
    case "closed_lost":
      return { cls: "label-chip label-chip-pass", text: "CLOSED LOST" };
    case "dead":
    default:
      return { cls: "label-chip label-chip-pass", text: "DEAD" };
  }
}

export function LeadsTable({ leads, isLoading, error, onEvent }: Props) {
  if (isLoading) {
    return (
      <>
        <div className="panel-header">
          <h2>Top Leads</h2>
          <div className="muted">Ranking leads by conversion likelihood…</div>
        </div>
        <div className="skeleton-list">
          <div className="skeleton-card" />
          <div className="skeleton-card" />
          <div className="skeleton-card" />
        </div>
      </>
    );
  }

  if (error) {
    return (
      <>
        <div className="panel-header">
          <h2>Top Leads</h2>
        </div>
        <div className="error-banner">{error}</div>
      </>
    );
  }

  if (!leads || leads.length === 0) {
    return (
      <>
        <div className="panel-header">
          <h2>Top Leads</h2>
          <div className="muted">
            No leads found. Click “Generate Leads” first.
          </div>
        </div>
      </>
    );
  }

  return (
    <>
      <div className="panel-header">
        <h2>Top Leads</h2>
        <div className="muted">{leads.length} leads ranked by lead score.</div>
      </div>

      <div className="deal-list">
        {leads.map((l) => {
          const chip = stageChip(l.stage);
          return (
            <div key={l.lead_id} className="deal-card">
              <div className="deal-card-price-row">
                <div>
                  <div className="deal-card-price">
                    {l.lead_score.toFixed(1)} / 100
                  </div>
                  <div className="deal-card-address">{l.address}</div>
                  <span className="deal-card-city">
                    {l.city}, {l.state} {l.zipcode}
                  </span>
                </div>
                <div>
                  <span className={chip.cls}>{chip.text}</span>
                </div>
              </div>

              <div className="deal-card-metrics">
                <div className="metric">
                  <div className="metric-label">List Price</div>
                  <div className="metric-value">
                    {formatMoney(l.list_price ?? null)}
                  </div>
                </div>
                <div className="metric">
                  <div className="metric-label">DSCR</div>
                  <div className="metric-value">
                    {l.dscr === null || l.dscr === undefined
                      ? "-"
                      : l.dscr.toFixed(2)}
                  </div>
                </div>
                <div className="metric">
                  <div className="metric-label">CoC</div>
                  <div className="metric-value">
                    {formatPct(l.cash_on_cash_return ?? null)}
                  </div>
                </div>
                <div className="metric">
                  <div className="metric-label">Touches</div>
                  <div className="metric-value">{l.touches ?? 0}</div>
                </div>
              </div>

              <p className="deal-card-reason">
                <strong>Last touched:</strong>{" "}
                {formatDate(l.last_contacted_at ?? null)}
              </p>

              <div
                className="deal-card-footer"
                style={{ display: "flex", gap: 8, flexWrap: "wrap" }}
              >
                <button
                  className="secondary-button"
                  onClick={() => onEvent?.(l.lead_id, "attempt")}
                >
                  Attempt
                </button>
                <button
                  className="secondary-button"
                  onClick={() => onEvent?.(l.lead_id, "contacted")}
                >
                  Contacted
                </button>
                <button
                  className="secondary-button"
                  onClick={() => onEvent?.(l.lead_id, "appointment")}
                >
                  Appointment
                </button>
                <button
                  className="secondary-button"
                  onClick={() => onEvent?.(l.lead_id, "contract")}
                >
                  Contract
                </button>
                <button
                  className="danger-button"
                  onClick={() => onEvent?.(l.lead_id, "dead")}
                >
                  Mark Dead
                </button>
              </div>
            </div>
          );
        })}
      </div>
    </>
  );
}
