// frontend/src/App.tsx
import React, { useEffect, useState } from "react";
import { TopDealsTable, type TopDealItem } from "./components/TopDealsTable";
import { TopDealsMap } from "./components/TopDealsMap";
import { AuctionLinksPanel } from "./components/AuctionLinksPanel";

type Strategy = "rental" | "flip";

// Lead shape from backend
type LeadItem = {
  lead_id: number;
  address: string;
  city: string;
  state: string;
  zipcode: string;
  lat?: number | null;
  lon?: number | null;
  source: string;
  external_id?: string | null;

  lead_score: number;
  stage:
    | "new"
    | "contacted"
    | "appointment"
    | "contract"
    | "closed_won"
    | "closed_lost";

  created_at: string;
  updated_at: string;
  last_contacted_at?: string | null;
  touches: number;
  owner?: string | null;

  list_price?: number | null;
  dscr?: number | null;
  cash_on_cash_return?: number | null;
  rank_score?: number | null;
  label?: "buy" | "maybe" | "pass" | null;
  reason?: string | null;
};

function apiBase(): string {
  // If you use vite proxy, VITE_API_BASE_URL can be empty.
  // If proxy breaks, set VITE_API_BASE_URL=http://127.0.0.1:8000
  return (import.meta as any).env?.VITE_API_BASE_URL?.trim?.() || "";
}

function leadToTopDeal(lead: LeadItem): TopDealItem {
  // Temporary mapping so the existing TopDealsTable/Map works today.
  return {
    id: String(lead.lead_id),
    external_id: lead.external_id || undefined,

    address: lead.address,
    city: lead.city,
    state: lead.state,
    zipcode: lead.zipcode,

    list_price: Number(lead.list_price ?? 0),

    dscr: Number(lead.dscr ?? 0),
    cash_on_cash_return: Number(lead.cash_on_cash_return ?? 0),
    breakeven_occupancy_pct: 0, // not in leads yet
    rank_score: Number(lead.rank_score ?? lead.lead_score ?? 0),

    label: (lead.label ?? "maybe") as any,
    reason:
      lead.reason ??
      `Stage: ${lead.stage} · Touches: ${lead.touches} · LeadScore: ${(
        lead.lead_score ?? 0
      ).toFixed(2)}`,
    source: lead.source,

    lat: lead.lat ?? undefined,
    lon: lead.lon ?? undefined,
    dom: null,
  };
}

function App() {
  const [zip, setZip] = useState("48009");
  const [activeZip, setActiveZip] = useState("48009");

  const [maxPriceInput, setMaxPriceInput] = useState<string>("800000");
  const [activeMaxPrice, setActiveMaxPrice] = useState<number>(800000);

  const [strategy, setStrategy] = useState<Strategy>("rental");

  const [minDscr, setMinDscr] = useState<string>("");
  const [minCoc, setMinCoc] = useState<string>("");
  const [minLabel, setMinLabel] = useState<"all" | "maybe" | "buy">("all");

  const [deals, setDeals] = useState<TopDealItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState<string | null>(null);

  const [selectedId, setSelectedId] = useState<string | null>(null);

  async function loadLeads(targetZip: string, limit = 200) {
    const cleanedZip = targetZip.trim();
    if (!cleanedZip) return;

    setLoading(true);
    setError(null);

    const params = new URLSearchParams();
    params.set("zip", cleanedZip);
    params.set("limit", String(limit));

    // Optional filters (backend can ignore if not implemented yet)
    if (minDscr.trim()) params.set("min_dscr", minDscr.trim());
    if (minCoc.trim()) params.set("min_coc", minCoc.trim());
    if (minLabel !== "all") params.set("min_label", minLabel);

    const url = `${apiBase()}/top-leads?${params.toString()}`;

    try {
      const r = await fetch(url);
      if (!r.ok) throw new Error(`API ${r.status}`);
      const data: LeadItem[] = await r.json();

      const mapped = data.map(leadToTopDeal);
      setDeals(mapped);
      setSelectedId(null);
      setLastUpdated(new Date().toLocaleString());
    } catch (e) {
      console.error(e);
      setError("Unable to load leads. Please try again.");
      setDeals([]);
    } finally {
      setLoading(false);
    }
  }

  async function generateLeads(
    targetZip: string,
    targetMaxPrice: number,
    limit = 300
  ) {
    const cleanedZip = targetZip.trim();
    if (!cleanedZip) return;

    setLoading(true);
    setError(null);

    const params = new URLSearchParams();
    params.set("zip", cleanedZip);
    params.set("max_price", String(targetMaxPrice));
    params.set("limit", String(limit));
    params.set("strategy", strategy);

    const url = `${apiBase()}/leads/from-properties?${params.toString()}`;

    try {
      const r = await fetch(url, { method: "POST" });
      if (!r.ok) throw new Error(`API ${r.status}`);

      // this returns {"zip": "...", "created": n, "updated": n}
      await r.json();

      // now refresh list
      await loadLeads(cleanedZip, 200);
    } catch (e) {
      console.error(e);
      setError("Lead generation failed. Check backend logs.");
    } finally {
      setLoading(false);
    }
  }

  // Initial load
  useEffect(() => {
    loadLeads(activeZip, 200);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Reload on strategy change (doesn't matter much today, but keeps behavior consistent)
  useEffect(() => {
    loadLeads(activeZip, 200);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [strategy]);

  const onSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const cleanedZip = zip.trim();
    if (!cleanedZip) return;

    const parsedMaxPrice = Number(maxPriceInput.replace(/,/g, "").trim());
    const finalMaxPrice =
      Number.isFinite(parsedMaxPrice) && parsedMaxPrice > 0
        ? parsedMaxPrice
        : 800000;

    setActiveZip(cleanedZip);
    setActiveMaxPrice(finalMaxPrice);

    // load existing leads; generation is a separate action
    loadLeads(cleanedZip, 200);
  };

  return (
    <div className="app-shell">
      <header className="app-header">
        <div className="app-header-left">
          <div className="app-logo">Haven</div>
          <div className="app-tagline">Real Estate Intelligence</div>
        </div>
        <div className="app-header-right">
          <span className="app-header-badge">Investor Preview</span>
        </div>
      </header>

      <section className="search-bar-section">
        <form className="search-bar" onSubmit={onSubmit}>
          <div className="search-field">
            <label htmlFor="zip-input" className="search-label">
              ZIP Code
            </label>
            <input
              id="zip-input"
              value={zip}
              onChange={(e) => setZip(e.target.value)}
              placeholder="e.g. 48009"
              className="search-input"
            />
          </div>

          <div className="search-field">
            <label htmlFor="max-price-input" className="search-label">
              Max Price
            </label>
            <input
              id="max-price-input"
              value={maxPriceInput}
              onChange={(e) => setMaxPriceInput(e.target.value)}
              placeholder="800,000"
              className="search-input"
            />
          </div>

          <div className="search-actions" style={{ display: "flex", gap: 10 }}>
            <button type="submit" className="primary-button">
              Refresh Leads
            </button>

            <button
              type="button"
              className="primary-button"
              onClick={() => generateLeads(activeZip, activeMaxPrice, 300)}
            >
              Load Leads
            </button>
          </div>
        </form>

        <div className="search-strategy-row">
          <span className="search-label">Strategy</span>
          <div className="strategy-toggle">
            <button
              type="button"
              className={
                strategy === "rental"
                  ? "strategy-btn strategy-btn-active"
                  : "strategy-btn"
              }
              onClick={() => setStrategy("rental")}
            >
              Rental (Buy & Hold)
            </button>
            <button
              type="button"
              className={
                strategy === "flip"
                  ? "strategy-btn strategy-btn-active"
                  : "strategy-btn"
              }
              onClick={() => setStrategy("flip")}
            >
              Flip (Value-Add)
            </button>
          </div>
        </div>

        <div className="search-bar secondary-filters">
          <div className="search-field">
            <label htmlFor="min-dscr-input" className="search-label">
              Min DSCR
            </label>
            <input
              id="min-dscr-input"
              type="number"
              step="0.05"
              value={minDscr}
              onChange={(e) => setMinDscr(e.target.value)}
              placeholder="e.g. 1.20"
              className="search-input"
            />
          </div>

          <div className="search-field">
            <label htmlFor="min-coc-input" className="search-label">
              Min CoC (decimal)
            </label>
            <input
              id="min-coc-input"
              type="number"
              step="0.01"
              value={minCoc}
              onChange={(e) => setMinCoc(e.target.value)}
              placeholder="e.g. 0.08"
              className="search-input"
            />
          </div>

          <div className="search-field">
            <label htmlFor="min-label-select" className="search-label">
              Label Filter
            </label>
            <select
              id="min-label-select"
              value={minLabel}
              onChange={(e) =>
                setMinLabel(e.target.value as "all" | "maybe" | "buy")
              }
              className="search-input"
            >
              <option value="all">All (BUY/MAYBE/PASS)</option>
              <option value="maybe">At least MAYBE</option>
              <option value="buy">BUY only</option>
            </select>
          </div>
        </div>

        <div className="search-meta">
          <div className="search-meta-primary">
            Viewing{" "}
            <span className="pill">
              {activeZip} · ≤ ${activeMaxPrice.toLocaleString()}
            </span>{" "}
            · <span className="pill">Leads (temporary Top Deals UI)</span>
          </div>
          {lastUpdated && (
            <div className="search-meta-secondary">
              Last updated: {lastUpdated}
            </div>
          )}
        </div>
      </section>

      <main className="app-main">
        <section className="app-main-left">
          <div className="panel">
            <TopDealsTable
              deals={deals}
              isLoading={loading}
              error={error}
              selectedDealId={selectedId}
              onSelectDeal={setSelectedId}
            />
          </div>
        </section>

        <section className="app-main-right">
          <div className="panel map-panel">
            <TopDealsMap
              deals={deals}
              selectedDealId={selectedId}
              onSelectDeal={setSelectedId}
            />
          </div>

          <AuctionLinksPanel />
        </section>
      </main>
    </div>
  );
}

export default App;
