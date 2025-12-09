// frontend/src/App.tsx
import React, { useEffect, useState } from "react";
import { TopDealsTable, type TopDealItem } from "./components/TopDealsTable";
import { TopDealsMap } from "./components/TopDealsMap";
import { AuctionLinksPanel } from "./components/AuctionLinksPanel";

type Strategy = "rental" | "flip";

function App() {
  const [zip, setZip] = useState("48009");
  const [activeZip, setActiveZip] = useState("48009");

  const [maxPriceInput, setMaxPriceInput] = useState<string>("800000");
  const [activeMaxPrice, setActiveMaxPrice] = useState<number>(800000);

  const [strategy, setStrategy] = useState<Strategy>("rental");

  // New investor-style filters
  const [minDscr, setMinDscr] = useState<string>("");
  const [minCoc, setMinCoc] = useState<string>("");
  const [minLabel, setMinLabel] = useState<"all" | "maybe" | "buy">("all");

  const [deals, setDeals] = useState<TopDealItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState<string | null>(null);

  const loadDeals = (
    targetZip: string,
    targetMaxPrice: number,
    targetStrategy: Strategy = strategy
  ) => {
    const cleanedZip = targetZip.trim();
    if (!cleanedZip) return;

    const maxPrice = Number(targetMaxPrice) || 0;
    if (maxPrice <= 0) return;

    setLoading(true);
    setError(null);

    const params = new URLSearchParams();
    params.set("zip", cleanedZip);
    params.set("max_price", String(maxPrice));
    params.set("strategy", targetStrategy);

    // Wire up investor-style filters
    if (minDscr.trim()) {
      params.set("min_dscr", minDscr.trim());
    }
    if (minCoc.trim()) {
      params.set("min_coc", minCoc.trim());
    }
    if (minLabel !== "all") {
      params.set("min_label", minLabel);
    }

    const url = `/top-deals?${params.toString()}`;

    fetch(url)
      .then((r) => {
        if (!r.ok) {
          throw new Error(`API ${r.status}`);
        }
        return r.json();
      })
      .then((data: TopDealItem[]) => {
        setDeals(data);
        setLastUpdated(new Date().toLocaleString());
      })
      .catch((err: unknown) => {
        console.error(err);
        setError("Unable to load deals. Please try again.");
        setDeals([]);
      })
      .finally(() => {
        setLoading(false);
      });
  };

  // Initial load
  useEffect(() => {
    loadDeals(activeZip, activeMaxPrice, strategy);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Reload when strategy changes
  useEffect(() => {
    loadDeals(activeZip, activeMaxPrice, strategy);
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
    loadDeals(cleanedZip, finalMaxPrice, strategy);
  };

  return (
    <div className="app-shell">
      {/* Top Navigation / Branding */}
      <header className="app-header">
        <div className="app-header-left">
          <div className="app-logo">Haven</div>
          <div className="app-tagline">Real Estate Intelligence</div>
        </div>
        <div className="app-header-right">
          <span className="app-header-badge">Investor Preview</span>
        </div>
      </header>

      {/* Search Controls */}
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

          <div className="search-actions">
            <button type="submit" className="primary-button">
              Run Analysis
            </button>
          </div>
        </form>

        {/* Strategy Toggle */}
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

        {/* Investor-style filters */}
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
            ·{" "}
            <span className="pill">
              {strategy === "rental"
                ? "Rental / Cashflow"
                : "Flip / ARV Spread"}
            </span>
          </div>
          {lastUpdated && (
            <div className="search-meta-secondary">
              Last updated: {lastUpdated}
            </div>
          )}
        </div>
      </section>

      {/* Main Content: List + Map + Auctions */}
      <main className="app-main">
        <section className="app-main-left">
          <div className="panel">
            <TopDealsTable deals={deals} isLoading={loading} error={error} />
          </div>
        </section>

        <section className="app-main-right">
          <div className="panel map-panel">
            <TopDealsMap deals={deals} />
          </div>

          <AuctionLinksPanel />
        </section>
      </main>
    </div>
  );
}

export default App;
