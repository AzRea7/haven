// frontend/src/App.tsx

import { useEffect, useState } from "react";
import { TopDealsTable, type TopDealItem } from "./components/TopDealsTable";
import { TopDealsMap } from "./components/TopDealsMap";

function App() {
  const [zip, setZip] = useState("48009");
  const [activeZip, setActiveZip] = useState("48009");

  const [maxPriceInput, setMaxPriceInput] = useState<string>("800000");
  const [activeMaxPrice, setActiveMaxPrice] = useState<number>(800000);

  const [deals, setDeals] = useState<TopDealItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState<string | null>(null);

  const loadDeals = (targetZip: string, targetMaxPrice: number) => {
    if (!targetZip || !targetMaxPrice) return;

    setLoading(true);
    setError(null);

    const url = `/top-deals?zip=${encodeURIComponent(
      targetZip
    )}&max_price=${targetMaxPrice}`;

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
    loadDeals(activeZip, activeMaxPrice);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

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
    loadDeals(cleanedZip, finalMaxPrice);
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

        <div className="search-meta">
          <div className="search-meta-primary">
            Viewing results for{" "}
            <span className="pill">
              {activeZip} · ≤ ${activeMaxPrice.toLocaleString()}
            </span>
          </div>
          {lastUpdated && (
            <div className="search-meta-secondary">
              Last updated: {lastUpdated}
            </div>
          )}
        </div>
      </section>

      {/* Main Content: List + Map */}
      <main className="app-main">
        <section className="app-main-left">
          <TopDealsTable
            zip={activeZip}
            deals={deals}
            loading={loading}
            error={error}
          />
        </section>

        <section className="app-main-right">
          <TopDealsMap deals={deals} />
        </section>
      </main>
    </div>
  );
}

export default App;
