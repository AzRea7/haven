// frontend/src/App.tsx

import { useEffect, useState } from "react";
import { TopDealsTable, type TopDealItem } from "./components/TopDealsTable";
import { TopDealsMap } from "./components/TopDealsMap";

function App() {
  const [zip, setZip] = useState("48009");
  const [activeZip, setActiveZip] = useState("48009");
  const [deals, setDeals] = useState<TopDealItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const maxPrice = 800000;

  const loadDeals = (targetZip: string) => {
    if (!targetZip) return;

    setLoading(true);
    setError(null);

    const url = `/top-deals?zip=${encodeURIComponent(
      targetZip
    )}&max_price=${maxPrice}`;

    fetch(url)
      .then((r) => {
        if (!r.ok) {
          throw new Error(`API ${r.status}`);
        }
        return r.json();
      })
      .then((data: TopDealItem[]) => {
        setDeals(data);
      })
      .catch((err: any) => {
        console.error(err);
        setError("Failed to load deals");
        setDeals([]);
      })
      .finally(() => {
        setLoading(false);
      });
  };

  // Initial load
  useEffect(() => {
    loadDeals(activeZip);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const onSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const cleaned = zip.trim();
    if (!cleaned) return;
    setActiveZip(cleaned);
    loadDeals(cleaned);
  };

  return (
    <div style={{ padding: "1.5rem", fontFamily: "system-ui, sans-serif" }}>
      <h1>Haven — Top Deals</h1>
      <form
        onSubmit={onSubmit}
        style={{ marginTop: "1rem", marginBottom: "1rem" }}
      >
        <label>
          Zip:
          <input
            value={zip}
            onChange={(e) => setZip(e.target.value)}
            style={{ marginLeft: "0.5rem" }}
          />
        </label>
        <button type="submit" style={{ marginLeft: "0.75rem" }}>
          Load
        </button>
      </form>

      <TopDealsTable
        zip={activeZip}
        deals={deals}
        loading={loading}
        error={error}
      />

      <TopDealsMap deals={deals} />
    </div>
  );
}

export default App;
