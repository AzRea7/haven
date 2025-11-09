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

export function TopDealsTable({ zip, deals, loading, error }: Props) {
  if (!zip) {
    return <p>Enter a zip to view ranked deals.</p>;
  }

  if (loading) {
    return <p>Loading deals...</p>;
  }

  if (error) {
    return <p>{error}</p>;
  }

  if (deals.length === 0) {
    return <p>No deals found for {zip}.</p>;
  }

  return (
    <div style={{ overflowX: "auto" }}>
      <table
        style={{
          width: "100%",
          borderCollapse: "collapse",
          marginTop: "1rem",
        }}
      >
        <thead>
          <tr>
            <th>Address</th>
            <th>Price</th>
            <th>DSCR</th>
            <th>CoC %</th>
            <th>Rank</th>
            <th>Label</th>
          </tr>
        </thead>
        <tbody>
          {deals.map((d) => (
            <tr key={`${d.source}-${d.external_id}`}>
              <td title={d.reason}>
                {d.address}, {d.city}, {d.state} {d.zipcode}
              </td>
              <td>${d.list_price.toLocaleString()}</td>
              <td>{d.dscr.toFixed(2)}</td>
              <td>{(d.cash_on_cash_return * 100).toFixed(1)}%</td>
              <td>{d.rank_score.toFixed(1)}</td>
              <td>{d.label}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
