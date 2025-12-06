import React, { useMemo } from "react";
import { MapContainer, TileLayer, Marker, Tooltip } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import L from "leaflet";
import type { TopDealItem } from "./TopDealsTable";

// Default markers
const DefaultIcon = L.icon({
  iconUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png",
  iconRetinaUrl:
    "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png",
  shadowUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png",
  iconSize: [25, 41],
  iconAnchor: [12, 41],
  popupAnchor: [1, -34],
  shadowSize: [41, 41],
});

const SelectedIcon = L.icon({
  iconUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png",
  iconRetinaUrl:
    "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png",
  shadowUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png",
  iconSize: [30, 48],
  iconAnchor: [15, 48],
  popupAnchor: [1, -40],
  shadowSize: [48, 48],
});

(L.Marker.prototype as any).options.icon = DefaultIcon;

type Props = {
  deals: TopDealItem[];
  selectedDealId?: string | null;
  onSelectDeal?: (id: string | null) => void;
};

function formatMoney(value: number): string {
  if (!Number.isFinite(value)) return "-";
  return value.toLocaleString("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0,
  });
}

/**
 * Same key logic as TopDealsTable so selection stays in sync.
 */
function dealKey(deal: TopDealItem): string {
  return (
    deal.id ||
    deal.external_id ||
    `${deal.address}-${deal.zipcode}-${deal.list_price}`
  );
}

export function TopDealsMap({ deals, selectedDealId, onSelectDeal }: Props) {
  const geoDeals = useMemo(
    () =>
      deals.filter(
        (d) =>
          typeof d.lat === "number" &&
          typeof d.lon === "number" &&
          d.lat !== null &&
          d.lon !== null
      ),
    [deals]
  );

  const center = useMemo(() => {
    if (geoDeals.length === 0) {
      return { lat: 42.67, lon: -83.14 }; // default SE Michigan-ish
    }
    const avgLat =
      geoDeals.reduce((acc, d) => acc + (d.lat || 0), 0) / geoDeals.length;
    const avgLon =
      geoDeals.reduce((acc, d) => acc + (d.lon || 0), 0) / geoDeals.length;
    return { lat: avgLat, lon: avgLon };
  }, [geoDeals]);

  return (
    <>
      <div className="panel-header">
        <h2>Map</h2>
        <div className="muted">Click markers to view score details.</div>
      </div>

      <div className="map-container">
        <MapContainer
          center={[center.lat, center.lon]}
          zoom={11}
          style={{ height: "100%", width: "100%" }}
          scrollWheelZoom={false}
        >
          <TileLayer
            attribution='&copy; <a href="https://www.openstreetmap.org/">OpenStreetMap</a> contributors'
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          />

          {geoDeals.map((deal) => {
            const key = dealKey(deal);
            const isSelected = selectedDealId === key;

            return (
              <Marker
                key={key}
                position={[deal.lat as number, deal.lon as number]}
                icon={isSelected ? SelectedIcon : DefaultIcon}
                eventHandlers={{
                  click: () => onSelectDeal?.(isSelected ? null : key),
                }}
              >
                <Tooltip direction="top" offset={[0, -10]}>
                  <div style={{ fontSize: "11px" }}>
                    <div style={{ fontWeight: 600 }}>{deal.address}</div>
                    <div>
                      {deal.city}, {deal.state} {deal.zipcode}
                    </div>
                    <div style={{ marginTop: 4 }}>
                      {formatMoney(deal.list_price)}
                    </div>
                    <div style={{ marginTop: 4 }}>
                      DSCR {deal.dscr.toFixed(2)} · CoC{" "}
                      {(deal.cash_on_cash_return * 100).toFixed(1)}% · Score{" "}
                      {deal.rank_score.toFixed(1)} · {deal.label.toUpperCase()}
                    </div>
                  </div>
                </Tooltip>
              </Marker>
            );
          })}
        </MapContainer>
      </div>
    </>
  );
}
