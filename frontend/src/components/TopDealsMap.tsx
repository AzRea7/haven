import { MapContainer, TileLayer, Marker, Tooltip } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import L from "leaflet";
import type { TopDealItem } from "./TopDealsTable";

// Fix default Leaflet icon paths in bundlers
const defaultIcon = L.icon({
  iconUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png",
  iconRetinaUrl:
    "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png",
  shadowUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png",
  iconSize: [25, 41],
  iconAnchor: [12, 41],
});
L.Marker.prototype.options.icon = defaultIcon;

type Props = {
  deals: TopDealItem[];
};

export function TopDealsMap({ deals }: Props) {
  const withCoords = deals.filter(
    (d) =>
      d.lat !== null &&
      d.lat !== undefined &&
      d.lon !== null &&
      d.lon !== undefined
  );

  if (withCoords.length === 0) {
    return null;
  }

  const center: [number, number] = [
    withCoords[0].lat as number,
    withCoords[0].lon as number,
  ];

  return (
    <div style={{ marginTop: "1rem" }}>
      <MapContainer
        center={center}
        zoom={12}
        style={{ height: "400px", width: "100%" }}
      >
        <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
        {withCoords.map((d) => (
          <Marker
            key={`${d.source}-${d.external_id}`}
            position={[d.lat as number, d.lon as number]}
          >
            <Tooltip>
              {d.address}
              <br />
              Score: {d.rank_score.toFixed(1)} ({d.label})
              <br />
              CoC: {(d.cash_on_cash_return * 100).toFixed(1)}%
            </Tooltip>
          </Marker>
        ))}
      </MapContainer>
    </div>
  );
}
