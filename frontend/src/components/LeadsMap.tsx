import React, { useMemo } from "react";
import { MapContainer, TileLayer, Marker, Tooltip } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import L from "leaflet";
import type { LeadItem, LeadStage } from "./LeadsTable";

function iconForStage(stage: LeadStage) {
  // Simple approach: use the same marker asset, but change size by stage.
  const baseSize =
    stage === "new"
      ? 24
      : stage === "contacted"
      ? 28
      : stage === "appointment"
      ? 30
      : stage === "contract"
      ? 32
      : stage === "closed_won"
      ? 34
      : 22;

  return L.icon({
    iconUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png",
    iconRetinaUrl:
      "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png",
    shadowUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png",
    iconSize: [baseSize, Math.round(baseSize * 1.6)],
    iconAnchor: [Math.round(baseSize / 2), Math.round(baseSize * 1.6)],
    popupAnchor: [1, -34],
    shadowSize: [41, 41],
  });
}

type Props = {
  leads: LeadItem[];
};

export function LeadsMap({ leads }: Props) {
  const geoLeads = useMemo(
    () =>
      leads.filter(
        (d) =>
          typeof d.lat === "number" &&
          typeof d.lon === "number" &&
          d.lat !== null &&
          d.lon !== null
      ),
    [leads]
  );

  const center = useMemo(() => {
    if (geoLeads.length === 0) return { lat: 42.67, lon: -83.14 };
    const avgLat =
      geoLeads.reduce((acc, d) => acc + (d.lat || 0), 0) / geoLeads.length;
    const avgLon =
      geoLeads.reduce((acc, d) => acc + (d.lon || 0), 0) / geoLeads.length;
    return { lat: avgLat, lon: avgLon };
  }, [geoLeads]);

  return (
    <>
      <div className="panel-header">
        <h2>Leads Map</h2>
        <div className="muted">
          Markers scale by stage. Tooltip shows lead score.
        </div>
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

          {geoLeads.map((lead) => (
            <Marker
              key={lead.lead_id}
              position={[lead.lat as number, lead.lon as number]}
              icon={iconForStage(lead.stage)}
            >
              <Tooltip direction="top" offset={[0, -10]}>
                <div style={{ fontSize: 11 }}>
                  <div style={{ fontWeight: 600 }}>{lead.address}</div>
                  <div>
                    {lead.city}, {lead.state} {lead.zipcode}
                  </div>
                  <div style={{ marginTop: 4 }}>
                    <strong>Score:</strong> {lead.lead_score.toFixed(1)} / 100
                  </div>
                  <div>
                    <strong>Stage:</strong> {lead.stage.toUpperCase()}
                  </div>
                  <div>
                    <strong>Touches:</strong> {lead.touches ?? 0}
                  </div>
                </div>
              </Tooltip>
            </Marker>
          ))}
        </MapContainer>
      </div>
    </>
  );
}
