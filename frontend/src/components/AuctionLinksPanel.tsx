// frontend/src/components/AuctionLinksPanel.tsx
import React from "react";

export type AuctionLinkCategory =
  | "foreclosure_marketplace"
  | "tax_sale"
  | "online_marketplace"
  | "regional";

export type AuctionLink = {
  id: string;
  name: string;
  url: string;
  category: AuctionLinkCategory;
  description: string;
  notes?: string;
};

/**
 * Curated Michigan auction / foreclosure sources.
 * These are intentionally high-signal platforms for MI:
 * - Auction.com: statewide foreclosure & REO marketplace
 * - Michigan Treasury: official state-run real property auctions
 * - Tax-Sale.info: county tax-foreclosure auctions across Michigan
 * - Hubzu: online foreclosure / REO auctions with MI filter
 * - LASTBIDrealestate: Michigan-based online auction company
 */
const MI_AUCTIONS: AuctionLink[] = [
  {
    id: "auction-dot-com-mi",
    name: "Auction.com – Michigan Foreclosures",
    url: "https://www.auction.com/residential/mi",
    category: "foreclosure_marketplace",
    description:
      "Large statewide marketplace for foreclosure and bank-owned properties across Michigan.",
    notes: "Great for bank-owned and trustee sales; filter by city/ZIP.",
  },
  {
    id: "mi-treasury-real-property",
    name: "Michigan Treasury – Real Property Auctions",
    url: "https://www.michigan.gov/treasury/auctions/real-property",
    category: "tax_sale",
    description:
      "Official Michigan Department of Treasury page linking to state-managed real property auctions.",
    notes:
      "Good entry point into county-level tax auctions and official notices.",
  },
  {
    id: "tax-sale-info",
    name: "Tax-Sale.info – Michigan County Tax Foreclosures",
    url: "https://www.tax-sale.info/",
    category: "tax_sale",
    description:
      "Primary auction platform for many Michigan county tax-foreclosed properties.",
    notes: "Key site for tax foreclosure deals; check county schedules.",
  },
  {
    id: "hubzu-mi",
    name: "Hubzu – Michigan Auctions",
    url: "https://www.hubzu.com/property/list/state/MI",
    category: "online_marketplace",
    description:
      "Online auction platform featuring foreclosures, bank-owned, and investment properties.",
    notes: "Useful for additional deal flow beyond Auction.com.",
  },
  {
    id: "lastbidrealestate",
    name: "LASTBIDrealestate – Michigan Real Estate Auctions",
    url: "https://www.lastbidrealestate.com/",
    category: "regional",
    description:
      "Michigan-based online auction company handling residential, commercial, and land auctions.",
    notes: "Good regional source; check upcoming auction calendar.",
  },
];

function categoryLabel(category: AuctionLinkCategory): string {
  switch (category) {
    case "foreclosure_marketplace":
      return "Foreclosure marketplace";
    case "tax_sale":
      return "Tax foreclosure / county sale";
    case "online_marketplace":
      return "Online auction marketplace";
    case "regional":
      return "Regional MI auction house";
    default:
      return "Auction source";
  }
}

function categoryChipClass(category: AuctionLinkCategory): string {
  switch (category) {
    case "foreclosure_marketplace":
      return "auction-chip auction-chip-foreclosure";
    case "tax_sale":
      return "auction-chip auction-chip-taxsale";
    case "online_marketplace":
      return "auction-chip auction-chip-online";
    case "regional":
    default:
      return "auction-chip auction-chip-regional";
  }
}

export function AuctionLinksPanel() {
  return (
    <div className="panel auctions-panel">
      <div className="panel-header">
        <h2>Auctions & Foreclosures</h2>
        <div className="muted">
          Curated Michigan auction platforms for foreclosure, tax sales, and
          investor-friendly deals.
        </div>
      </div>

      <div className="auction-list">
        {MI_AUCTIONS.map((link) => (
          <a
            key={link.id}
            href={link.url}
            target="_blank"
            rel="noreferrer"
            className="auction-card"
          >
            <div className="auction-card-header">
              <div>
                <div className="auction-name">{link.name}</div>
                <div className="auction-description">{link.description}</div>
              </div>
              <span className={categoryChipClass(link.category)}>
                {categoryLabel(link.category)}
              </span>
            </div>

            {link.notes && (
              <div className="auction-notes">
                <span className="auction-notes-label">Why it matters: </span>
                {link.notes}
              </div>
            )}

            <div className="auction-url-hint">Open site →</div>
          </a>
        ))}
      </div>
    </div>
  );
}
