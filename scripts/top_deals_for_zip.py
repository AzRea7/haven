from haven.services.deals import get_top_deals_for_zip


def main() -> None:
    zip_code = "48009"

    # Fetch top deals already ranked & sorted
    deals = get_top_deals_for_zip(
        zip_code,
        limit_properties=200,
        limit_results=20
    )

    print(f"Haven — Top Deals for ZIP {zip_code}")
    print("Address\tPrice\tDSCR\tCoC %\tRank\tLabel")

    for d in deals:
        a = d["address"]
        fin = d["finance"]
        pricing = d.get("pricing", {})
        sc = d["score"]

        # Address string
        addr_str = f"{a['address']}, {a['city']}, {a['state']} {a['zipcode']}"

        # Price: new analyzer returns it under 'pricing'
        price = (
            pricing.get("ask_price") or
            fin.get("purchase_price") or
            d.get("list_price") or
            0
        )

        # Finance metrics
        dscr = float(fin.get("dscr", 0.0))
        coc_pct = float(fin.get("cash_on_cash_return", 0.0)) * 100.0
        rank_score = float(sc.get("rank_score", 0.0))
        label = str(sc.get("label", "unknown"))

        print(
            f"{addr_str}\t"
            f"${price:,.0f}\t"
            f"{dscr:.2f}\t"
            f"{coc_pct:.1f}%\t"
            f"{rank_score:.1f}\t"
            f"{label}"
        )


if __name__ == "__main__":
    main()
