import argparse
from haven.adapters.zillow_hasdata import fetch_zillow_dump
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--type", choices=["sold","forSale"], required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    fetch_zillow_dump(args.type, args.out)

if __name__ == "__main__":
    main()
