import argparse

from haven.adapters.storage import read_df, write_df
from haven.services.features import normalize_sold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    args = ap.parse_args()

    df = read_df(args.inp)
    out = normalize_sold(df)
    write_df(out, args.out)

if __name__ == "__main__":
    main()
