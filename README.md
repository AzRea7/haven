# Haven — Logistic Financial Decisioning for Real Estate

**What:** Calibrated logistic models that predict financially-defined success:
- **Flip Success** = P(net profit ≥ target within 6–12 months)
- **Rent Viability** = P(DSCR ≥ 1.25 and CoC ≥ target)
- **Live-in Safety** = P(5-yr downside within tolerance and payment affordable)

**No sentiment or NLP.** All features are numeric, finance-only.

## Why logistic?
Logistic models produce calibrated probabilities for clearly-defined financial events, enabling threshold tuning for **precision** when capital is scarce.

## Finance metrics (deterministic)
NOI, Cap Rate, DSCR, CoC, IRR/NPV, Flip Net Profit. These feed labels and features; models learn the probability of meeting thresholds.

## Data
- Listings, sales, rents (Zillow/Redfin/MLS/ZORI)
- Taxes, insurance proxies, hazard layers
- Market rings (0.25–1.0 mi): price/sqft, DOM, MoS, momentum
- Financing scenarios (rate, LTV, points)

> Use only data you’re licensed to use.

## Modeling
- Baselines: Logistic Regression (L2) + Isotonic calibration  
- Optional: LightGBM (binary objective) + calibration  
- Time-aware CV (forward chaining) by quarter/city

## Metrics
Precision/Recall/F1/ROC-AUC, PR-AUC; **Precision@K**, Profit@Top-K, NDCG@K; calibration curves.

## Quickstart
```bash
conda create -n refinlogit python=3.11 -y && conda activate refinlogit
pip install -r requirements.txt
python scripts/build_features.py            # create data/curated/train.parquet
python scripts/train_flip.py
python scripts/score_candidates.py
