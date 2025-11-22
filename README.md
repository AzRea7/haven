# Haven – Real Estate Investment Analysis Platform

Haven is an end-to-end real estate analytics platform that ingests Zillow-style listings, predicts rents with machine learning, computes investment metrics (DSCR, Cash-on-Cash, etc.), and surfaces the best deals via a FastAPI backend and a React/Vite frontend.

The goal is simple: **given a ZIP code, show the top investment properties in seconds**.

---

## Table of Contents

1. Features  
2. Architecture  
3. Tech Stack  
4. Getting Started  
   - Prerequisites  
   - Backend Setup  
   - Frontend Setup  
5. Environment Configuration  
6. Data Ingestion & Model Training  
7. Running the Application  
8. Key API Endpoints  
9. Project Structure  
10. Testing & Quality  
11. Troubleshooting  

---

## Features

- **Automated Property Ingestion**  
  Ingests listings in parallel with rate-limit awareness.

- **Machine Learning–Based Rent Estimation**  
  LightGBM rent model with quantile predictions.

- **Financial Deal Analysis**  
  DSCR, CoC, Cap Rate, and risk-based labels.

- **Top Deals UI (Zillow-Style)**  
  React + Vite interface with map and sortable tables.

- **FastAPI Backend**  
  Clean REST endpoints with automatic documentation.

---

## Architecture

High-level design:

```
            Frontend (React + Vite)
              │
              ▼
            Backend (FastAPI)
              │
              ▼
       ML Models (LightGBM)
              │
              ▼
        SQLite Database
```

---

## Tech Stack

**Backend:** FastAPI, Python 3.10+, Uvicorn, SQLAlchemy, LightGBM  
**Frontend:** React, Vite, TailwindCSS, Leaflet  
**Tools:** Ruff, Mypy, Pytest, Docker (optional)

---

## Getting Started

### Prerequisites

- Python 3.10+
- Node.js 18+
- Git

Clone repo:

```
git clone https://github.com/AzRea7/haven.git
cd haven
```

---

## Backend Setup

Create virtual environment:

```
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```
pip install -r requirements.txt
```

Copy example env:

```
cp .env.example .env
```

Initialize database:

```
python scripts/init_db.py
```

---

## Frontend Setup

Enter frontend folder:

```
cd frontend
```

Install dependencies:

```
npm install
```

Create `.env`:

```
VITE_API_BASE_URL=http://localhost:8000
```

---

## Environment Configuration

Backend `.env`:

```
DATABASE_URL=sqlite:///haven.db
ZILLOW_API_KEY=your_key
RENT_MODEL_PATH=models/rent_lightgbm.pkl
FLIP_MODEL_PATH=models/flip_model.pkl
LOG_LEVEL=info
```

Frontend `.env`:

```
VITE_API_BASE_URL=http://localhost:8000
```

---

## Data Ingestion & Model Training

### Ingest properties:

```
python scripts/ingest_properties_parallel.py --zip 48009 48363 --workers=2
```

### Build features:

```
python scripts/build_features_parallel.py
```

### Train rent model:

```
python scripts/train_rent_quantiles.py
```

### Train flip model:

```
python scripts/train_flip.py
```

---

## Running the Application

### Start backend:

```
uvicorn haven.api.main:app --reload --port 8000
```

### Start frontend:

```
npm run dev
```

Visit UI:

```
http://localhost:5173
```

Visit backend docs:

```
http://localhost:8000/docs
```

---

## Key API Endpoints

- `GET /health`
- `GET /api/top-deals?zip=48009`
- `POST /api/analyze-deal`

Interactive docs at `/docs`.

---

## Project Structure

```
haven/
├── src/haven/
│   ├── api/
│   ├── adapters/
│   ├── services/
│   └── ...
├── scripts/
├── frontend/
├── models/
├── data/
└── haven.db
```

---

## Testing & Quality

```
ruff src
mypy src
pytest
```

---

## Troubleshooting

**No properties appear?**  
Run ingestion again.

**Rate limited?**  
Reduce worker count.

**Frontend shows blank map?**  
Adjust container height.

---

Haven is designed to scale into GPU scoring, multi-region ingestion, and RL-based ranking in future versions.
