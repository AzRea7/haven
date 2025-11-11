# Haven Real Estate Intelligence Platform

**Haven** is an AI-powered real estate analytics platform that ranks and visualizes investment opportunities by ZIP code.  
It combines **FastAPI** (Python) for financial modeling and data analysis with a **React + Vite** frontend for live visualization.  

Haven helps investors identify, score, and monitor the most profitable residential deals in their target markets.

---

## 🌍 Overview

| Component | Tech Stack | Description |
|------------|-------------|--------------|
| **Backend API** | Python • FastAPI • SQLite | Runs the core property scoring logic, financial modeling, and quantile-based ML predictions. |
| **Frontend App** | React • Vite • TypeScript | Displays ranked investment opportunities in a table and on an interactive map. |
| **ML Models (Optional)** | LightGBM | Provide quantile estimates for rent and after-repair value (ARV) uncertainty. |

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/<yourname>/haven.git
cd haven
2. Backend Setup (FastAPI)
Create and activate a virtual environment

bash
Copy code
python -m venv venv
source venv/bin/activate     # macOS/Linux
venv\Scripts\activate        # Windows
Install dependencies

bash
Copy code
pip install -r requirements.txt
Run the API server

bash
Copy code
uvicorn haven.api.http:app --reload --port 8000
Then open the docs:
👉 http://127.0.0.1:8000/docs

Example request:

bash
Copy code
GET /top-deals?zip=48009&max_price=800000
Returns a ranked list of properties with investment metrics and labels (buy, maybe, pass).

3. Frontend Setup (React + Vite)
Open a new terminal:

bash
Copy code
cd frontend
npm install
npm run dev
Visit the web app at http://localhost:5173.

You’ll see:

A table of ranked properties

An interactive map with deal markers shaded by rank score

⚙️ Configuration
All environment configuration is handled via src/haven/adapters/config.py (Pydantic-based).

Default Screening Constants (in api/http.py)
python
Copy code
DEFAULT_DOWN_PAYMENT_PCT = 0.25
DEFAULT_INTEREST_RATE = 0.065
DEFAULT_LOAN_TERM_YEARS = 30
DEFAULT_TAXES_ANNUAL = 3000.0
DEFAULT_INSURANCE_ANNUAL = 1200.0
Optional ML Model Bundles
If you have trained quantile models for rent or ARV, save them under models/:

Copy code
models/
 ├── rent_quantiles.pkl
 └── arv_quantiles.pkl
Set environment variables (optional):

ini
Copy code
RENT_QUANTILE_PATH=models/rent_quantiles.pkl
ARV_QUANTILE_PATH=models/arv_quantiles.pkl
If these are not present, Haven automatically falls back to ±10% heuristic bands for uncertainty estimation.

🧠 How It Works
Core Pipeline (Triggered by /top-deals)

Load property listings from the local SQL repository (SqlPropertyRepository).

Construct Property objects with standardized financing assumptions.

Compute financial metrics (DSCR, Cash-on-Cash Return, Breakeven Occupancy).

Estimate rent and ARV quantiles using LightGBM models or fallback heuristics.

Compute a risk-adjusted rank score and assign a label (buy, maybe, pass).

Save each analysis in the database for calibration and trend tracking.

Return a sorted JSON list to the frontend.

📊 Frontend Architecture
Located in /frontend/:

File	Purpose
src/components/TopDealsTable.tsx	Renders a sortable list of deals and financial metrics.
src/components/TopDealsMap.tsx	Displays interactive map markers via React Leaflet.
src/App.tsx	Manages state, fetches /top-deals, and connects the table + map.
vite.config.ts	Proxy setup for seamless backend API calls during development.

Proxy Configuration

ts
Copy code
server: {
  port: 5173,
  host: "127.0.0.1",
  proxy: {
    "/top-deals": {
      target: "http://127.0.0.1:8000",
      changeOrigin: true,
    },
  },
}
🧩 Example Output
Address	Price	DSCR	CoC %	Rank	Label
1225 Derby Rd APT 1	$210,000	-0.35	-27.5%	-48.0	pass
444 Chester St APT 425	$289,900	-0.25	-25.5%	-48.0	pass

Map markers are positioned using lat and lon from the dataset, colored by rank_score intensity.

🧰 Developer Commands
Action	Command
Run Backend	uvicorn haven.api.http:app --reload --port 8000
Run Frontend	npm run dev
Run Tests	pytest -v
Type Check	mypy src/
Format Code	black src/