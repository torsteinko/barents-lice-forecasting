# Barents Lice Forecasting

Forecasting, exploratory analysis, and map-based decision support for weekly salmon-lice pressure at Norwegian aquaculture sites.

This repository was built around the BarentsWatch lice data challenge. It combines a leakage-safe modeling pipeline, a latest-week operational snapshot, an interactive FastAPI viewer, and a Gemini-backed natural-language chat layer over the prepared data.

The project is best understood as a reproducible research and demo repository rather than a production deployment.

## What is in this repository

- Exploratory analysis of lice counts, treatments, geography, temperature, and breach patterns.
- Predictive models for site-level lice-limit breaches at 1, 2, and 12 week horizons.
- A latest-snapshot pipeline that applies saved models to the newest raw data without retraining on 2026 rows.
- An interactive web map for browsing current site status, forecast risk, and model outputs.
- A Gemini-powered SQL chat endpoint for querying the prepared datasets in natural language.
- A separate Python port of the Stata-based salmon growth model from the case bonus task.

## Project highlights

- Training and holdout evaluation are capped before 2026 to avoid leakage into the predictive task.
- Latest operational scoring uses the newest raw reporting week for current status, while forecasts are anchored to the latest reliable scoring week when end-of-series reporting is incomplete.
- The viewer and chat both consume the same latest snapshot artifacts when they exist.
- Chat is read-only and uses `site_snapshot` for current and latest questions.
- Chat uses `master_table` for historical week-by-week analysis.
- The map viewer includes search, production-area and county filters, risk filtering, detail cards, and site jump actions from chat results.

## Repository layout

```text
data/                   Raw inputs and processed parquet tables
notebooks/              Walkthrough notebook
results/                Metrics, predictions, documentation, and saved models
src/lice/               Core pipeline, features, models, API, and chat service
stata/                  Python port of the growth-model bonus task
viewer/                 Frontend assets for the map application
run_baseline.py         Baseline training and evaluation entrypoint
run_latest_snapshot.py  Latest-data scoring entrypoint
run_web.py              FastAPI viewer and chat entrypoint
```

## Data

The project expects the BarentsWatch source files at these paths:

- `data/vlice.csv`
- `data/vtreatment.csv`

### **Due to the size they are not uploaded to the repository, so you will have to upload them yourself.**

The challenge data is based on public BarentsWatch lice and treatment datasets. If you intend to publish this repository publicly with raw CSVs or generated derivative artifacts committed, verify the relevant redistribution terms before doing so.

## Modeling policy

- No 2026 data is used for model training or leakage-prone feature construction.
- The baseline pipeline creates both binary breach targets and breach-count targets for 1, 2, and 12 week horizons.
- Feature engineering includes site history, seasonal context, treatment history, geography, production-area pressure, and 50 km neighbor-pressure features.
- Saved model outputs are post-processed so forecast probabilities are non-decreasing across the nested 1 week, 2 week, and 12 week windows.
- The latest operational snapshot separates the latest raw reporting week for current site state from the latest reliable forecast anchor week for model scoring.

That separation matters because the newest raw weeks can be operationally incomplete even when they are useful for showing the most recent reported status.

## Setup

### 1. Create a virtual environment

```powershell
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

macOS or Linux:

```bash
source .venv/bin/activate
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Configure Gemini chat access

The interactive map works without Gemini, but the `/chat` endpoint requires Vertex AI credentials.

Create a `.env` file in the repository root with values like these:

```dotenv
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_LOCATION=global
VERTEX_GEMINI_MODEL=gemini-3.1-flash-lite
```

The repository also includes `.env.example` with equivalent PowerShell assignments.

Authentication should come from Google Application Default Credentials, for example:

- `gcloud auth application-default login`
- or a service-account-based environment in your deployment target

## Quick start

### Build the baseline training artifacts

```powershell
python run_baseline.py
```

This step builds the cleaned master table, creates targets and features, trains the horizon-specific models, and writes evaluation artifacts under `data/processed/` and `results/`.

### Refresh the latest operational snapshot

```powershell
python run_latest_snapshot.py
```

This step reuses the saved models in `results/models/` to score the newest available data and refresh the latest viewer and chat artifacts.

### Start the API and interactive viewer

```powershell
python run_web.py
```

Then open:

- `http://127.0.0.1:8000/`

The main API endpoints are:

- `GET /api/health`
- `GET /api/sites`
- `POST /chat`

## Frontend options

The default workflow is to serve the viewer through FastAPI at `http://127.0.0.1:8000/`.

If you want to serve the static frontend separately, keep FastAPI running for the API endpoints and start a static file server as well:

```powershell
python run_web.py
python -m http.server 8765
```

Then open:

- `http://127.0.0.1:8765/viewer/index.html`

If the backend is running on a different origin, append `?api=http://host:port` to the viewer URL.

## Chat behavior

The chat experience is designed to answer two different kinds of questions from the right dataset surface:

- Current or latest questions use `site_snapshot`, including current limit ratio, current over-limit sites, latest raw-week status, and current forecast cards.
- Historical questions use `master_table`, including repeated breaches, treatment intensity over time, weekly trends, and area-level historical comparisons.

The backend builds a fresh in-memory DuckDB database per request, restricts the agent to read-only SQL, and blocks admin, write, and file-reading SQL operations server-side.

## Key outputs

### Processed data

- `data/processed/master_table.parquet`: baseline cleaned site-week table with features and targets.
- `data/processed/master_table_latest.parquet`: refreshed full-history table used by the latest snapshot viewer and chat.
- `results/data_audit.json`: raw-data audit summary.

### Model and evaluation artifacts

- `results/model_metrics.json`: consolidated holdout metrics.
- `results/model_comparison.csv`: comparison view across model variants.
- `results/holdout_predictions.csv`: holdout predictions for inspection.
- `results/feature_columns.csv`: feature list used in modeling.
- `results/final_model_documentation.md`: model documentation and recommendations.
- `results/model_tuning_journal.md`: tuning log and decision trail.
- `results/xgb_short_horizon_benchmark.csv`: focused short-horizon benchmark artifact.
- `results/models/*.pkl`: serialized classifier and regressor artifacts.

### Latest snapshot artifacts

- `results/latest_predictions.csv`: long-form latest model predictions.
- `results/latest_site_snapshot.csv`: one-row-per-site merged snapshot.
- `results/site_map.geojson`: baseline case-viewer snapshot aligned to the validated training cutoff.
- `results/site_map_latest.geojson`: latest viewer dataset used by the map and chat when present.

### Analysis and presentation artifacts

- `results/exploratory_analysis_summary.md`: summary findings from the exploratory analysis task.
- `notebooks/01_baseline_walkthrough.ipynb`: walkthrough notebook for the pipeline and artifacts.
- `stata/output/`: outputs from the Stata-to-Python growth-model task.

## Additional task deliverables

### Notebook walkthrough

Open `notebooks/01_baseline_walkthrough.ipynb` in VS Code or Jupyter and select the project virtual environment as the kernel.

### Stata bonus-task port

Run the salmon growth module with:

```powershell
python -m stata
```

This writes the default outputs under `stata/output/`.

## Operational caveats

- The latest raw reporting week is not always suitable as a direct forecast anchor because reporting completeness can drop sharply at the end of the series.
- Current site-state fields and forecast fields are intentionally allowed to come from different effective weeks when that produces a more operationally honest output.
- Gemini chat depends on Vertex configuration and Application Default Credentials; if that configuration is missing, the viewer can still load, but chat will not be available.
- This repository does not include a formal automated test suite yet; validation is currently driven by artifact generation, metric review, and API-level spot checks.
