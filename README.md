# Barents Lice Forecasting

Minimal first-pass implementation.

This baseline does four things:

1. Loads and audits the raw BarentsWatch lice and treatment CSV files.
2. Builds a single site-week master table anchored on `vlice`.
3. Creates leakage-safe targets for 1, 2, and 12 week breach forecasting.
4. Trains horizon-specific models and saves holdout metrics, predictions, and a map-ready site snapshot.

## Setup

```powershell
python -m pip install -r requirements.txt
```

## Run

```powershell
python run_baseline.py
```

## Notebook

Open `notebooks/01_baseline_walkthrough.ipynb` in VS Code and select the `.venv` kernel.

The notebook is artifact-driven by default and now explains:

- the neighbor-pressure feature slice,
- the short-horizon XGBoost benchmark outcome,
- the selective GPU model policy,
- and the interactive map workflow.

## Interactive map and API

Start the FastAPI app:

```powershell
python run_web.py
```

Then open `http://127.0.0.1:8000/` in a browser.

If you instead serve the frontend statically, for example with `python -m http.server 8765`, the chat and `/api/sites` calls still need the FastAPI backend running separately. The viewer now defaults to `http://127.0.0.1:8000` as the backend when opened from a localhost static server.

Static frontend plus FastAPI backend:

```powershell
python run_web.py
python -m http.server 8765
```

Then open `http://127.0.0.1:8765/viewer/index.html`.

If your backend is on a different origin, append `?api=http://host:port` to the viewer URL.

The viewer is now a full-screen map application with:

- floating filter and detail panels,
- risk-status filtering on the active breach metric,
- site search, county, and production-area filters,
- a hovering chat box that can rank sites and jump the map to a returned site,
- and FastAPI endpoints for the site dataset and chat workflow.

Primary endpoints:

- `GET /api/health`
- `GET /api/sites`
- `POST /chat`

The chat endpoint now uses a read-only LangChain SQL agent backed by DuckDB over `data/processed/master_table.parquet`.

- Each request builds two SQL views: `visible_master_table` for the current map scope and `master_table` for the full validated dataset.
- The agent defaults to `visible_master_table` and only broadens to `master_table` when the user explicitly asks for all sites or broader history.
- The backend enforces read-only SQL server-side and blocks non-`SELECT`/`WITH` statements and file-reading SQL functions before execution.

Vertex Gemini configuration is required for `/chat`. Environment variables are shown in `.env.example`:

- `GOOGLE_CLOUD_PROJECT`
- `GOOGLE_CLOUD_LOCATION`
- `VERTEX_GEMINI_MODEL` with default `gemini-3.1-flash-lite`

Authentication should come from Application Default Credentials, for example through `gcloud auth application-default login` or a service-account-backed environment.

## Outputs

- `data/processed/master_table.parquet`: cleaned master site-week table with targets and features.
- `results/data_audit.json`: dataset audit summary.
- `results/model_metrics.json`: holdout metrics for all trained baselines.
- `results/holdout_predictions.csv`: holdout predictions for inspection.
- `results/xgb_short_horizon_benchmark.csv`: focused 1w and 2w XGBoost benchmark used to decide the short-horizon policy.
- `results/site_map.geojson`: latest site snapshot used by the interactive map viewer.
- `results/feature_columns.csv`: feature list used by the models.
- `results/models/*.pkl`: serialized baseline models.
- `notebooks/01_baseline_walkthrough.ipynb`: presentation notebook built on top of the reusable pipeline code.
- `viewer/`: frontend assets served by the FastAPI map application.

## Current modeling scope

- Strictly excludes prediction rows whose target window would extend into 2026.
- Uses lagged site history, production-area pressure, and 50 km neighbor-pressure features.
- Keeps `HistGradientBoosting` for the 1w and 2w classifiers.
- Uses GPU XGBoost for all current count regressors and the 12w classifier where focused benchmarks improved holdout performance.
- Keeps the code small and easy to extend before adding cross-validation, SHAP, or deep learning.
