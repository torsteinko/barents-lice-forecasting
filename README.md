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

## Interactive map

Serve the repository root and open the static viewer:

```powershell
python -m http.server 8765
```

Then open `http://127.0.0.1:8765/viewer/` in a browser.

The map shows each site with current status, last treatment context, surrounding pressure, and the latest validated holdout predictions.

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
- `viewer/`: static MapLibre viewer for site-level inspection.

## Current modeling scope

- Strictly excludes prediction rows whose target window would extend into 2026.
- Uses lagged site history, production-area pressure, and 50 km neighbor-pressure features.
- Keeps `HistGradientBoosting` for the 1w and 2w classifiers.
- Uses GPU XGBoost for all current count regressors and the 12w classifier where focused benchmarks improved holdout performance.
- Keeps the code small and easy to extend before adding cross-validation, SHAP, or deep learning.
