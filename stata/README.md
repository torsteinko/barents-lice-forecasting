# Task 5: Stata Port To Python

This folder ports the Atlantic salmon growth model from the Stata snippet into reusable Python code and a presentation notebook.

## What is included

- `salmon_growth.py`: reusable growth model, CLI entrypoint, CSV/JSON/SVG export.
- `salmon_growth_walkthrough.ipynb`: interview-friendly notebook with narrative, tables, and an interactive hover chart.
- `__main__.py`: enables `python -m stata` from the repository root.
- `output/`: generated artifacts after running the script.

## Enhancements beyond the original Stata code

- Parameterized CLI for temperatures, horizon length, starting weight, and calibration inputs.
- Structured exports for downstream analysis: wide CSV, long CSV, JSON summary, interactive HTML chart, and optional SVG fallback.
- Built-in validation checks so the calibration point is explicit and reproducible.
- Reusable functions that can be imported into notebooks or integrated into a larger pipeline.
- Interactive hoverable visualization for presentation and manual inspection.

## Calibration note

The original Stata comments say the model is calibrated to `1.53 %/day` at `W = 150 g` and `T = 14C`, but the executed formula uses `Topt_base = 13.5C`. That means the effective value at `14C` is slightly lower at roughly `1.523 %/day`. The Python port preserves the executed Stata logic and reports that delta in the JSON summary.

## Run it

From the repository root:

```powershell
.\.venv\Scripts\python.exe -m stata
```

Or with custom parameters:

```powershell
.\.venv\Scripts\python.exe -m stata --days 500 --temperatures-c 6,10,14,18 --output-dir stata/output_custom
```

To open the notebook walkthrough, use [stata/salmon_growth_walkthrough.ipynb](../stata/salmon_growth_walkthrough.ipynb).

## Outputs

- `weights_wide.csv`: day-by-day weights in Stata-style wide format.
- `weights_long.csv`: tidy long format with SGR and daily gain.
- `growth_summary.json`: ranked scenario summary and validation results.
- `growth_curves.html`: interactive chart with hover labels for exact values.
- `growth_curves.svg`: static fallback chart.