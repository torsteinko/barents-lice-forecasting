from __future__ import annotations

import json

import pandas as pd

from .config import HORIZONS, PROCESSED_DIR, RESULTS_DIR, SITE_WEEK_KEYS
from .data import (
    build_audit_summary,
    clean_treatment,
    clean_vlice,
    ensure_output_dirs,
    load_raw_tables,
)
from .features import add_baseline_features, build_master_table, create_targets
from .map_data import write_site_map_geojson
from .model import train_all_models


def main() -> None:
    ensure_output_dirs()

    raw_vlice, raw_treatment = load_raw_tables()
    vlice = clean_vlice(raw_vlice)
    vtreatment = clean_treatment(raw_treatment)

    audit_summary = build_audit_summary(vlice, vtreatment)
    save_json(RESULTS_DIR / "data_audit.json", audit_summary)

    if audit_summary["vlice_duplicate_site_weeks"] != 0:
        raise ValueError("vlice is not unique at site-week grain.")

    master = build_master_table(vlice, vtreatment)
    if int(master.duplicated(SITE_WEEK_KEYS).sum()) != 0:
        raise ValueError(
            "Master table is not unique at site-week grain after joining treatments."
        )

    master = create_targets(master, horizons=HORIZONS)
    master, feature_columns = add_baseline_features(master)
    master.to_parquet(PROCESSED_DIR / "master_table.parquet", index=False)

    pd.DataFrame({"feature": feature_columns}).to_csv(
        RESULTS_DIR / "feature_columns.csv", index=False
    )

    metrics, predictions, comparison, error_summary = train_all_models(
        master, feature_columns
    )
    save_json(RESULTS_DIR / "model_metrics.json", metrics)
    predictions.to_csv(RESULTS_DIR / "holdout_predictions.csv", index=False)
    comparison.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
    error_summary.to_csv(RESULTS_DIR / "error_summary_by_area.csv", index=False)
    write_site_map_geojson(
        master, predictions, vtreatment, RESULTS_DIR / "site_map.geojson"
    )

    print("Saved audit to", RESULTS_DIR / "data_audit.json")
    print("Saved master table to", PROCESSED_DIR / "master_table.parquet")
    print("Saved metrics to", RESULTS_DIR / "model_metrics.json")
    print("Saved predictions to", RESULTS_DIR / "holdout_predictions.csv")
    print("Saved model comparison to", RESULTS_DIR / "model_comparison.csv")
    print("Saved area error summary to", RESULTS_DIR / "error_summary_by_area.csv")
    print("Saved site map data to", RESULTS_DIR / "site_map.geojson")
    print(json.dumps(metrics, indent=2))


def save_json(path, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
