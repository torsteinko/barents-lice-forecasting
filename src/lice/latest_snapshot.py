from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    HORIZONS,
    LATEST_MASTER_TABLE_PATH,
    LATEST_PREDICTIONS_PATH,
    LATEST_SITE_MAP_PATH,
    LATEST_SITE_SNAPSHOT_PATH,
    MODEL_METRICS_PATH,
    MODELS_DIR,
)
from .data import clean_treatment, clean_vlice, ensure_output_dirs, load_raw_tables
from .features import add_baseline_features, build_master_table, create_targets
from .map_data import write_site_map_geojson
from .model import build_feature_matrix


def main() -> None:
    ensure_output_dirs()

    raw_vlice, raw_treatment = load_raw_tables()
    vlice = clean_vlice(raw_vlice)
    vtreatment = clean_treatment(raw_treatment)

    master = build_master_table(vlice, vtreatment)
    master = create_targets(master, horizons=HORIZONS)
    master, feature_columns = add_baseline_features(master)
    master.to_parquet(LATEST_MASTER_TABLE_PATH, index=False)

    latest_rows, latest_dataset_date = build_latest_snapshot_rows(master)
    model_specs = load_model_specs()
    predictions = score_latest_site_rows(latest_rows, feature_columns, model_specs)
    predictions.to_csv(LATEST_PREDICTIONS_PATH, index=False)

    site_map = write_site_map_geojson(
        latest_rows,
        predictions,
        vtreatment,
        LATEST_SITE_MAP_PATH,
    )
    site_map.to_csv(LATEST_SITE_SNAPSHOT_PATH, index=False)

    print("Saved latest master table to", LATEST_MASTER_TABLE_PATH)
    print("Saved latest predictions to", LATEST_PREDICTIONS_PATH)
    print("Saved latest site snapshot to", LATEST_SITE_SNAPSHOT_PATH)
    print("Saved latest map data to", LATEST_SITE_MAP_PATH)
    if pd.notna(latest_dataset_date):
        print("Latest dataset week:", latest_dataset_date.date().isoformat())
    print("Scored site rows:", len(latest_rows))

    top_columns = [
        column
        for column in [
            "sitenumber",
            "sitename",
            "productionarea",
            "latest_reporting_week_label",
            "classifier_12w_score",
            "count_12w_prediction",
        ]
        if column in site_map.columns
    ]
    top_sites = site_map.sort_values(
        ["classifier_12w_score", "count_12w_prediction", "femaleadult_to_limit_ratio"],
        ascending=[False, False, False],
    )
    print(top_sites[top_columns].head(10).to_string(index=False))


def build_latest_snapshot_rows(
    master: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Timestamp]:
    dated = master[master["week_start_date"].notna()].copy()
    if dated.empty:
        raise ValueError("No dated rows are available for latest snapshot scoring.")

    latest_dataset_date = pd.to_datetime(dated["week_start_date"], errors="coerce").max()
    snapshot_rows = dated[dated["week_start_date"].eq(latest_dataset_date)].copy()
    if snapshot_rows.empty:
        raise ValueError("The latest reporting week has no rows available for scoring.")

    snapshot_rows = snapshot_rows.sort_values(["sitenumber", "week_start_date"]).reset_index(drop=True)
    return snapshot_rows, latest_dataset_date


def load_model_specs() -> dict[tuple[int, str], dict[str, object]]:
    if not MODEL_METRICS_PATH.exists():
        raise FileNotFoundError(f"Missing model metrics at {MODEL_METRICS_PATH}")

    metrics = json.loads(MODEL_METRICS_PATH.read_text(encoding="utf-8"))
    model_specs: dict[tuple[int, str], dict[str, object]] = {}

    for horizon in HORIZONS:
        horizon_key = f"{horizon}w"
        classifier_metrics = metrics[horizon_key]["classifier_any"]
        regressor_metrics = metrics[horizon_key]["regressor_count"]

        model_specs[(horizon, "classifier_any")] = {
            "model": load_pickle(MODELS_DIR / f"classifier_{horizon_key}.pkl"),
            "candidate_model": str(classifier_metrics["selected_model"]),
            "decision_threshold": float(classifier_metrics["decision_threshold"]),
        }
        model_specs[(horizon, "regressor_count")] = {
            "model": load_pickle(MODELS_DIR / f"regressor_{horizon_key}.pkl"),
            "candidate_model": str(regressor_metrics["selected_model"]),
            "decision_threshold": None,
        }

    return model_specs


def load_pickle(path: Path) -> object:
    if not path.exists():
        raise FileNotFoundError(f"Missing serialized model at {path}")

    with open(path, "rb") as handle:
        return pickle.load(handle)


def score_latest_site_rows(
    latest_rows: pd.DataFrame,
    feature_columns: list[str],
    model_specs: dict[tuple[int, str], dict[str, object]],
) -> pd.DataFrame:
    feature_matrix = build_feature_matrix(latest_rows, feature_columns)
    classifier_scores_by_horizon: dict[int, np.ndarray] = {}
    regressor_predictions_by_horizon: dict[int, np.ndarray] = {}
    prediction_frames: list[pd.DataFrame] = []

    for horizon in HORIZONS:
        classifier_spec = model_specs[(horizon, "classifier_any")]
        classifier_scores_by_horizon[horizon] = np.asarray(
            classifier_spec["model"].predict_proba(feature_matrix)[:, 1],
            dtype=float,
        )

        regressor_spec = model_specs[(horizon, "regressor_count")]
        regressor_predictions_by_horizon[horizon] = np.clip(
            np.asarray(regressor_spec["model"].predict(feature_matrix), dtype=float),
            0,
            None,
        )

    classifier_scores_by_horizon = enforce_non_decreasing_horizon_values(
        classifier_scores_by_horizon
    )
    regressor_predictions_by_horizon = enforce_non_decreasing_horizon_values(
        regressor_predictions_by_horizon
    )

    for horizon in HORIZONS:
        classifier_spec = model_specs[(horizon, "classifier_any")]
        classifier_scores = classifier_scores_by_horizon[horizon]
        classifier_threshold = float(classifier_spec["decision_threshold"])
        classifier_predictions = (classifier_scores >= classifier_threshold).astype(int)
        prediction_frames.append(
            build_scored_prediction_frame(
                latest_rows,
                horizon,
                "classifier_any",
                classifier_predictions,
                classifier_scores,
                str(classifier_spec["candidate_model"]),
                classifier_threshold,
            )
        )

        regressor_spec = model_specs[(horizon, "regressor_count")]
        regressor_predictions = regressor_predictions_by_horizon[horizon]
        prediction_frames.append(
            build_scored_prediction_frame(
                latest_rows,
                horizon,
                "regressor_count",
                regressor_predictions,
                regressor_predictions,
                str(regressor_spec["candidate_model"]),
                None,
            )
        )

    return pd.concat(prediction_frames, ignore_index=True)


def enforce_non_decreasing_horizon_values(
    values_by_horizon: dict[int, np.ndarray],
) -> dict[int, np.ndarray]:
    adjusted: dict[int, np.ndarray] = {}
    running_values: np.ndarray | None = None

    for horizon in sorted(values_by_horizon):
        current_values = np.asarray(values_by_horizon[horizon], dtype=float)
        if running_values is None:
            running_values = current_values.copy()
        else:
            running_values = np.maximum(running_values, current_values)
        adjusted[horizon] = running_values.copy()

    return adjusted


def build_scored_prediction_frame(
    latest_rows: pd.DataFrame,
    horizon: int,
    model_type: str,
    predictions: np.ndarray,
    scores: np.ndarray,
    candidate_model: str,
    decision_threshold: float | None,
) -> pd.DataFrame:
    frame = latest_rows[
        [
            "sitenumber",
            "sitename",
            "productionareaid",
            "productionarea",
            "week_start_date",
        ]
    ].copy()
    frame["actual"] = np.nan
    frame["prediction"] = predictions
    frame["score"] = scores
    frame["candidate_model"] = candidate_model
    frame["decision_threshold"] = decision_threshold
    frame["horizon"] = horizon
    frame["model_type"] = model_type
    return frame