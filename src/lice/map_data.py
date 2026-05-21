from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

MAP_STATUS_COLUMNS = [
    "sitenumber",
    "sitename",
    "municipality",
    "county",
    "productionareaid",
    "productionarea",
    "latitude",
    "longitude",
    "week_start_date",
    "femaleadult",
    "mobilelice",
    "persistentlice",
    "licelimitweek",
    "femaleadult_to_limit_ratio",
    "breach_this_week",
    "overthelicelimitweek",
    "havecountedlice",
    "likelynofish",
    "seatemperature",
    "treatment_count",
    "any_treatment",
    "mechanical_treatment_count",
    "medicinal_treatment_count",
    "bath_treatment_count",
    "feed_treatment_count",
    "cleanerfish_treatment_count",
    "weeks_since_last_counted",
    "weeks_since_last_breach",
    "weeks_since_any_treatment",
    "weeks_since_medicinal_treatment",
    "weeks_since_mechanical_treatment",
    "pa_breach_rate_lag1",
    "pa_treatment_rate_lag1",
    "neighbor_site_count",
    "neighbor_breach_this_week_lag1",
    "neighbor_femaleadult_to_limit_ratio_lag1",
]


def write_site_map_geojson(
    master: pd.DataFrame,
    predictions: pd.DataFrame,
    vtreatment: pd.DataFrame,
    path: Path,
) -> pd.DataFrame:
    site_map = build_site_map_frame(master, predictions, vtreatment)
    geojson = build_site_map_geojson(site_map)
    path.write_text(json.dumps(geojson, indent=2), encoding="utf-8")
    return site_map


def build_site_map_frame(
    master: pd.DataFrame,
    predictions: pd.DataFrame,
    vtreatment: pd.DataFrame,
) -> pd.DataFrame:
    latest_status = _build_latest_status(master)
    latest_treatment = _build_latest_treatment(vtreatment)
    prediction_snapshot = _build_prediction_snapshot(predictions)

    frame = latest_status.merge(
        latest_treatment,
        on="sitenumber",
        how="left",
        validate="one_to_one",
    )
    frame = frame.merge(
        prediction_snapshot,
        on="sitenumber",
        how="left",
        validate="one_to_one",
    )

    classifier_score_columns = [
        column
        for column in [
            "classifier_1w_score",
            "classifier_2w_score",
            "classifier_12w_score",
        ]
        if column in frame.columns
    ]
    if classifier_score_columns:
        score_frame = frame[classifier_score_columns]
        frame["max_breach_risk"] = score_frame.max(axis=1, skipna=True)
        best_score_column = score_frame.fillna(-np.inf).idxmax(axis=1)
        frame["priority_horizon"] = best_score_column.str.extract(
            r"classifier_(\d+w)_score",
            expand=False,
        )
        frame.loc[score_frame.isna().all(axis=1), "priority_horizon"] = None
    else:
        frame["max_breach_risk"] = np.nan
        frame["priority_horizon"] = None

    frame["current_limit_excess"] = (
        frame["femaleadult"].fillna(0.0) - frame["licelimitweek"].fillna(np.inf)
    ).clip(lower=0.0)
    frame["currently_over_limit"] = np.where(
        frame["overthelicelimitweek"].notna(),
        frame["overthelicelimitweek"],
        frame["femaleadult"].fillna(-np.inf) > frame["licelimitweek"].fillna(np.inf),
    )
    frame["risk_band"] = np.select(
        [
            frame["max_breach_risk"].fillna(0.0) >= 0.8,
            frame["max_breach_risk"].fillna(0.0) >= 0.6,
            frame["max_breach_risk"].fillna(0.0) >= 0.4,
        ],
        ["critical", "high", "watch"],
        default="stable",
    )
    frame["latest_observation_date"] = frame["week_start_date"].dt.date.astype(str)
    frame = frame.sort_values(
        ["max_breach_risk", "femaleadult_to_limit_ratio", "femaleadult"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    return frame


def build_site_map_geojson(site_map: pd.DataFrame) -> dict[str, object]:
    features: list[dict[str, object]] = []
    property_columns = [
        column for column in site_map.columns if column not in {"latitude", "longitude"}
    ]

    for row in site_map.itertuples(index=False):
        latitude = getattr(row, "latitude", None)
        longitude = getattr(row, "longitude", None)
        if pd.isna(latitude) or pd.isna(longitude):
            continue

        properties = {
            column: _jsonify(getattr(row, column)) for column in property_columns
        }
        features.append(
            {
                "type": "Feature",
                "properties": properties,
                "geometry": {
                    "type": "Point",
                    "coordinates": [float(longitude), float(latitude)],
                },
            }
        )

    return {
        "type": "FeatureCollection",
        "metadata": {
            "feature_count": len(features),
            "description": "Latest site snapshot with latest holdout predictions and treatment context.",
        },
        "features": features,
    }


def _build_latest_status(master: pd.DataFrame) -> pd.DataFrame:
    available_columns = [
        column for column in MAP_STATUS_COLUMNS if column in master.columns
    ]
    latest_status = (
        master[available_columns]
        .sort_values(["sitenumber", "week_start_date"])
        .groupby("sitenumber", as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )
    return latest_status


def _build_latest_treatment(vtreatment: pd.DataFrame) -> pd.DataFrame:
    treatment_columns = [
        "sitenumber",
        "week_start_date",
        "action",
        "typeoftreatment",
        "activeingredient",
        "scope",
        "cleanerfish",
    ]
    latest_treatment = (
        vtreatment[treatment_columns]
        .sort_values(["sitenumber", "week_start_date"])
        .groupby("sitenumber", as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )
    return latest_treatment.rename(
        columns={
            "week_start_date": "last_treatment_date",
            "action": "last_treatment_action",
            "typeoftreatment": "last_treatment_type",
            "activeingredient": "last_treatment_activeingredient",
            "scope": "last_treatment_scope",
            "cleanerfish": "last_treatment_cleanerfish",
        }
    )


def _build_prediction_snapshot(predictions: pd.DataFrame) -> pd.DataFrame:
    merged: pd.DataFrame | None = None

    for horizon in sorted(predictions["horizon"].dropna().unique()):
        horizon_int = int(horizon)
        classifier_snapshot = _latest_prediction_rows(
            predictions,
            horizon_int,
            "classifier_any",
            {
                "week_start_date": f"classifier_{horizon_int}w_date",
                "actual": f"classifier_{horizon_int}w_actual",
                "prediction": f"classifier_{horizon_int}w_prediction",
                "score": f"classifier_{horizon_int}w_score",
                "candidate_model": f"classifier_{horizon_int}w_model",
                "decision_threshold": f"classifier_{horizon_int}w_threshold",
            },
        )
        regressor_snapshot = _latest_prediction_rows(
            predictions,
            horizon_int,
            "regressor_count",
            {
                "week_start_date": f"count_{horizon_int}w_date",
                "actual": f"count_{horizon_int}w_actual",
                "prediction": f"count_{horizon_int}w_prediction",
                "candidate_model": f"count_{horizon_int}w_model",
            },
        )

        for snapshot in (classifier_snapshot, regressor_snapshot):
            if merged is None:
                merged = snapshot
            else:
                merged = merged.merge(
                    snapshot,
                    on="sitenumber",
                    how="outer",
                    validate="one_to_one",
                )

    if merged is None:
        return pd.DataFrame(columns=["sitenumber"])
    return merged


def _latest_prediction_rows(
    predictions: pd.DataFrame,
    horizon: int,
    model_type: str,
    rename_map: dict[str, str],
) -> pd.DataFrame:
    subset = predictions[
        (predictions["horizon"] == horizon) & (predictions["model_type"] == model_type)
    ].copy()
    if subset.empty:
        return pd.DataFrame(columns=["sitenumber", *rename_map.values()])

    latest_rows = (
        subset.sort_values(["sitenumber", "week_start_date"])
        .groupby("sitenumber", as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )
    keep_columns = ["sitenumber", *rename_map.keys()]
    latest_rows = latest_rows[keep_columns].rename(columns=rename_map)
    return latest_rows


def _jsonify(value):
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, str) and not value.strip():
        return None
    return value
