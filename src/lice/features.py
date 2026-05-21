from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .config import (
    HORIZONS,
    MAX_NEIGHBORS,
    NEIGHBOR_RADIUS_KM,
    SITE_WEEK_KEYS,
    TRAINING_MAX_YEAR,
)


def build_master_table(vlice: pd.DataFrame, vtreatment: pd.DataFrame) -> pd.DataFrame:
    aggregated_treatment = aggregate_treatments(vtreatment)
    master = vlice.merge(
        aggregated_treatment, on=SITE_WEEK_KEYS, how="left", validate="one_to_one"
    )

    fill_zero_columns = [
        "treatment_count",
        "any_treatment",
        "mechanical_treatment_count",
        "medicinal_treatment_count",
        "bath_treatment_count",
        "feed_treatment_count",
        "full_site_treatment_count",
        "partial_site_treatment_count",
        "cleanerfish_treatment_count",
        "active_ingredient_count",
    ]
    master[fill_zero_columns] = master[fill_zero_columns].fillna(0)
    master = master.sort_values(["sitenumber", "week_start_date"]).reset_index(
        drop=True
    )
    return master


def aggregate_treatments(vtreatment: pd.DataFrame) -> pd.DataFrame:
    frame = vtreatment.copy()
    for column in (
        "action",
        "typeoftreatment",
        "activeingredient",
        "scope",
        "cleanerfish",
    ):
        frame[column] = frame[column].fillna("").astype(str).str.strip().str.lower()

    frame["is_mechanical"] = (
        frame["action"].str.contains("mekanisk", na=False).astype(int)
    )
    frame["is_medicinal"] = (
        frame["action"].str.contains("medikament", na=False).astype(int)
    )
    frame["is_bath"] = (
        frame["typeoftreatment"].str.contains("bade", na=False).astype(int)
    )
    frame["is_feed"] = (
        frame["typeoftreatment"]
        .str.contains("f.rbehandling|forbehandling", regex=True, na=False)
        .astype(int)
    )
    frame["is_full_scope"] = frame["scope"].str.contains("hele", na=False).astype(int)
    frame["is_partial_scope"] = (
        frame["scope"].str.contains("deler", na=False).astype(int)
    )
    frame["has_cleanerfish"] = frame["cleanerfish"].ne("").astype(int)
    frame["activeingredient_nonempty"] = frame["activeingredient"].replace("", np.nan)

    aggregated = (
        frame.groupby(SITE_WEEK_KEYS)
        .agg(
            treatment_count=("sitenumber", "size"),
            mechanical_treatment_count=("is_mechanical", "sum"),
            medicinal_treatment_count=("is_medicinal", "sum"),
            bath_treatment_count=("is_bath", "sum"),
            feed_treatment_count=("is_feed", "sum"),
            full_site_treatment_count=("is_full_scope", "sum"),
            partial_site_treatment_count=("is_partial_scope", "sum"),
            cleanerfish_treatment_count=("has_cleanerfish", "sum"),
            active_ingredient_count=(
                "activeingredient_nonempty",
                lambda series: series.dropna().nunique(),
            ),
        )
        .reset_index()
    )
    aggregated["any_treatment"] = (aggregated["treatment_count"] > 0).astype(int)
    return aggregated


def create_targets(
    master: pd.DataFrame, horizons: tuple[int, ...] = HORIZONS
) -> pd.DataFrame:
    frame = master.sort_values(["sitenumber", "week_start_date"]).copy()

    can_label = (
        frame["havecountedlice"].fillna(False)
        & frame["femaleadult"].notna()
        & frame["licelimitweek"].notna()
    )
    reported_breach = frame["overthelicelimitweek"].notna() & frame[
        "havecountedlice"
    ].fillna(False)

    frame["breach_this_week"] = np.nan
    frame.loc[reported_breach, "breach_this_week"] = frame.loc[
        reported_breach, "overthelicelimitweek"
    ].astype(int)

    derived_breach = can_label & ~reported_breach
    frame.loc[derived_breach, "breach_this_week"] = (
        frame.loc[derived_breach, "femaleadult"]
        > frame.loc[derived_breach, "licelimitweek"]
    ).astype(int)

    frame["breach_label_source"] = np.where(
        reported_breach,
        "reported_flag",
        np.where(derived_breach, "derived_from_femaleadult", "unlabeled"),
    )

    grouped_breach = frame.groupby("sitenumber")["breach_this_week"]
    grouped_date = frame.groupby("sitenumber")["week_start_date"]

    for horizon in horizons:
        future_steps = [grouped_breach.shift(-step) for step in range(1, horizon + 1)]
        future_matrix = pd.concat(future_steps, axis=1)
        valid_future_window = future_matrix.notna().all(axis=1)
        future_counts = future_matrix.fillna(0).sum(axis=1)

        frame[f"target_{horizon}w_count"] = future_counts.where(valid_future_window)
        frame[f"target_{horizon}w_any"] = (
            (future_counts > 0).astype(float).where(valid_future_window)
        )

        window_end_date = grouped_date.shift(-horizon)
        frame[f"target_{horizon}w_window_end"] = window_end_date
        frame[f"target_{horizon}w_pre_2026_only"] = window_end_date.dt.year.le(
            TRAINING_MAX_YEAR
        )

    return frame


def add_baseline_features(master: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    frame = master.sort_values(["sitenumber", "week_start_date"]).copy()

    week_of_year = frame["week_start_date"].dt.isocalendar().week.astype(int)
    frame["week_sin"] = np.sin(2 * math.pi * week_of_year / 52)
    frame["week_cos"] = np.cos(2 * math.pi * week_of_year / 52)
    frame["month"] = frame["week_start_date"].dt.month.astype(int)
    frame["quarter"] = frame["week_start_date"].dt.quarter.astype(int)
    frame["year_index"] = frame["year"] - frame["year"].min()
    frame["likelynofish_flag"] = frame["likelynofish"].fillna(False).astype(int)
    frame["havecountedlice_flag"] = frame["havecountedlice"].fillna(False).astype(int)
    frame["seatemperature_missing"] = frame["seatemperature"].isna().astype(int)
    frame["femaleadult_to_limit_ratio"] = _safe_divide(
        frame["femaleadult"], frame["licelimitweek"]
    ).clip(upper=10)
    frame["mobile_to_female_ratio"] = _safe_divide(
        frame["mobilelice"], frame["femaleadult"] + 0.05
    ).clip(upper=20)
    frame["persistent_to_female_ratio"] = _safe_divide(
        frame["persistentlice"], frame["femaleadult"] + 0.05
    ).clip(upper=20)

    site_groups = frame.groupby("sitenumber", sort=False)
    previous_breach = site_groups["breach_this_week"].shift(1)
    previous_ratio = site_groups["femaleadult_to_limit_ratio"].shift(1)

    frame["site_breach_rate_expanding"] = (
        previous_breach.groupby(frame["sitenumber"])
        .expanding()
        .mean()
        .reset_index(level=0, drop=True)
    )
    frame["site_limit_ratio_expanding"] = (
        previous_ratio.groupby(frame["sitenumber"])
        .expanding()
        .mean()
        .reset_index(level=0, drop=True)
    )
    frame["weeks_since_last_counted"] = site_groups["havecountedlice_flag"].transform(
        _weeks_since_last_event
    )
    frame["weeks_since_last_breach"] = site_groups["breach_this_week"].transform(
        _weeks_since_last_positive
    )
    frame["weeks_since_any_treatment"] = site_groups["any_treatment"].transform(
        _weeks_since_last_positive
    )
    frame["weeks_since_medicinal_treatment"] = site_groups[
        "medicinal_treatment_count"
    ].transform(_weeks_since_last_positive)
    frame["weeks_since_mechanical_treatment"] = site_groups[
        "mechanical_treatment_count"
    ].transform(_weeks_since_last_positive)

    lag_columns = [
        "femaleadult",
        "mobilelice",
        "persistentlice",
        "seatemperature",
        "breach_this_week",
        "femaleadult_to_limit_ratio",
    ]
    lag_steps = (1, 2, 4, 8, 12)
    feature_columns = [
        "femaleadult",
        "mobilelice",
        "persistentlice",
        "seatemperature",
        "licelimitweek",
        "femaleadult_to_limit_ratio",
        "mobile_to_female_ratio",
        "persistent_to_female_ratio",
        "latitude",
        "longitude",
        "productionareaid",
        "countynumber",
        "week_sin",
        "week_cos",
        "month",
        "quarter",
        "year_index",
        "likelynofish_flag",
        "havecountedlice_flag",
        "seatemperature_missing",
        "site_breach_rate_expanding",
        "site_limit_ratio_expanding",
        "weeks_since_last_counted",
        "weeks_since_last_breach",
        "weeks_since_any_treatment",
        "weeks_since_medicinal_treatment",
        "weeks_since_mechanical_treatment",
        "treatment_count",
        "any_treatment",
        "mechanical_treatment_count",
        "medicinal_treatment_count",
        "bath_treatment_count",
        "feed_treatment_count",
        "full_site_treatment_count",
        "partial_site_treatment_count",
        "cleanerfish_treatment_count",
        "active_ingredient_count",
    ]

    for column in lag_columns:
        for step in lag_steps:
            feature_name = f"{column}_lag_{step}"
            frame[feature_name] = site_groups[column].shift(step)
            feature_columns.append(feature_name)

        lagged = site_groups[column].shift(1)
        rolling_group = lagged.groupby(frame["sitenumber"])
        mean_name = f"{column}_roll4_mean"
        max_name = f"{column}_roll12_max"
        std_name = f"{column}_roll12_std"
        frame[mean_name] = (
            rolling_group.rolling(window=4, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        frame[max_name] = (
            rolling_group.rolling(window=12, min_periods=1)
            .max()
            .reset_index(level=0, drop=True)
        )
        frame[std_name] = (
            rolling_group.rolling(window=12, min_periods=2)
            .std()
            .reset_index(level=0, drop=True)
        )
        feature_columns.extend([mean_name, max_name, std_name])

    prior_breach = site_groups["breach_this_week"].shift(1)
    prior_breach_group = prior_breach.groupby(frame["sitenumber"])
    for window in (4, 12, 26):
        feature_name = f"breach_this_week_roll{window}_sum"
        frame[feature_name] = (
            prior_breach_group.rolling(window=window, min_periods=1)
            .sum()
            .reset_index(level=0, drop=True)
        )
        feature_columns.append(feature_name)

    treatment_roll_columns = [
        "treatment_count",
        "mechanical_treatment_count",
        "medicinal_treatment_count",
        "any_treatment",
    ]

    for column in treatment_roll_columns:
        rolling_group = frame.groupby("sitenumber")[column]
        sum4_name = f"{column}_roll4_sum"
        sum12_name = f"{column}_roll12_sum"
        frame[sum4_name] = (
            rolling_group.rolling(window=4, min_periods=1)
            .sum()
            .reset_index(level=0, drop=True)
        )
        frame[sum12_name] = (
            rolling_group.rolling(window=12, min_periods=1)
            .sum()
            .reset_index(level=0, drop=True)
        )
        feature_columns.extend([sum4_name, sum12_name])

    production_area_features = _build_production_area_features(frame)
    frame = frame.merge(
        production_area_features,
        on=["productionareaid", "week_start_date"],
        how="left",
        validate="many_to_one",
    )
    feature_columns.extend(
        [
            column
            for column in production_area_features.columns
            if column not in {"productionareaid", "week_start_date"}
        ]
    )

    neighbor_features = _build_neighbor_features(frame)
    frame = frame.merge(
        neighbor_features,
        on=["sitenumber", "week_start_date"],
        how="left",
        validate="one_to_one",
    )
    feature_columns.extend(
        [
            column
            for column in neighbor_features.columns
            if column not in {"sitenumber", "week_start_date"}
        ]
    )

    feature_columns = list(dict.fromkeys(feature_columns))
    return frame, feature_columns


def _build_production_area_features(frame: pd.DataFrame) -> pd.DataFrame:
    area_week = (
        frame.groupby(["productionareaid", "week_start_date"], as_index=False)
        .agg(
            pa_active_sites=("sitenumber", "nunique"),
            pa_femaleadult_mean=("femaleadult", "mean"),
            pa_mobilelice_mean=("mobilelice", "mean"),
            pa_breach_rate=("breach_this_week", "mean"),
            pa_treatment_rate=("any_treatment", "mean"),
            pa_temperature_mean=("seatemperature", "mean"),
        )
        .sort_values(["productionareaid", "week_start_date"])
        .reset_index(drop=True)
    )

    area_groups = area_week.groupby("productionareaid", sort=False)
    derived_columns: list[str] = []
    for column in [
        "pa_active_sites",
        "pa_femaleadult_mean",
        "pa_mobilelice_mean",
        "pa_breach_rate",
        "pa_treatment_rate",
        "pa_temperature_mean",
    ]:
        lagged = area_groups[column].shift(1)
        lag_name = f"{column}_lag1"
        roll_name = f"{column}_roll4_mean"
        area_week[lag_name] = lagged
        area_week[roll_name] = (
            lagged.groupby(area_week["productionareaid"])
            .rolling(window=4, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        derived_columns.extend([lag_name, roll_name])

    return area_week[["productionareaid", "week_start_date", *derived_columns]]


def _build_neighbor_features(frame: pd.DataFrame) -> pd.DataFrame:
    site_locations = (
        frame[["sitenumber", "latitude", "longitude"]]
        .dropna()
        .drop_duplicates("sitenumber")
        .sort_values("sitenumber")
        .reset_index(drop=True)
    )
    if site_locations.empty:
        return frame[["sitenumber", "week_start_date"]].drop_duplicates().copy()

    neighbor_edges = _build_neighbor_edges(site_locations)
    if neighbor_edges.empty:
        return frame[["sitenumber", "week_start_date"]].drop_duplicates().copy()

    lag_source_columns = [
        "femaleadult",
        "mobilelice",
        "persistentlice",
        "breach_this_week",
        "any_treatment",
        "femaleadult_to_limit_ratio",
    ]
    lagged = frame[["sitenumber", "week_start_date", *lag_source_columns]].copy()
    lag_groups = lagged.groupby("sitenumber", sort=False)
    rename_map = {"sitenumber": "neighbor_sitenumber"}
    lagged_metric_columns: list[str] = []

    for column in lag_source_columns:
        lag_name = f"neighbor_{column}_lag1"
        lagged[lag_name] = lag_groups[column].shift(1)
        lagged_metric_columns.append(lag_name)
        rename_map[column] = f"raw_{column}"

    lagged = lagged.rename(columns=rename_map)
    joined = neighbor_edges.merge(
        lagged[["neighbor_sitenumber", "week_start_date", *lagged_metric_columns]],
        on="neighbor_sitenumber",
        how="left",
        validate="many_to_many",
    )

    for column in lagged_metric_columns:
        valid_weight_column = f"{column}_valid_weight"
        weighted_column = f"{column}_weighted"
        joined[valid_weight_column] = joined["neighbor_weight"].where(
            joined[column].notna(),
            0.0,
        )
        joined[weighted_column] = (
            joined[column].fillna(0.0) * joined[valid_weight_column]
        )

    aggregation_columns: dict[str, tuple[str, str]] = {
        "neighbor_site_count": ("neighbor_sitenumber", "count"),
        "neighbor_distance_mean_km": ("neighbor_distance_km", "mean"),
    }
    for column in lagged_metric_columns:
        aggregation_columns[f"{column}_valid_weight_sum"] = (
            f"{column}_valid_weight",
            "sum",
        )
        aggregation_columns[f"{column}_weighted_sum"] = (f"{column}_weighted", "sum")

    aggregated = (
        joined.groupby(["sitenumber", "week_start_date"], as_index=False)
        .agg(**aggregation_columns)
        .sort_values(["sitenumber", "week_start_date"])
        .reset_index(drop=True)
    )

    for column in lagged_metric_columns:
        aggregated[column] = _safe_divide(
            aggregated[f"{column}_weighted_sum"],
            aggregated[f"{column}_valid_weight_sum"],
        )

    keep_columns = [
        "sitenumber",
        "week_start_date",
        "neighbor_site_count",
        "neighbor_distance_mean_km",
    ]
    keep_columns.extend(lagged_metric_columns)
    return aggregated[keep_columns]


def _build_neighbor_edges(site_locations: pd.DataFrame) -> pd.DataFrame:
    site_ids = site_locations["sitenumber"].to_numpy()
    lat = np.radians(site_locations["latitude"].to_numpy())
    lon = np.radians(site_locations["longitude"].to_numpy())

    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat)[:, None] * np.cos(lat)[None, :] * np.sin(dlon / 2.0) ** 2
    )
    distances = 2.0 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    np.fill_diagonal(distances, np.inf)

    edges: list[dict[str, float | int]] = []
    for index, site_id in enumerate(site_ids):
        valid_indices = np.flatnonzero(distances[index] <= NEIGHBOR_RADIUS_KM)
        if len(valid_indices) == 0:
            valid_indices = np.argsort(distances[index])[:MAX_NEIGHBORS]
        else:
            valid_indices = valid_indices[
                np.argsort(distances[index, valid_indices])[:MAX_NEIGHBORS]
            ]

        for neighbor_index in valid_indices:
            distance_km = float(distances[index, neighbor_index])
            if not np.isfinite(distance_km):
                continue
            edges.append(
                {
                    "sitenumber": int(site_id),
                    "neighbor_sitenumber": int(site_ids[neighbor_index]),
                    "neighbor_distance_km": distance_km,
                    "neighbor_weight": 1.0 / (1.0 + distance_km),
                }
            )

    return pd.DataFrame(edges)


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    divided = numerator / denominator.replace(0, np.nan)
    return divided.replace([np.inf, -np.inf], np.nan)


def _weeks_since_last_event(series: pd.Series) -> pd.Series:
    values = series.fillna(0).astype(int).to_numpy()
    output = np.full(len(values), np.nan)
    last_event_index = None
    for index, value in enumerate(values):
        if value > 0:
            last_event_index = index
            output[index] = 0.0
        elif last_event_index is not None:
            output[index] = float(index - last_event_index)
    return pd.Series(output, index=series.index)


def _weeks_since_last_positive(series: pd.Series) -> pd.Series:
    binary_series = series.fillna(0).gt(0).astype(int)
    return _weeks_since_last_event(binary_series)
