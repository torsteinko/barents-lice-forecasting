from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import (
    MODELS_DIR,
    PROCESSED_DIR,
    RAW_VLICE_PATH,
    RAW_VTREATMENT_PATH,
    RESULTS_DIR,
    SITE_WEEK_KEYS,
    YES_NO_MAP,
)

NUMERIC_VLICE_COLUMNS = [
    "week",
    "year",
    "sitenumber",
    "femaleadult",
    "mobilelice",
    "persistentlice",
    "municipalitynumber",
    "countynumber",
    "latitude",
    "longitude",
    "licelimitweek",
    "seatemperature",
    "productionareaid",
]

NUMERIC_TREATMENT_COLUMNS = [
    "week",
    "year",
    "sitenumber",
    "municipalitynumber",
    "countynumber",
    "latitude",
    "longitude",
    "productionareaid",
]

TEXT_VLICE_COLUMNS = ["sitename", "municipality", "county", "productionarea"]
TEXT_TREATMENT_COLUMNS = [
    "sitename",
    "action",
    "typeoftreatment",
    "activeingredient",
    "speciesid",
    "cleanerfish",
    "scope",
    "municipality",
    "county",
    "productionarea",
]


def ensure_output_dirs() -> None:
    for path in (PROCESSED_DIR, RESULTS_DIR, MODELS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def load_raw_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    vlice = pd.read_csv(RAW_VLICE_PATH)
    vtreatment = pd.read_csv(RAW_VTREATMENT_PATH)
    return vlice, vtreatment


def clean_vlice(vlice: pd.DataFrame) -> pd.DataFrame:
    frame = _standardize_columns(vlice)

    if "lice_hk" in frame.columns:
        frame = frame.drop_duplicates(subset=["lice_hk"]).copy()

    for column in ("likelynofish", "havecountedlice", "overthelicelimitweek"):
        frame[column] = frame[column].map(YES_NO_MAP).astype("boolean")

    for column in NUMERIC_VLICE_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    for column in TEXT_VLICE_COLUMNS:
        frame[column] = frame[column].fillna("").astype(str).str.strip()

    frame["week_start_date"] = _parse_week_start(frame["year"], frame["week"])
    frame = frame.sort_values(["sitenumber", "week_start_date"]).reset_index(drop=True)
    return frame


def clean_treatment(vtreatment: pd.DataFrame) -> pd.DataFrame:
    frame = _standardize_columns(vtreatment)

    for column in NUMERIC_TREATMENT_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    for column in TEXT_TREATMENT_COLUMNS:
        frame[column] = frame[column].fillna("").astype(str).str.strip()

    frame["week_start_date"] = _parse_week_start(frame["year"], frame["week"])
    frame = frame.sort_values(["sitenumber", "week_start_date"]).reset_index(drop=True)
    return frame


def build_audit_summary(
    vlice: pd.DataFrame, vtreatment: pd.DataFrame
) -> dict[str, object]:
    return {
        "vlice_rows": int(len(vlice)),
        "vtreatment_rows": int(len(vtreatment)),
        "vlice_min_year": int(vlice["year"].min()),
        "vlice_max_year": int(vlice["year"].max()),
        "vtreatment_min_year": int(vtreatment["year"].min()),
        "vtreatment_max_year": int(vtreatment["year"].max()),
        "vlice_unique_sites": int(vlice["sitenumber"].nunique()),
        "vtreatment_unique_sites": int(vtreatment["sitenumber"].nunique()),
        "vlice_unique_production_areas": int(vlice["productionareaid"].nunique()),
        "vtreatment_unique_production_areas": int(
            vtreatment["productionareaid"].nunique()
        ),
        "vlice_duplicate_site_weeks": int(vlice.duplicated(SITE_WEEK_KEYS).sum()),
        "vtreatment_site_weeks_with_multiple_rows": int(
            (vtreatment.groupby(SITE_WEEK_KEYS).size() > 1).sum()
        ),
        "vlice_counted_weeks": int(vlice["havecountedlice"].fillna(False).sum()),
        "vlice_likely_no_fish_weeks": int(vlice["likelynofish"].fillna(False).sum()),
    }


def _standardize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    renamed = frame.copy()
    renamed.columns = [column.strip().lower() for column in renamed.columns]
    renamed = renamed.rename(
        columns={
            "muncipalitynumber": "municipalitynumber",
            "muncipality": "municipality",
        }
    )
    return renamed


def _parse_week_start(year: pd.Series, week: pd.Series) -> pd.Series:
    iso_token = (
        year.astype("Int64").astype(str)
        + week.astype("Int64").astype(str).str.zfill(2)
        + "1"
    )
    return pd.to_datetime(iso_token, format="%G%V%u", errors="coerce")
