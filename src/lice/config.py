from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
RAW_VLICE_PATH = DATA_DIR / "vlice.csv"
RAW_VTREATMENT_PATH = DATA_DIR / "vtreatment.csv"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = ROOT_DIR / "results"
MODELS_DIR = RESULTS_DIR / "models"
MASTER_TABLE_PATH = PROCESSED_DIR / "master_table.parquet"
LATEST_MASTER_TABLE_PATH = PROCESSED_DIR / "master_table_latest.parquet"
SITE_MAP_PATH = RESULTS_DIR / "site_map.geojson"
LATEST_SITE_MAP_PATH = RESULTS_DIR / "site_map_latest.geojson"
MODEL_METRICS_PATH = RESULTS_DIR / "model_metrics.json"
LATEST_PREDICTIONS_PATH = RESULTS_DIR / "latest_predictions.csv"
LATEST_SITE_SNAPSHOT_PATH = RESULTS_DIR / "latest_site_snapshot.csv"

RANDOM_SEED = 42
HORIZONS = (1, 2, 12)
HOLDOUT_WEEKS = 12
CALIBRATION_WEEKS = 12
TRAINING_MAX_YEAR = 2025
SITE_WEEK_KEYS = ["sitenumber", "year", "week"]
YES_NO_MAP = {"Ja": True, "Nei": False}
NEIGHBOR_RADIUS_KM = 50.0
MAX_NEIGHBORS = 10
