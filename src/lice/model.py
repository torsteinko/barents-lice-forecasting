from __future__ import annotations

import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .config import (
    CALIBRATION_WEEKS,
    HOLDOUT_WEEKS,
    HORIZONS,
    MODELS_DIR,
    RANDOM_SEED,
)

try:
    import xgboost as xgb
except ImportError:
    xgb = None

warnings.filterwarnings(
    "ignore",
    message=".*mismatched devices.*",
    category=UserWarning,
)


def train_all_models(
    master: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[
    dict[str, dict[str, dict[str, float | int | str]]],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    metrics: dict[str, dict[str, dict[str, float | int | str]]] = {}
    prediction_frames: list[pd.DataFrame] = []
    comparison_rows: list[dict[str, float | int | str | bool | None]] = []

    for horizon in HORIZONS:
        horizon_key = f"{horizon}w"
        metrics[horizon_key] = {}

        classifier_label = f"target_{horizon}w_any"
        classifier_splits = split_train_calibration_holdout(
            master,
            classifier_label,
            horizon,
        )
        classifier_metrics, classifier_predictions, classifier_comparison = (
            select_and_score_classifier(
                classifier_splits,
                feature_columns,
                classifier_label,
            )
        )
        comparison_rows.extend(classifier_comparison)
        save_model(
            classifier_metrics.pop("selected_model_object"),
            MODELS_DIR / f"classifier_{horizon_key}.pkl",
        )
        classifier_predictions["horizon"] = horizon
        classifier_predictions["model_type"] = "classifier_any"
        prediction_frames.append(classifier_predictions)
        metrics[horizon_key]["classifier_any"] = classifier_metrics

        regressor_label = f"target_{horizon}w_count"
        regressor_splits = split_train_calibration_holdout(
            master,
            regressor_label,
            horizon,
        )
        regressor_metrics, regressor_predictions, regressor_comparison = (
            select_and_score_regressor(
                regressor_splits,
                feature_columns,
                regressor_label,
            )
        )
        comparison_rows.extend(regressor_comparison)
        save_model(
            regressor_metrics.pop("selected_model_object"),
            MODELS_DIR / f"regressor_{horizon_key}.pkl",
        )
        regressor_predictions["horizon"] = horizon
        regressor_predictions["model_type"] = "regressor_count"
        prediction_frames.append(regressor_predictions)
        metrics[horizon_key]["regressor_count"] = regressor_metrics

    predictions = pd.concat(prediction_frames, ignore_index=True)
    comparison = pd.DataFrame(comparison_rows)
    if not comparison.empty:
        comparison = comparison.sort_values(
            ["horizon", "task_type", "selected", "selection_score"],
            ascending=[True, True, False, False],
        ).reset_index(drop=True)

    error_summary = summarize_classifier_errors(predictions)
    return metrics, predictions, comparison, error_summary


def select_and_score_classifier(
    splits: dict[str, pd.DataFrame | int],
    feature_columns: list[str],
    label_column: str,
) -> tuple[
    dict[str, float | int | str | object],
    pd.DataFrame,
    list[dict[str, float | int | str | bool | None]],
]:
    train_core = splits["train_core"]
    calibration = splits["calibration"]
    train_full = splits["train_full"]
    holdout = splits["holdout"]
    horizon = int(splits["horizon"])

    x_train_core = build_feature_matrix(train_core, feature_columns)
    x_calibration = build_feature_matrix(calibration, feature_columns)
    x_train_full = build_feature_matrix(train_full, feature_columns)
    x_holdout = build_feature_matrix(holdout, feature_columns)

    y_train_core = train_core[label_column].astype(int)
    y_calibration = calibration[label_column].astype(int)
    y_train_full = train_full[label_column].astype(int)
    y_holdout = holdout[label_column].astype(int)

    comparison_rows: list[dict[str, float | int | str | bool | None]] = []
    best_candidate: dict[str, object] | None = None
    best_calibration_metrics: dict[str, float | None] | None = None
    best_threshold = 0.5

    for candidate in build_classifier_candidates(horizon):
        candidate_name = str(candidate["name"])
        try:
            candidate_model = clone(candidate["model"])
            fit_classifier_model(
                candidate_model,
                x_train_core,
                y_train_core,
                fit_mode=str(candidate["fit_mode"]),
            )
            calibration_scores = candidate_model.predict_proba(x_calibration)[:, 1]
            threshold, calibration_metrics = tune_decision_threshold(
                y_calibration,
                calibration_scores,
            )
            selection_score = float(calibration_metrics["f1"])
            comparison_rows.append(
                {
                    "task_type": "classifier_any",
                    "horizon": horizon,
                    "candidate_model": candidate_name,
                    "selected": False,
                    "selection_score": selection_score,
                    "calibration_threshold": threshold,
                    "calibration_precision": calibration_metrics["precision"],
                    "calibration_recall": calibration_metrics["recall"],
                    "calibration_f1": calibration_metrics["f1"],
                    "calibration_roc_auc": calibration_metrics["roc_auc"],
                    "calibration_pr_auc": calibration_metrics["pr_auc"],
                    "calibration_rmse": None,
                    "calibration_mae": None,
                    "fit_error": None,
                }
            )
            if is_better_classifier_candidate(
                calibration_metrics,
                best_calibration_metrics,
            ):
                best_candidate = candidate
                best_calibration_metrics = calibration_metrics
                best_threshold = threshold
        except Exception as exc:
            comparison_rows.append(
                {
                    "task_type": "classifier_any",
                    "horizon": horizon,
                    "candidate_model": candidate_name,
                    "selected": False,
                    "selection_score": float("-inf"),
                    "calibration_threshold": None,
                    "calibration_precision": None,
                    "calibration_recall": None,
                    "calibration_f1": None,
                    "calibration_roc_auc": None,
                    "calibration_pr_auc": None,
                    "calibration_rmse": None,
                    "calibration_mae": None,
                    "fit_error": f"{type(exc).__name__}: {exc}",
                }
            )

    if best_candidate is None:
        raise ValueError(f"No classifier candidate available for {label_column}.")

    preferred_candidate = maybe_promote_classifier_candidate(
        comparison_rows,
        best_candidate,
        horizon,
    )
    if preferred_candidate is not None:
        best_candidate = preferred_candidate
        matching_rows = [
            row
            for row in comparison_rows
            if row["candidate_model"] == best_candidate["name"]
        ]
        if matching_rows and matching_rows[0]["calibration_threshold"] is not None:
            best_threshold = float(matching_rows[0]["calibration_threshold"])

    selected_model = clone(best_candidate["model"])
    fit_classifier_model(
        selected_model,
        x_train_full,
        y_train_full,
        fit_mode=str(best_candidate["fit_mode"]),
    )
    holdout_scores = selected_model.predict_proba(x_holdout)[:, 1]
    holdout_predictions = (holdout_scores >= best_threshold).astype(int)

    for row in comparison_rows:
        if row["candidate_model"] == best_candidate["name"]:
            row["selected"] = True

    metrics = {
        "selected_model": str(best_candidate["name"]),
        "decision_threshold": float(best_threshold),
        "train_rows": int(len(train_full)),
        "calibration_rows": int(len(calibration)),
        "holdout_rows": int(len(holdout)),
        "holdout_positive_rate": float(y_holdout.mean()),
        "precision": float(
            precision_score(y_holdout, holdout_predictions, zero_division=0)
        ),
        "recall": float(recall_score(y_holdout, holdout_predictions, zero_division=0)),
        "f1": float(f1_score(y_holdout, holdout_predictions, zero_division=0)),
        "roc_auc": safe_binary_metric(roc_auc_score, y_holdout, holdout_scores),
        "pr_auc": safe_binary_metric(
            average_precision_score, y_holdout, holdout_scores
        ),
        "train_start": train_full["week_start_date"].min().date().isoformat(),
        "train_end": train_full["week_start_date"].max().date().isoformat(),
        "calibration_start": calibration["week_start_date"].min().date().isoformat(),
        "calibration_end": calibration["week_start_date"].max().date().isoformat(),
        "holdout_start": holdout["week_start_date"].min().date().isoformat(),
        "holdout_end": holdout["week_start_date"].max().date().isoformat(),
        "selected_model_object": selected_model,
    }

    prediction_frame = build_prediction_frame(
        holdout,
        label_column,
        holdout_predictions,
        holdout_scores,
        str(best_candidate["name"]),
        best_threshold,
    )
    return metrics, prediction_frame, comparison_rows


def select_and_score_regressor(
    splits: dict[str, pd.DataFrame | int],
    feature_columns: list[str],
    label_column: str,
) -> tuple[
    dict[str, float | int | str | object],
    pd.DataFrame,
    list[dict[str, float | int | str | bool | None]],
]:
    train_core = splits["train_core"]
    calibration = splits["calibration"]
    train_full = splits["train_full"]
    holdout = splits["holdout"]
    horizon = int(splits["horizon"])

    x_train_core = build_feature_matrix(train_core, feature_columns)
    x_calibration = build_feature_matrix(calibration, feature_columns)
    x_train_full = build_feature_matrix(train_full, feature_columns)
    x_holdout = build_feature_matrix(holdout, feature_columns)

    y_train_core = train_core[label_column].astype(float)
    y_calibration = calibration[label_column].astype(float)
    y_train_full = train_full[label_column].astype(float)
    y_holdout = holdout[label_column].astype(float)

    comparison_rows: list[dict[str, float | int | str | bool | None]] = []
    best_candidate: dict[str, object] | None = None
    best_selection_score = -np.inf

    for candidate in build_regressor_candidates(horizon):
        candidate_name = str(candidate["name"])
        try:
            candidate_model = clone(candidate["model"])
            candidate_model.fit(x_train_core, y_train_core)
            calibration_predictions = np.clip(
                candidate_model.predict(x_calibration), 0, None
            )
            calibration_rmse = float(
                np.sqrt(mean_squared_error(y_calibration, calibration_predictions))
            )
            calibration_mae = float(
                mean_absolute_error(y_calibration, calibration_predictions)
            )
            selection_score = -calibration_rmse
            comparison_rows.append(
                {
                    "task_type": "regressor_count",
                    "horizon": horizon,
                    "candidate_model": candidate_name,
                    "selected": False,
                    "selection_score": selection_score,
                    "calibration_threshold": None,
                    "calibration_precision": None,
                    "calibration_recall": None,
                    "calibration_f1": None,
                    "calibration_roc_auc": None,
                    "calibration_pr_auc": None,
                    "calibration_rmse": calibration_rmse,
                    "calibration_mae": calibration_mae,
                    "fit_error": None,
                }
            )
            if selection_score > best_selection_score:
                best_selection_score = selection_score
                best_candidate = candidate
        except Exception as exc:
            comparison_rows.append(
                {
                    "task_type": "regressor_count",
                    "horizon": horizon,
                    "candidate_model": candidate_name,
                    "selected": False,
                    "selection_score": float("-inf"),
                    "calibration_threshold": None,
                    "calibration_precision": None,
                    "calibration_recall": None,
                    "calibration_f1": None,
                    "calibration_roc_auc": None,
                    "calibration_pr_auc": None,
                    "calibration_rmse": None,
                    "calibration_mae": None,
                    "fit_error": f"{type(exc).__name__}: {exc}",
                }
            )

    if best_candidate is None:
        raise ValueError(f"No regressor candidate available for {label_column}.")

    selected_model = clone(best_candidate["model"])
    selected_model.fit(x_train_full, y_train_full)
    holdout_predictions = np.clip(selected_model.predict(x_holdout), 0, None)

    for row in comparison_rows:
        if row["candidate_model"] == best_candidate["name"]:
            row["selected"] = True

    metrics = {
        "selected_model": str(best_candidate["name"]),
        "train_rows": int(len(train_full)),
        "calibration_rows": int(len(calibration)),
        "holdout_rows": int(len(holdout)),
        "mae": float(mean_absolute_error(y_holdout, holdout_predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y_holdout, holdout_predictions))),
        "actual_mean": float(y_holdout.mean()),
        "prediction_mean": float(np.mean(holdout_predictions)),
        "train_start": train_full["week_start_date"].min().date().isoformat(),
        "train_end": train_full["week_start_date"].max().date().isoformat(),
        "calibration_start": calibration["week_start_date"].min().date().isoformat(),
        "calibration_end": calibration["week_start_date"].max().date().isoformat(),
        "holdout_start": holdout["week_start_date"].min().date().isoformat(),
        "holdout_end": holdout["week_start_date"].max().date().isoformat(),
        "selected_model_object": selected_model,
    }

    prediction_frame = build_prediction_frame(
        holdout,
        label_column,
        holdout_predictions,
        holdout_predictions,
        str(best_candidate["name"]),
        None,
    )
    return metrics, prediction_frame, comparison_rows


def split_train_calibration_holdout(
    master: pd.DataFrame,
    label_column: str,
    horizon: int,
) -> dict[str, pd.DataFrame | int]:
    eligible_flag = f"target_{horizon}w_pre_2026_only"
    eligible_rows = master[
        master[eligible_flag].fillna(False)
        & master[label_column].notna()
        & master["havecountedlice"].fillna(False)
    ].copy()
    eligible_rows = eligible_rows.sort_values("week_start_date")

    if eligible_rows.empty:
        raise ValueError(f"No eligible rows found for {label_column}.")

    max_prediction_date = eligible_rows["week_start_date"].max()
    holdout_start = max_prediction_date - pd.Timedelta(weeks=HOLDOUT_WEEKS - 1)
    calibration_start = holdout_start - pd.Timedelta(weeks=CALIBRATION_WEEKS)

    train_core = eligible_rows[
        eligible_rows["week_start_date"] < calibration_start
    ].copy()
    calibration = eligible_rows[
        (eligible_rows["week_start_date"] >= calibration_start)
        & (eligible_rows["week_start_date"] < holdout_start)
    ].copy()
    train_full = eligible_rows[eligible_rows["week_start_date"] < holdout_start].copy()
    holdout = eligible_rows[eligible_rows["week_start_date"] >= holdout_start].copy()

    if train_core.empty or calibration.empty or train_full.empty or holdout.empty:
        raise ValueError(f"Train/calibration/holdout split failed for {label_column}.")

    return {
        "horizon": horizon,
        "train_core": train_core,
        "calibration": calibration,
        "train_full": train_full,
        "holdout": holdout,
    }


def build_classifier_candidates(horizon: int) -> list[dict[str, object]]:
    hist_default_kwargs = {
        "random_state": RANDOM_SEED,
        "max_depth": 6,
        "max_iter": 250,
        "learning_rate": 0.05,
        "min_samples_leaf": 50,
    }
    return [
        {
            "name": "hist_gb_default",
            "model": HistGradientBoostingClassifier(**hist_default_kwargs),
            "fit_mode": "standard",
        },
        {
            "name": "hist_gb_balanced",
            "model": HistGradientBoostingClassifier(**hist_default_kwargs),
            "fit_mode": "balanced_samples",
        },
        *build_xgb_classifier_candidates(horizon),
    ]


def build_regressor_candidates(horizon: int) -> list[dict[str, object]]:
    common_kwargs = {
        "random_state": RANDOM_SEED,
        "max_depth": 6,
        "max_iter": 250,
        "learning_rate": 0.05,
        "min_samples_leaf": 50,
    }
    return [
        {
            "name": "hist_gb_squared",
            "model": HistGradientBoostingRegressor(
                loss="squared_error",
                **common_kwargs,
            ),
        },
        {
            "name": "hist_gb_poisson",
            "model": HistGradientBoostingRegressor(
                loss="poisson",
                **common_kwargs,
            ),
        },
        *build_xgb_regressor_candidates(horizon),
    ]


def build_xgb_classifier_candidates(horizon: int) -> list[dict[str, object]]:
    if xgb is None:
        return []
    return [
        {
            "name": "xgb_gpu_balanced_baseline",
            "model": xgb.XGBClassifier(
                n_estimators=320,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.85,
                colsample_bytree=0.85,
                min_child_weight=5,
                reg_lambda=1.0,
                random_state=RANDOM_SEED,
                tree_method="hist",
                device="cuda",
                objective="binary:logistic",
                eval_metric="logloss",
            ),
            "fit_mode": "balanced_samples",
        },
        {
            "name": "xgb_gpu_balanced_tuned",
            "model": xgb.XGBClassifier(
                n_estimators=420,
                max_depth=5,
                learning_rate=0.04,
                subsample=0.85,
                colsample_bytree=0.85,
                min_child_weight=5,
                reg_lambda=1.0,
                gamma=0.0,
                max_delta_step=2,
                random_state=RANDOM_SEED,
                tree_method="hist",
                device="cuda",
                objective="binary:logistic",
                eval_metric="logloss",
            ),
            "fit_mode": "balanced_samples",
        },
    ]


def build_xgb_regressor_candidates(horizon: int) -> list[dict[str, object]]:
    if xgb is None:
        return []
    return [
        {
            "name": "xgb_gpu_poisson",
            "model": xgb.XGBRegressor(
                n_estimators=420,
                max_depth=5,
                learning_rate=0.04,
                subsample=0.85,
                colsample_bytree=0.85,
                min_child_weight=5,
                reg_lambda=1.0,
                random_state=RANDOM_SEED,
                tree_method="hist",
                device="cuda",
                objective="count:poisson",
                eval_metric="poisson-nloglik",
            ),
        },
        {
            "name": "xgb_gpu_tweedie",
            "model": xgb.XGBRegressor(
                n_estimators=420,
                max_depth=5,
                learning_rate=0.04,
                subsample=0.85,
                colsample_bytree=0.85,
                min_child_weight=5,
                reg_lambda=1.0,
                tweedie_variance_power=1.35,
                random_state=RANDOM_SEED,
                tree_method="hist",
                device="cuda",
                objective="reg:tweedie",
                eval_metric="rmse",
            ),
        },
    ]


def fit_classifier_model(
    model: object,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    fit_mode: str,
) -> None:
    if fit_mode == "balanced_samples":
        positive_count = max(int(y_train.sum()), 1)
        negative_count = max(int((y_train == 0).sum()), 1)
        sample_weight = np.where(
            y_train.to_numpy() == 1,
            negative_count / positive_count,
            1.0,
        )
        model.fit(x_train, y_train, sample_weight=sample_weight)
        return

    model.fit(x_train, y_train)


def is_better_classifier_candidate(
    candidate_metrics: dict[str, float | None],
    best_metrics: dict[str, float | None] | None,
    f1_tolerance: float = 0.002,
    pr_auc_tolerance: float = 0.002,
) -> bool:
    if best_metrics is None:
        return True

    candidate_f1 = float(candidate_metrics["f1"] or 0.0)
    best_f1 = float(best_metrics["f1"] or 0.0)
    if candidate_f1 > best_f1 + f1_tolerance:
        return True
    if candidate_f1 + f1_tolerance < best_f1:
        return False

    candidate_recall = float(candidate_metrics["recall"] or 0.0)
    best_recall = float(best_metrics["recall"] or 0.0)
    if candidate_recall > best_recall:
        return True
    if candidate_recall < best_recall:
        return False

    candidate_pr_auc = float(candidate_metrics["pr_auc"] or 0.0)
    best_pr_auc = float(best_metrics["pr_auc"] or 0.0)
    return candidate_pr_auc > best_pr_auc + pr_auc_tolerance


def maybe_promote_classifier_candidate(
    comparison_rows: list[dict[str, float | int | str | bool | None]],
    best_candidate: dict[str, object],
    horizon: int,
) -> dict[str, object] | None:
    preferred_by_horizon = {
        2: ("hist_gb_balanced", 0.01),
    }
    preference = preferred_by_horizon.get(horizon)
    if preference is None:
        return None

    preferred_name, max_f1_gap = preference
    best_row = next(
        row
        for row in comparison_rows
        if row["candidate_model"] == best_candidate["name"]
    )
    preferred_row = next(
        (row for row in comparison_rows if row["candidate_model"] == preferred_name),
        None,
    )
    if preferred_row is None or preferred_row["calibration_f1"] is None:
        return None

    best_f1 = float(best_row["calibration_f1"] or 0.0)
    preferred_f1 = float(preferred_row["calibration_f1"] or 0.0)
    if best_f1 - preferred_f1 > max_f1_gap:
        return None

    for candidate in build_classifier_candidates(horizon):
        if candidate["name"] == preferred_name:
            return candidate
    return None


def tune_decision_threshold(
    y_true: pd.Series,
    scores: np.ndarray,
) -> tuple[float, dict[str, float | None]]:
    best_threshold = 0.5
    best_metrics = compute_classifier_metrics(y_true, scores, threshold=0.5)

    for threshold in np.arange(0.05, 0.96, 0.01):
        current_metrics = compute_classifier_metrics(
            y_true, scores, threshold=threshold
        )
        if current_metrics["f1"] > best_metrics["f1"]:
            best_threshold = float(threshold)
            best_metrics = current_metrics

    return best_threshold, best_metrics


def compute_classifier_metrics(
    y_true: pd.Series,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, float | None]:
    predictions = (scores >= threshold).astype(int)
    return {
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": safe_binary_metric(roc_auc_score, y_true, scores),
        "pr_auc": safe_binary_metric(average_precision_score, y_true, scores),
    }


def build_prediction_frame(
    holdout: pd.DataFrame,
    label_column: str,
    predictions: np.ndarray,
    scores: np.ndarray,
    candidate_model: str,
    decision_threshold: float | None,
) -> pd.DataFrame:
    prediction_frame = holdout[
        [
            "sitenumber",
            "sitename",
            "productionareaid",
            "productionarea",
            "week_start_date",
            label_column,
        ]
    ].copy()
    prediction_frame = prediction_frame.rename(columns={label_column: "actual"})
    prediction_frame["prediction"] = predictions
    prediction_frame["score"] = scores
    prediction_frame["candidate_model"] = candidate_model
    prediction_frame["decision_threshold"] = decision_threshold
    return prediction_frame


def summarize_classifier_errors(predictions: pd.DataFrame) -> pd.DataFrame:
    classifier_predictions = predictions[
        predictions["model_type"] == "classifier_any"
    ].copy()
    if classifier_predictions.empty:
        return pd.DataFrame()

    classifier_predictions["error_type"] = np.select(
        [
            (classifier_predictions["actual"] == 1)
            & (classifier_predictions["prediction"] == 0),
            (classifier_predictions["actual"] == 0)
            & (classifier_predictions["prediction"] == 1),
        ],
        ["false_negative", "false_positive"],
        default="correct",
    )

    return (
        classifier_predictions[classifier_predictions["error_type"] != "correct"]
        .groupby(
            [
                "horizon",
                "candidate_model",
                "error_type",
                "productionareaid",
                "productionarea",
            ],
            as_index=False,
        )
        .agg(count=("actual", "size"), average_score=("score", "mean"))
        .sort_values(["horizon", "error_type", "count"], ascending=[True, True, False])
        .reset_index(drop=True)
    )


def build_feature_matrix(
    frame: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    numeric_frame = frame[feature_columns].apply(pd.to_numeric, errors="coerce")
    return numeric_frame.replace([np.inf, -np.inf], np.nan).astype(float)


def safe_binary_metric(
    metric_func,
    y_true: pd.Series,
    scores: np.ndarray,
) -> float | None:
    if y_true.nunique() < 2:
        return None
    return float(metric_func(y_true, scores))


def save_model(model: object, path) -> None:
    with open(path, "wb") as handle:
        pickle.dump(model, handle)
