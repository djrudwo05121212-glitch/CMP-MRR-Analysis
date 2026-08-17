"""CMP MRR prediction and provisional process-reference analysis.

The script deliberately separates engineering specifications from statistical
reference ranges.  Sensor and MRR limits calculated here are temporary,
data-driven review bands; they are not product specifications.
"""

from __future__ import annotations

import json
import math
import re
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, RandomizedSearchCV, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor


warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = (
    ROOT
    / "PHM-Data-Challenge-source"
    / "PHM-Data-Challenge-master"
    / "data"
    / "2016 PHM Data Challenge"
)
TRAIN_ROOT = DATA_ROOT / "2016 PHM DATA CHALLENGE CMP DATA SET"
VALID_ROOT = DATA_ROOT / "2016 PHM DATA CHALLENGE CMP VALIDATION DATA SET"
OUT = ROOT / "outputs" / "cmp_mrr_analysis_20260809"
OUT.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
KEYS = ["WAFER_ID", "STAGE"]
ID_COLS = {"SOURCE_FILE", "MACHINE_ID", "MACHINE_DATA", "TIMESTAMP", "WAFER_ID", "STAGE", "CHAMBER"}


def find_split_paths(split: str) -> tuple[Path, Path]:
    """Return the sensor directory and labelled MRR file for a split."""
    if split == "training":
        return TRAIN_ROOT / "CMP-data" / "training", TRAIN_ROOT / "CMP-training-removalrate.csv"
    if split == "validation":
        return VALID_ROOT / "validation", VALID_ROOT / "CMP-validation-removalrate.csv"
    if split == "test":
        return TRAIN_ROOT / "CMP-data" / "test", TRAIN_ROOT / "CMP-test-removalrate.csv"
    raise ValueError(split)


def safe_slope(values: np.ndarray, timestamps: np.ndarray) -> float:
    """Robustly return the least-squares slope per second for one run."""
    mask = np.isfinite(values) & np.isfinite(timestamps)
    if mask.sum() < 3:
        return np.nan
    x = timestamps[mask].astype(float)
    y = values[mask].astype(float)
    x = x - x[0]
    if np.ptp(x) == 0:
        return np.nan
    return float(np.polyfit(x, y, 1)[0])


def summarize_group(group: pd.DataFrame, source_file: str) -> dict[str, object]:
    """Convert one Wafer-Stage time series into one model row."""
    group = group.sort_values("TIMESTAMP")
    row: dict[str, object] = {
        "WAFER_ID": group["WAFER_ID"].iloc[0],
        "STAGE": str(group["STAGE"].iloc[0]),
        "SOURCE_FILE": source_file,
        "FILE_INDEX": int(re.search(r"(\d+)", source_file).group(1)),
        "MACHINE_ID": str(group["MACHINE_ID"].mode(dropna=True).iloc[0]),
        "MACHINE_DATA": str(group["MACHINE_DATA"].mode(dropna=True).iloc[0]),
        "CHAMBER": str(group["CHAMBER"].mode(dropna=True).iloc[0]),
        "N_SAMPLES": int(len(group)),
        "DURATION_SEC": float(group["TIMESTAMP"].max() - group["TIMESTAMP"].min()),
        "START_TIMESTAMP": float(group["TIMESTAMP"].min()),
    }

    numeric_cols = [c for c in group.select_dtypes(include=[np.number]).columns if c not in ID_COLS]
    ts = group["TIMESTAMP"].to_numpy(dtype=float)
    for col in numeric_cols:
        s = pd.to_numeric(group[col], errors="coerce")
        values = s.to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        prefix = col
        row[f"{prefix}__missing_rate"] = float(s.isna().mean())
        if finite.size == 0:
            for stat in ("median", "std", "p05", "p95", "range", "delta", "slope"):
                row[f"{prefix}__{stat}"] = np.nan
            continue
        row[f"{prefix}__median"] = float(np.median(finite))
        row[f"{prefix}__std"] = float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0
        row[f"{prefix}__p05"] = float(np.quantile(finite, 0.05))
        row[f"{prefix}__p95"] = float(np.quantile(finite, 0.95))
        row[f"{prefix}__range"] = float(np.max(finite) - np.min(finite))
        valid_indices = np.flatnonzero(np.isfinite(values))
        row[f"{prefix}__delta"] = float(values[valid_indices[-1]] - values[valid_indices[0]])
        row[f"{prefix}__slope"] = safe_slope(values, ts)
    return row


def build_split(split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate every file in a split and join the labelled MRR table."""
    sensor_dir, target_path = find_split_paths(split)
    feature_rows: list[dict[str, object]] = []
    raw_quality_rows: list[dict[str, object]] = []
    raw_frames: list[pd.DataFrame] = []
    for path in sorted(sensor_dir.glob("*.csv")):
        df = pd.read_csv(path)
        raw_quality_rows.append(
            {
                "SPLIT": split,
                "SOURCE_FILE": path.name,
                "RAW_ROWS": len(df),
                "RAW_COLUMNS": len(df.columns),
                "MISSING_CELLS": int(df.isna().sum().sum()),
                "MISSING_RATE": float(df.isna().sum().sum() / df.size) if df.size else np.nan,
            }
        )
        if not df.empty:
            df["SOURCE_FILE"] = path.name
            raw_frames.append(df)

    # Some Wafer-Stage runs cross source-file boundaries. Aggregate globally so
    # each target remains exactly one modelling row.
    all_rows = pd.concat(raw_frames, ignore_index=True)
    for _, group in all_rows.groupby(KEYS, sort=False, dropna=False):
        source_file = str(group["SOURCE_FILE"].mode().iloc[0])
        feature_rows.append(summarize_group(group, source_file))

    features = pd.DataFrame(feature_rows)
    targets = pd.read_csv(target_path)
    merged = features.merge(targets, on=KEYS, how="outer", indicator=True, validate="one_to_one")
    if not (merged["_merge"] == "both").all():
        raise RuntimeError(f"{split}: sensor-target key mismatch: {merged['_merge'].value_counts().to_dict()}")
    merged = merged.drop(columns="_merge")
    return merged, pd.DataFrame(raw_quality_rows)


def robust_sigma(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    if values.size == 0:
        return np.nan
    med = np.median(values)
    return float(1.4826 * np.median(np.abs(values - med)))


def make_reference_limits(train: pd.DataFrame) -> pd.DataFrame:
    """Create run-level provisional limits by Stage/Machine/Chamber.

    These are descriptive robust bands based on aggregated medians, not
    equipment setpoints or engineering specifications.
    """
    median_features = [c for c in train.columns if c.endswith("__median")]
    records: list[dict[str, object]] = []
    for keys, group in train.groupby(["STAGE", "MACHINE_ID", "CHAMBER"], dropna=False):
        for feature in median_features:
            s = pd.to_numeric(group[feature], errors="coerce").dropna()
            if len(s) < 8:
                continue
            center = float(s.median())
            sigma = robust_sigma(s)
            q005, q995 = [float(x) for x in s.quantile([0.005, 0.995])]
            if not np.isfinite(sigma) or sigma == 0:
                lower, upper, method = q005, q995, "0.5–99.5 percentile"
            else:
                lower, upper, method = center - 3 * sigma, center + 3 * sigma, "median ± 3 robust sigma"
            records.append(
                {
                    "STAGE": keys[0],
                    "MACHINE_ID": keys[1],
                    "CHAMBER": keys[2],
                    "SENSOR_SUMMARY": feature.replace("__median", ""),
                    "N_RUNS": len(s),
                    "CENTER": center,
                    "ROBUST_SIGMA": sigma,
                    "TEMP_LOWER": lower,
                    "TEMP_UPPER": upper,
                    "Q005": q005,
                    "Q995": q995,
                    "METHOD": method,
                    "STATUS": "잠정 관리범위(공식 Spec 아님)",
                }
            )
    return pd.DataFrame(records)


def make_mrr_reference(train: pd.DataFrame) -> pd.DataFrame:
    records = []
    for stage, group in train.groupby("STAGE"):
        s = group["AVG_REMOVAL_RATE"].dropna()
        center = float(s.median())
        sigma = robust_sigma(s)
        records.append(
            {
                "STAGE": stage,
                "N_RUNS": len(s),
                "PROVISIONAL_TARGET": center,
                "ROBUST_SIGMA": sigma,
                "TEMP_LOWER": center - 3 * sigma,
                "TEMP_UPPER": center + 3 * sigma,
                "MEAN": float(s.mean()),
                "STD": float(s.std()),
                "MIN": float(s.min()),
                "MAX": float(s.max()),
                "STATUS": "과거 중앙값 기반 시나리오 목표(공식 Target 아님)",
            }
        )
    return pd.DataFrame(records)


def model_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    drop_cols = {"AVG_REMOVAL_RATE", "WAFER_ID", "SOURCE_FILE", "FILE_INDEX", "START_TIMESTAMP"}
    X = df[[c for c in df.columns if c not in drop_cols]].copy()
    y = df["AVG_REMOVAL_RATE"].astype(float)
    categorical = ["STAGE", "MACHINE_ID", "MACHINE_DATA", "CHAMBER"]
    numeric = [c for c in X.columns if c not in categorical]
    return X, y, numeric, categorical


def make_preprocessor(numeric: list[str], categorical: list[str], imputer: str = "median", scale: bool = False):
    if imputer == "knn":
        numeric_steps = [("imputer", KNNImputer(n_neighbors=5, weights="distance"))]
    else:
        numeric_steps = [("imputer", SimpleImputer(strategy="median", add_indicator=True))]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))
    return ColumnTransformer(
        [
            ("numeric", Pipeline(numeric_steps), numeric),
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def metric_dict(y_true: pd.Series | np.ndarray, pred: np.ndarray) -> dict[str, float]:
    y_arr = np.asarray(y_true, dtype=float)
    pred_arr = np.asarray(pred, dtype=float)
    rmse = math.sqrt(mean_squared_error(y_arr, pred_arr))
    return {
        "RMSE": rmse,
        "MAE": mean_absolute_error(y_arr, pred_arr),
        "R2": r2_score(y_arr, pred_arr),
        "NRMSE_MEAN": rmse / abs(float(np.mean(y_arr))),
    }


def bootstrap_ci(y_true: np.ndarray, pred: np.ndarray, repeats: int = 2000) -> dict[str, float]:
    rng = np.random.default_rng(RANDOM_STATE)
    n = len(y_true)
    rmse_values, mae_values = [], []
    for _ in range(repeats):
        idx = rng.integers(0, n, n)
        rmse_values.append(math.sqrt(mean_squared_error(y_true[idx], pred[idx])))
        mae_values.append(mean_absolute_error(y_true[idx], pred[idx]))
    return {
        "RMSE_CI_LOW": float(np.quantile(rmse_values, 0.025)),
        "RMSE_CI_HIGH": float(np.quantile(rmse_values, 0.975)),
        "MAE_CI_LOW": float(np.quantile(mae_values, 0.025)),
        "MAE_CI_HIGH": float(np.quantile(mae_values, 0.975)),
    }


def main() -> None:
    started = time.time()
    split_frames: dict[str, pd.DataFrame] = {}
    quality_frames = []
    for split in ("training", "validation", "test"):
        frame, quality = build_split(split)
        frame.to_csv(OUT / f"CMP_{split}_wafer_stage_features.csv", index=False, encoding="utf-8-sig")
        split_frames[split] = frame
        quality_frames.append(quality)

    train, valid, test = split_frames["training"], split_frames["validation"], split_frames["test"]
    raw_quality = pd.concat(quality_frames, ignore_index=True)
    raw_quality.to_csv(OUT / "CMP_raw_file_quality.csv", index=False, encoding="utf-8-sig")

    # Missingness is reported at both raw-sensor and engineered-feature levels.
    feature_quality_rows = []
    for split, frame in split_frames.items():
        for col in frame.columns:
            feature_quality_rows.append(
                {
                    "SPLIT": split,
                    "COLUMN": col,
                    "MISSING_COUNT": int(frame[col].isna().sum()),
                    "MISSING_RATE": float(frame[col].isna().mean()),
                    "N_UNIQUE": int(frame[col].nunique(dropna=True)),
                }
            )
    feature_quality = pd.DataFrame(feature_quality_rows)
    feature_quality.to_csv(OUT / "CMP_feature_quality.csv", index=False, encoding="utf-8-sig")

    sensor_limits = make_reference_limits(train)
    mrr_reference = make_mrr_reference(train)
    sensor_limits.to_csv(OUT / "CMP_provisional_sensor_limits.csv", index=False, encoding="utf-8-sig")
    mrr_reference.to_csv(OUT / "CMP_provisional_MRR_targets.csv", index=False, encoding="utf-8-sig")

    X_train, y_train, numeric, categorical = model_frame(train)
    X_valid, y_valid, _, _ = model_frame(valid)
    X_test, y_test, _, _ = model_frame(test)
    groups = train["WAFER_ID"].astype(str)
    cv = GroupKFold(n_splits=4)

    # Compare two defensible imputation methods inside cross-validation only.
    prep_rows = []
    for method in ("median", "knn"):
        pipe = Pipeline(
            [
                ("preprocess", make_preprocessor(numeric, categorical, imputer=method)),
                ("model", ExtraTreesRegressor(n_estimators=300, min_samples_leaf=2, max_features=0.7, random_state=RANDOM_STATE, n_jobs=-1)),
            ]
        )
        cv_result = cross_validate(
            pipe,
            X_train,
            y_train,
            groups=groups,
            cv=cv,
            scoring={"rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error", "r2": "r2"},
            n_jobs=1,
        )
        prep_rows.append(
            {
                "IMPUTATION": "중앙값" if method == "median" else "KNN",
                "CV_RMSE_MEAN": float(-cv_result["test_rmse"].mean()),
                "CV_RMSE_STD": float(cv_result["test_rmse"].std()),
                "CV_MAE_MEAN": float(-cv_result["test_mae"].mean()),
                "CV_R2_MEAN": float(cv_result["test_r2"].mean()),
                "FIT_TIME_SEC": float(cv_result["fit_time"].sum()),
            }
        )
    prep_comparison = pd.DataFrame(prep_rows).sort_values("CV_RMSE_MEAN")
    prep_comparison.to_csv(OUT / "CMP_preprocessing_comparison.csv", index=False, encoding="utf-8-sig")
    selected_imputer = "median" if prep_comparison.iloc[0]["IMPUTATION"] == "중앙값" else "knn"

    model_specs = {
        "Ridge": (
            Ridge(),
            {"model__alpha": np.logspace(-2, 3, 30)},
            True,
        ),
        "Random Forest": (
            RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
            {
                "model__n_estimators": [300, 600],
                "model__max_depth": [None, 10, 18],
                "model__min_samples_leaf": [1, 3, 6],
                "model__max_features": ["sqrt", 0.5, 0.8],
            },
            False,
        ),
        "Extra Trees": (
            ExtraTreesRegressor(random_state=RANDOM_STATE, n_jobs=-1),
            {
                "model__n_estimators": [300, 600],
                "model__max_depth": [None, 10, 18],
                "model__min_samples_leaf": [1, 2, 4],
                "model__max_features": ["sqrt", 0.5, 0.8],
            },
            False,
        ),
        "HistGradientBoosting": (
            HistGradientBoostingRegressor(random_state=RANDOM_STATE),
            {
                "model__learning_rate": [0.03, 0.05, 0.1],
                "model__max_iter": [200, 400],
                "model__max_leaf_nodes": [15, 31, 63],
                "model__l2_regularization": [0.0, 1.0, 10.0],
            },
            False,
        ),
        "XGBoost": (
            XGBRegressor(objective="reg:squarederror", random_state=RANDOM_STATE, n_jobs=1, tree_method="hist"),
            {
                "model__n_estimators": [300, 600],
                "model__max_depth": [2, 3, 4],
                "model__learning_rate": [0.03, 0.05, 0.1],
                "model__subsample": [0.75, 1.0],
                "model__colsample_bytree": [0.65, 0.85, 1.0],
                "model__min_child_weight": [1, 5, 10],
                "model__reg_lambda": [1, 10, 30],
            },
            False,
        ),
        "CatBoost": (
            CatBoostRegressor(loss_function="RMSE", random_seed=RANDOM_STATE, verbose=False, thread_count=1, allow_writing_files=False),
            {
                "model__iterations": [300, 600],
                "model__depth": [4, 6, 8],
                "model__learning_rate": [0.03, 0.05, 0.1],
                "model__l2_leaf_reg": [3, 10, 30],
            },
            False,
        ),
    }

    model_rows, fitted_searches = [], {}
    for name, (estimator, params, needs_scale) in model_specs.items():
        pipe = Pipeline(
            [
                ("preprocess", make_preprocessor(numeric, categorical, imputer=selected_imputer, scale=needs_scale)),
                ("model", estimator),
            ]
        )
        search = RandomizedSearchCV(
            pipe,
            param_distributions=params,
            n_iter=6,
            scoring="neg_root_mean_squared_error",
            cv=cv,
            random_state=RANDOM_STATE,
            n_jobs=1,
            refit=True,
            return_train_score=True,
        )
        model_started = time.time()
        search.fit(X_train, y_train, groups=groups)
        valid_pred = search.predict(X_valid)
        row = {
            "MODEL": name,
            "CV_RMSE": float(-search.best_score_),
            "CV_RMSE_STD": float(search.cv_results_["std_test_score"][search.best_index_]),
            **{f"VALID_{k}": v for k, v in metric_dict(y_valid, valid_pred).items()},
            "BEST_PARAMS": json.dumps(search.best_params_, ensure_ascii=False, sort_keys=True),
            "SEARCH_TIME_SEC": time.time() - model_started,
        }
        model_rows.append(row)
        fitted_searches[name] = search

    model_comparison = pd.DataFrame(model_rows).sort_values(["VALID_RMSE", "VALID_MAE"])
    model_comparison.to_csv(OUT / "CMP_model_comparison.csv", index=False, encoding="utf-8-sig")
    best_name = str(model_comparison.iloc[0]["MODEL"])
    best_search = fitted_searches[best_name]

    # Interpretation uses validation, while the official test set remains untouched.
    perm = permutation_importance(
        best_search.best_estimator_, X_valid, y_valid, scoring="neg_root_mean_squared_error", n_repeats=20,
        random_state=RANDOM_STATE, n_jobs=1,
    )
    importance = pd.DataFrame(
        {"FEATURE": X_valid.columns, "RMSE_INCREASE_MEAN": perm.importances_mean, "RMSE_INCREASE_STD": perm.importances_std}
    ).sort_values("RMSE_INCREASE_MEAN", ascending=False)
    importance["INTERPRETATION"] = "값을 섞었을 때 RMSE가 증가할수록 예측에 더 중요; 인과관계는 아님"
    importance.to_csv(OUT / "CMP_permutation_importance.csv", index=False, encoding="utf-8-sig")

    # Refit the selected configuration using training + validation, then evaluate test once.
    train_valid = pd.concat([train, valid], ignore_index=True)
    X_tv, y_tv, _, _ = model_frame(train_valid)
    final_model = clone(best_search.best_estimator_)
    final_model.fit(X_tv, y_tv)
    test_pred = final_model.predict(X_test)
    test_metrics = metric_dict(y_test, test_pred)
    test_metrics.update(bootstrap_ci(y_test.to_numpy(), test_pred))

    # Validation residuals define a fixed, distribution-free empirical prediction band.
    valid_residual = np.abs(y_valid.to_numpy() - best_search.predict(X_valid))
    residual_q90 = float(np.quantile(valid_residual, 0.90))
    prediction_rows = test[["WAFER_ID", "STAGE", "SOURCE_FILE", "MACHINE_ID", "CHAMBER", "AVG_REMOVAL_RATE"]].copy()
    prediction_rows["PREDICTED_MRR"] = test_pred
    prediction_rows["RESIDUAL"] = prediction_rows["AVG_REMOVAL_RATE"] - prediction_rows["PREDICTED_MRR"]
    prediction_rows["ABS_ERROR"] = prediction_rows["RESIDUAL"].abs()
    prediction_rows["PREDICTION_LOWER_90"] = test_pred - residual_q90
    prediction_rows["PREDICTION_UPPER_90"] = test_pred + residual_q90
    coverage = float(
        ((prediction_rows["AVG_REMOVAL_RATE"] >= prediction_rows["PREDICTION_LOWER_90"]) &
         (prediction_rows["AVG_REMOVAL_RATE"] <= prediction_rows["PREDICTION_UPPER_90"])).mean()
    )
    prediction_rows.to_csv(OUT / "CMP_test_predictions.csv", index=False, encoding="utf-8-sig")

    stage_metrics = []
    for stage, group in prediction_rows.groupby("STAGE"):
        metrics = metric_dict(group["AVG_REMOVAL_RATE"], group["PREDICTED_MRR"])
        stage_metrics.append({"STAGE": stage, "N": len(group), **metrics})
    pd.DataFrame(stage_metrics).to_csv(OUT / "CMP_test_metrics_by_stage.csv", index=False, encoding="utf-8-sig")

    # Historical candidates only: observed runs closest to each provisional target.
    target_map = mrr_reference.set_index("STAGE")["PROVISIONAL_TARGET"].to_dict()
    candidate_source = train_valid.copy()
    candidate_source["PROVISIONAL_TARGET"] = candidate_source["STAGE"].map(target_map)
    candidate_source["DISTANCE_TO_TARGET"] = (candidate_source["AVG_REMOVAL_RATE"] - candidate_source["PROVISIONAL_TARGET"]).abs()
    candidates = (
        candidate_source.sort_values(["STAGE", "DISTANCE_TO_TARGET"])
        .groupby("STAGE", as_index=False, group_keys=False)
        .head(10)
    )
    candidate_cols = [
        "STAGE", "WAFER_ID", "SOURCE_FILE", "MACHINE_ID", "CHAMBER", "AVG_REMOVAL_RATE",
        "PROVISIONAL_TARGET", "DISTANCE_TO_TARGET", "N_SAMPLES", "DURATION_SEC",
    ] + [c for c in candidates.columns if c.endswith("__median")]
    candidates[candidate_cols].to_csv(OUT / "CMP_historical_target_candidates.csv", index=False, encoding="utf-8-sig")

    summary = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "split_shapes": {k: {"rows": len(v), "columns": len(v.columns)} for k, v in split_frames.items()},
        "raw_missing": {
            "cells": int(raw_quality["MISSING_CELLS"].sum()),
            "rate": float(
                raw_quality["MISSING_CELLS"].sum()
                / (raw_quality["RAW_ROWS"] * raw_quality["RAW_COLUMNS"]).sum()
            ),
        },
        "selected_imputation": selected_imputer,
        "selected_model": best_name,
        "selected_parameters": best_search.best_params_,
        "validation_metrics": model_comparison.iloc[0][["VALID_RMSE", "VALID_MAE", "VALID_R2", "VALID_NRMSE_MEAN"]].to_dict(),
        "test_metrics": test_metrics,
        "test_rows": len(test),
        "test_prediction_interval_half_width": residual_q90,
        "test_prediction_interval_coverage": coverage,
        "provisional_mrr_targets": mrr_reference.to_dict(orient="records"),
        "reference_limit_note": "All MRR and sensor limits are provisional statistical references, not engineering specifications.",
        "runtime_sec": time.time() - started,
    }
    (OUT / "CMP_analysis_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
