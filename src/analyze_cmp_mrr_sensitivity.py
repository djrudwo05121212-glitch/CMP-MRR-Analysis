"""Sensitivity analysis for statistically extreme CMP MRR labels.

Four training labels are isolated by a pre-defined, regime-wise 3×IQR rule.
They are not asserted to be errors.  The script reports both the official-data
result and a review-candidate-excluded result so the impact is auditable.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from analyze_cmp_mrr import (
    OUT,
    RANDOM_STATE,
    bootstrap_ci,
    make_preprocessor,
    metric_dict,
    model_frame,
    robust_sigma,
)


REGIME = ["STAGE", "MACHINE_ID", "CHAMBER"]


def flag_extreme_labels(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    flags = pd.Series(False, index=train.index)
    audit_rows = []
    for keys, group in train.groupby(REGIME, dropna=False):
        y = group["AVG_REMOVAL_RATE"]
        q1, q3 = y.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower = q1 - 3 * iqr
        upper = q3 + 3 * iqr
        if len(group) >= 20 and iqr > 0:
            local_flag = (y < lower) | (y > upper)
            flags.loc[group.index] = local_flag
        audit_rows.append(
            {
                "STAGE": keys[0], "MACHINE_ID": keys[1], "CHAMBER": keys[2], "N_RUNS": len(group),
                "Q1": q1, "Q3": q3, "IQR": iqr, "REVIEW_LOWER": lower, "REVIEW_UPPER": upper,
                "FLAGGED_COUNT": int(flags.loc[group.index].sum()),
                "RULE": "Regime-wise Q1−3×IQR or Q3+3×IQR; statistical review only",
            }
        )
    candidates = train.loc[flags, ["WAFER_ID", "STAGE", "MACHINE_ID", "CHAMBER", "SOURCE_FILE", "AVG_REMOVAL_RATE"]].copy()
    candidates["STATUS"] = "MRR 정답 검토 후보(오류 확정 아님)"
    return candidates, pd.DataFrame(audit_rows)


def regime_targets(filtered_train: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in filtered_train.groupby(REGIME, dropna=False):
        if len(group) < 8:
            continue
        y = group["AVG_REMOVAL_RATE"]
        center = float(y.median())
        sigma = robust_sigma(y)
        rows.append(
            {
                "STAGE": keys[0], "MACHINE_ID": keys[1], "CHAMBER": keys[2], "N_RUNS": len(group),
                "PROVISIONAL_TARGET": center, "ROBUST_SIGMA": sigma,
                "TEMP_LOWER": center - 3 * sigma, "TEMP_UPPER": center + 3 * sigma,
                "MIN": float(y.min()), "MAX": float(y.max()),
                "STATUS": "운전조건별 과거 중앙값(공식 Target/Spec 아님)",
            }
        )
    return pd.DataFrame(rows)


def baseline_predict(train: pd.DataFrame, target: pd.DataFrame) -> np.ndarray:
    regime_map = train.groupby(REGIME)["AVG_REMOVAL_RATE"].median()
    stage_map = train.groupby("STAGE")["AVG_REMOVAL_RATE"].median()
    values = []
    for row in target.itertuples(index=False):
        key = (row.STAGE, row.MACHINE_ID, row.CHAMBER)
        values.append(regime_map.get(key, stage_map.get(row.STAGE, train["AVG_REMOVAL_RATE"].median())))
    return np.asarray(values, dtype=float)


def main() -> None:
    started = time.time()
    train = pd.read_csv(OUT / "CMP_training_wafer_stage_features.csv")
    valid = pd.read_csv(OUT / "CMP_validation_wafer_stage_features.csv")
    test = pd.read_csv(OUT / "CMP_test_wafer_stage_features.csv")

    candidates, audit = flag_extreme_labels(train)
    candidates.to_csv(OUT / "CMP_MRR_label_review_candidates.csv", index=False, encoding="utf-8-sig")
    audit.to_csv(OUT / "CMP_MRR_label_review_rule_audit.csv", index=False, encoding="utf-8-sig")
    filtered = train.drop(index=candidates.index).reset_index(drop=True)

    targets = regime_targets(filtered)
    targets.to_csv(OUT / "CMP_provisional_MRR_targets_by_regime.csv", index=False, encoding="utf-8-sig")

    X_train, y_train, numeric, categorical = model_frame(filtered)
    X_valid, y_valid, _, _ = model_frame(valid)
    X_test, y_test, _, _ = model_frame(test)
    groups = filtered["WAFER_ID"].astype(str)
    cv = GroupKFold(n_splits=4)

    model_specs = {
        "Ridge": (Ridge(), {"model__alpha": np.logspace(-2, 3, 30)}, True),
        "Random Forest": (
            RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
            {"model__n_estimators": [300, 600], "model__max_depth": [None, 10, 18],
             "model__min_samples_leaf": [1, 3, 6], "model__max_features": ["sqrt", 0.5, 0.8]}, False),
        "Extra Trees": (
            ExtraTreesRegressor(random_state=RANDOM_STATE, n_jobs=-1),
            {"model__n_estimators": [300, 600], "model__max_depth": [None, 10, 18],
             "model__min_samples_leaf": [1, 2, 4], "model__max_features": ["sqrt", 0.5, 0.8]}, False),
        "HistGradientBoosting": (
            HistGradientBoostingRegressor(random_state=RANDOM_STATE),
            {"model__learning_rate": [0.03, 0.05, 0.1], "model__max_iter": [200, 400],
             "model__max_leaf_nodes": [15, 31, 63], "model__l2_regularization": [0.0, 1.0, 10.0]}, False),
        "XGBoost": (
            XGBRegressor(objective="reg:squarederror", random_state=RANDOM_STATE, n_jobs=1, tree_method="hist"),
            {"model__n_estimators": [300, 600], "model__max_depth": [2, 3, 4],
             "model__learning_rate": [0.03, 0.05, 0.1], "model__subsample": [0.75, 1.0],
             "model__colsample_bytree": [0.65, 0.85, 1.0], "model__min_child_weight": [1, 5, 10],
             "model__reg_lambda": [1, 10, 30]}, False),
        "CatBoost": (
            CatBoostRegressor(loss_function="RMSE", random_seed=RANDOM_STATE, verbose=False,
                              thread_count=1, allow_writing_files=False),
            {"model__iterations": [300, 600], "model__depth": [4, 6, 8],
             "model__learning_rate": [0.03, 0.05, 0.1], "model__l2_leaf_reg": [3, 10, 30]}, False),
    }

    rows, searches = [], {}
    baseline_valid = baseline_predict(filtered, valid)
    rows.append({"MODEL": "Regime Median Baseline", "CV_RMSE": np.nan, "CV_RMSE_STD": np.nan,
                 **{f"VALID_{k}": v for k, v in metric_dict(y_valid, baseline_valid).items()},
                 "BEST_PARAMS": "Stage-Machine-Chamber historical median", "SEARCH_TIME_SEC": 0.0})

    for name, (model, params, scale) in model_specs.items():
        pipeline = Pipeline([
            ("preprocess", make_preprocessor(numeric, categorical, imputer="median", scale=scale)),
            ("model", model),
        ])
        search = RandomizedSearchCV(
            pipeline, params, n_iter=4, scoring="neg_root_mean_squared_error", cv=cv,
            random_state=RANDOM_STATE, n_jobs=1, refit=True,
        )
        t0 = time.time()
        search.fit(X_train, y_train, groups=groups)
        pred = search.predict(X_valid)
        rows.append({"MODEL": name, "CV_RMSE": -search.best_score_,
                     "CV_RMSE_STD": search.cv_results_["std_test_score"][search.best_index_],
                     **{f"VALID_{k}": v for k, v in metric_dict(y_valid, pred).items()},
                     "BEST_PARAMS": json.dumps(search.best_params_, ensure_ascii=False, sort_keys=True),
                     "SEARCH_TIME_SEC": time.time() - t0})
        searches[name] = search

    comparison = pd.DataFrame(rows).sort_values(["VALID_RMSE", "VALID_MAE"])
    comparison.to_csv(OUT / "CMP_model_comparison_review_candidates_excluded.csv", index=False, encoding="utf-8-sig")
    best_name = comparison[comparison["MODEL"] != "Regime Median Baseline"].iloc[0]["MODEL"]
    best_search = searches[best_name]

    perm = permutation_importance(best_search.best_estimator_, X_valid, y_valid,
                                  scoring="neg_root_mean_squared_error", n_repeats=20,
                                  random_state=RANDOM_STATE, n_jobs=1)
    importance = pd.DataFrame({"FEATURE": X_valid.columns, "RMSE_INCREASE_MEAN": perm.importances_mean,
                               "RMSE_INCREASE_STD": perm.importances_std}).sort_values("RMSE_INCREASE_MEAN", ascending=False)
    importance["INTERPRETATION"] = "검증 데이터에서 해당 변수를 섞었을 때 RMSE 증가량; 인과관계 아님"
    importance.to_csv(OUT / "CMP_permutation_importance_final.csv", index=False, encoding="utf-8-sig")

    train_valid = pd.concat([filtered, valid], ignore_index=True)
    X_tv, y_tv, _, _ = model_frame(train_valid)
    final_model = clone(best_search.best_estimator_).fit(X_tv, y_tv)
    test_pred = final_model.predict(X_test)
    metrics = metric_dict(y_test, test_pred)
    metrics.update(bootstrap_ci(y_test.to_numpy(), test_pred))

    valid_abs = np.abs(y_valid.to_numpy() - best_search.predict(X_valid))
    half_width = float(np.quantile(valid_abs, 0.90))
    pred_df = test[["WAFER_ID", "STAGE", "SOURCE_FILE", "MACHINE_ID", "CHAMBER", "AVG_REMOVAL_RATE"]].copy()
    pred_df["PREDICTED_MRR"] = test_pred
    pred_df["RESIDUAL"] = pred_df["AVG_REMOVAL_RATE"] - pred_df["PREDICTED_MRR"]
    pred_df["ABS_ERROR"] = pred_df["RESIDUAL"].abs()
    pred_df["PREDICTION_LOWER_90"] = test_pred - half_width
    pred_df["PREDICTION_UPPER_90"] = test_pred + half_width
    coverage = float(((pred_df.AVG_REMOVAL_RATE >= pred_df.PREDICTION_LOWER_90) &
                      (pred_df.AVG_REMOVAL_RATE <= pred_df.PREDICTION_UPPER_90)).mean())
    pred_df.to_csv(OUT / "CMP_test_predictions_final.csv", index=False, encoding="utf-8-sig")

    stage_rows = []
    for stage, group in pred_df.groupby("STAGE"):
        stage_rows.append({"STAGE": stage, "N": len(group),
                           **metric_dict(group.AVG_REMOVAL_RATE, group.PREDICTED_MRR)})
    pd.DataFrame(stage_rows).to_csv(OUT / "CMP_test_metrics_by_stage_final.csv", index=False, encoding="utf-8-sig")

    # Return only observed historical conditions close to each regime target.
    candidate_runs = train_valid.merge(targets[REGIME + ["PROVISIONAL_TARGET"]], on=REGIME, how="inner")
    candidate_runs["DISTANCE_TO_TARGET"] = (candidate_runs.AVG_REMOVAL_RATE - candidate_runs.PROVISIONAL_TARGET).abs()
    candidate_runs = (candidate_runs.sort_values(REGIME + ["DISTANCE_TO_TARGET"])
                      .groupby(REGIME, group_keys=False).head(10))
    keep = REGIME + ["WAFER_ID", "SOURCE_FILE", "AVG_REMOVAL_RATE", "PROVISIONAL_TARGET",
                     "DISTANCE_TO_TARGET", "N_SAMPLES", "DURATION_SEC"] + [c for c in candidate_runs if c.endswith("__median")]
    candidate_runs[keep].to_csv(OUT / "CMP_historical_target_candidates_final.csv", index=False, encoding="utf-8-sig")

    official_comparison = pd.read_csv(OUT / "CMP_model_comparison.csv")
    official_best = official_comparison.sort_values("VALID_RMSE").iloc[0]
    summary = {
        "label_review_candidates": len(candidates),
        "rule": "Within Stage-Machine-Chamber groups with N>=20: outside Q1−3×IQR or Q3+3×IQR",
        "official_inclusive_best_model": official_best.MODEL,
        "official_inclusive_validation_RMSE": official_best.VALID_RMSE,
        "filtered_selected_model": best_name,
        "filtered_selected_parameters": best_search.best_params_,
        "filtered_validation_metrics": comparison[comparison.MODEL == best_name].iloc[0][["VALID_RMSE", "VALID_MAE", "VALID_R2", "VALID_NRMSE_MEAN"]].to_dict(),
        "final_test_metrics": metrics,
        "test_rows": len(test),
        "prediction_interval_half_width_90": half_width,
        "prediction_interval_coverage": coverage,
        "regime_targets": targets.to_dict(orient="records"),
        "runtime_sec": time.time() - started,
        "limitations": [
            "Four labels are statistical review candidates, not confirmed errors.",
            "Provisional targets and sensor bands are historical references, not engineering specifications.",
            "Permutation importance is predictive association, not causality.",
        ],
    }
    (OUT / "CMP_final_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
