"""Stage-Chamber specific CMP analysis for the revised presentation."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from analyze_cmp_mrr import OUT, RANDOM_STATE
from analyze_cmp_mrr_sensitivity import flag_extreme_labels


REV = OUT / "revision"
REV.mkdir(exist_ok=True)
MAJOR = [("A", 1.0), ("A", 4.0), ("B", 4.0)]
PHYSICAL = [
    "PRESSURIZED_CHAMBER_PRESSURE", "MAIN_OUTER_AIR_BAG_PRESSURE",
    "CENTER_AIR_BAG_PRESSURE", "RETAINER_RING_PRESSURE", "RIPPLE_AIR_BAG_PRESSURE",
    "SLURRY_FLOW_LINE_A", "SLURRY_FLOW_LINE_B", "SLURRY_FLOW_LINE_C",
    "WAFER_ROTATION", "STAGE_ROTATION", "HEAD_ROTATION",
    "DRESSING_WATER_STATUS", "EDGE_AIR_BAG_PRESSURE",
]


def regime_name(stage: str, chamber: float) -> str:
    return f"Stage {stage}–Chamber {int(chamber)}"


def metrics(y, pred):
    return {
        "RMSE": float(mean_squared_error(y, pred) ** 0.5),
        "MAE": float(mean_absolute_error(y, pred)),
        "R2": float(r2_score(y, pred)),
    }


def tukey_stats(values: pd.Series) -> dict[str, float | int]:
    s = pd.to_numeric(values, errors="coerce").dropna()
    q1, med, q3 = [float(x) for x in s.quantile([0.25, 0.5, 0.75])]
    iqr = q3 - q1
    inside = s[(s >= q1 - 1.5 * iqr) & (s <= q3 + 1.5 * iqr)]
    return {
        "N": len(s), "Q1": q1, "MEDIAN": med, "Q3": q3, "IQR": iqr,
        "LOWER_WHISKER": float(inside.min()), "UPPER_WHISKER": float(inside.max()),
        "OUTLIER_COUNT": int(len(s) - len(inside)),
    }


def main():
    started = time.time()
    train = pd.read_csv(OUT / "CMP_training_wafer_stage_features.csv")
    valid = pd.read_csv(OUT / "CMP_validation_wafer_stage_features.csv")
    test = pd.read_csv(OUT / "CMP_test_wafer_stage_features.csv")
    flagged, _ = flag_extreme_labels(train)
    train["MRR_REVIEW_FLAG"] = train.index.isin(flagged.index)
    train = train[~train.MRR_REVIEW_FLAG].copy()

    count_rows = []
    for (stage, chamber), group in train.groupby(["STAGE", "CHAMBER"]):
        count_rows.append({
            "STAGE": stage, "CHAMBER": chamber, "N": len(group),
            "USE_IN_MAIN_ANALYSIS": (stage, chamber) in MAJOR,
            "REASON": "표본 충분" if (stage, chamber) in MAJOR else "표본 부족—참고자료로 분리",
        })
    counts = pd.DataFrame(count_rows).sort_values(["USE_IN_MAIN_ANALYSIS", "N"], ascending=[False, False])
    counts.to_csv(REV / "sample_scope.csv", index=False, encoding="utf-8-sig")

    # Tukey box-plot statistics for MRR by major Stage-Chamber condition.
    mrr_box_rows = []
    targets = []
    for stage, chamber in MAJOR:
        g = train[(train.STAGE == stage) & (train.CHAMBER == chamber)]
        stats = tukey_stats(g.AVG_REMOVAL_RATE)
        mrr_box_rows.append({"REGIME": regime_name(stage, chamber), **stats})
        targets.append({
            "REGIME": regime_name(stage, chamber), "STAGE": stage, "CHAMBER": chamber,
            "TARGET_MRR": stats["MEDIAN"], "TARGET_BAND_LOWER": stats["Q1"],
            "TARGET_BAND_UPPER": stats["Q3"], "N": stats["N"],
            "DEFINITION": "과거 중앙값을 분석용 Target으로 사용; Q1–Q3는 중심 50% 범위",
        })
    pd.DataFrame(mrr_box_rows).to_csv(REV / "mrr_boxplot_stats.csv", index=False, encoding="utf-8-sig")
    target_df = pd.DataFrame(targets)
    target_df.to_csv(REV / "target_mrr_by_regime.csv", index=False, encoding="utf-8-sig")

    # Add actual arithmetic means of each time-series sensor for Box Plot analysis.
    raw_path = OUT.parent / "cmp_integration" / "CMP_training_sensor_rows_combined.csv"
    raw = pd.read_csv(raw_path)
    mean_cols = PHYSICAL + [c for c in raw.columns if c.startswith("USAGE_OF_")]
    means = raw.groupby(["WAFER_ID", "STAGE"], as_index=False)[mean_cols].mean()
    train = train.merge(means, on=["WAFER_ID", "STAGE"], how="left", validate="one_to_one")

    excluded = {"WAFER_ID", "AVG_REMOVAL_RATE", "SOURCE_FILE", "FILE_INDEX", "START_TIMESTAMP",
                "STAGE", "CHAMBER", "MACHINE_ID", "MACHINE_DATA", "MRR_REVIEW_FLAG"}
    feature_cols = [
        c for c in train.columns
        if c not in excluded and c not in mean_cols
        and (c in {"N_SAMPLES", "DURATION_SEC"}
             or c.endswith("__median") or c.endswith("__std") or c.endswith("__range"))
    ]
    numeric_features = [c for c in feature_cols if pd.api.types.is_numeric_dtype(train[c])]

    specs = {
        "Ridge": (Ridge(), {"model__alpha": np.logspace(-2, 3, 25)}, True),
        "Random Forest": (
            RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
            {"model__n_estimators": [200, 300], "model__max_depth": [10, None],
             "model__min_samples_leaf": [1, 3, 6], "model__max_features": ["sqrt", 0.6, 0.9]}, False),
        "Extra Trees": (
            ExtraTreesRegressor(random_state=RANDOM_STATE, n_jobs=-1),
            {"model__n_estimators": [200, 300], "model__max_depth": [10, None],
             "model__min_samples_leaf": [1, 2, 4], "model__max_features": ["sqrt", 0.6, 0.9]}, False),
        "HistGradientBoosting": (
            HistGradientBoostingRegressor(random_state=RANDOM_STATE),
            {"model__learning_rate": [0.05, 0.1], "model__max_iter": [200, 300],
             "model__max_leaf_nodes": [15, 31], "model__l2_regularization": [0, 3, 10]}, False),
        "XGBoost": (
            XGBRegressor(objective="reg:squarederror", random_state=RANDOM_STATE, n_jobs=1, tree_method="hist"),
            {"model__n_estimators": [200, 300], "model__max_depth": [2, 4],
             "model__learning_rate": [0.05, 0.1], "model__subsample": [0.8, 1.0],
             "model__colsample_bytree": [0.7, 1.0], "model__reg_lambda": [1, 10]}, False),
        "CatBoost": (
            CatBoostRegressor(loss_function="RMSE", random_seed=RANDOM_STATE, verbose=False,
                              thread_count=1, allow_writing_files=False),
            {"model__iterations": [200, 300], "model__depth": [4, 6],
             "model__learning_rate": [0.05, 0.1], "model__l2_leaf_reg": [3, 10]}, False),
    }

    comparison_rows, test_rows, importance_rows, best_models = [], [], [], {}
    for stage, chamber in MAJOR:
        tr = train[(train.STAGE == stage) & (train.CHAMBER == chamber)].copy()
        va = valid[(valid.STAGE == stage) & (valid.CHAMBER == chamber)].copy()
        te = test[(test.STAGE == stage) & (test.CHAMBER == chamber)].copy()
        cv = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        searches = {}
        for name, (estimator, params, scale) in specs.items():
            num_steps = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
            if scale:
                num_steps.append(("scale", StandardScaler()))
            prep = ColumnTransformer([("numeric", Pipeline(num_steps), numeric_features)])
            pipe = Pipeline([("preprocess", prep), ("model", estimator)])
            search = RandomizedSearchCV(
                pipe, params, n_iter=2, scoring="neg_root_mean_squared_error", cv=cv,
                random_state=RANDOM_STATE, n_jobs=1, refit=True,
            )
            search.fit(tr[numeric_features], tr.AVG_REMOVAL_RATE)
            pred = search.predict(va[numeric_features])
            comparison_rows.append({
                "REGIME": regime_name(stage, chamber), "MODEL": name,
                "CV_RMSE": -search.best_score_, **{f"VALID_{k}": v for k, v in metrics(va.AVG_REMOVAL_RATE, pred).items()},
                "BEST_PARAMS": json.dumps(search.best_params_, ensure_ascii=False, sort_keys=True),
            })
            searches[name] = search

        local = pd.DataFrame([r for r in comparison_rows if r["REGIME"] == regime_name(stage, chamber)])
        best_name = local.sort_values(["VALID_RMSE", "VALID_MAE"]).iloc[0].MODEL
        selected = searches[best_name]
        best_models[(stage, chamber)] = selected

        # Interpretation is computed on validation only.
        perm = permutation_importance(
            selected.best_estimator_, va[numeric_features], va.AVG_REMOVAL_RATE,
            scoring="neg_root_mean_squared_error", n_repeats=15,
            random_state=RANDOM_STATE, n_jobs=1,
        )
        raw_imp = pd.DataFrame({"FEATURE": numeric_features, "IMPORTANCE": perm.importances_mean})
        for sensor in PHYSICAL:
            value = raw_imp.loc[raw_imp.FEATURE.str.startswith(sensor + "__"), "IMPORTANCE"].clip(lower=0).sum()
            importance_rows.append({"REGIME": regime_name(stage, chamber), "SENSOR": sensor, "IMPORTANCE": value})

        # Refit after selection and evaluate the official test partition once.
        tv = pd.concat([tr, va], ignore_index=True)
        final = clone(selected.best_estimator_).fit(tv[numeric_features], tv.AVG_REMOVAL_RATE)
        test_pred = final.predict(te[numeric_features])
        test_rows.append({
            "REGIME": regime_name(stage, chamber), "MODEL": best_name, "N_TEST": len(te),
            **metrics(te.AVG_REMOVAL_RATE, test_pred),
            "BEST_PARAMS": json.dumps(selected.best_params_, ensure_ascii=False, sort_keys=True),
        })

    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(REV / "model_comparison_by_regime.csv", index=False, encoding="utf-8-sig")
    test_metrics = pd.DataFrame(test_rows)
    test_metrics.to_csv(REV / "test_metrics_by_regime.csv", index=False, encoding="utf-8-sig")
    importance = pd.DataFrame(importance_rows)
    importance["RANK"] = importance.groupby("REGIME")["IMPORTANCE"].rank(ascending=False, method="first")
    importance.to_csv(REV / "sensor_importance_by_regime.csv", index=False, encoding="utf-8-sig")

    # Standard Box Plot values by MRR quartile for the three most predictive physical sensors.
    box_rows, candidate_rows = [], []
    for stage, chamber in MAJOR:
        regime = regime_name(stage, chamber)
        g = train[(train.STAGE == stage) & (train.CHAMBER == chamber)].copy()
        g["MRR_QUARTILE"] = pd.qcut(g.AVG_REMOVAL_RATE, 4, labels=["Q1 낮음", "Q2", "Q3", "Q4 높음"])
        top_sensors = importance[importance.REGIME == regime].nsmallest(3, "RANK").SENSOR.tolist()
        mrr_q1, target, mrr_q3 = g.AVG_REMOVAL_RATE.quantile([0.25, 0.5, 0.75])
        target_group = g[g.AVG_REMOVAL_RATE.between(mrr_q1, mrr_q3)]
        for sensor in top_sensors:
            for quartile, qg in g.groupby("MRR_QUARTILE", observed=True):
                stat = tukey_stats(qg[sensor])
                box_rows.append({"REGIME": regime, "SENSOR": sensor, "MRR_QUARTILE": str(quartile), **stat})
            sstat = tukey_stats(target_group[sensor])
            candidate_rows.append({
                "REGIME": regime, "SENSOR": sensor, "TARGET_MRR": target,
                "TARGET_MRR_LOWER_Q1": mrr_q1, "TARGET_MRR_UPPER_Q3": mrr_q3,
                "SENSOR_CANDIDATE_LOWER_Q1": sstat["Q1"],
                "SENSOR_CANDIDATE_MEDIAN": sstat["MEDIAN"],
                "SENSOR_CANDIDATE_UPPER_Q3": sstat["Q3"],
                "INTERPRETATION": "목표 MRR 중심 50%에서 관측된 Sensor 평균값의 Q1–Q3; 최적 Setpoint 확정 아님",
            })
    pd.DataFrame(box_rows).to_csv(REV / "sensor_boxplot_stats.csv", index=False, encoding="utf-8-sig")
    sensor_candidates = pd.DataFrame(candidate_rows)
    sensor_candidates.to_csv(REV / "sensor_condition_candidates.csv", index=False, encoding="utf-8-sig")

    summary = {
        "major_regimes": [regime_name(*x) for x in MAJOR],
        "excluded_sparse_samples": int(counts.loc[~counts.USE_IN_MAIN_ANALYSIS, "N"].sum()),
        "targets": target_df.to_dict(orient="records"),
        "test_metrics": test_metrics.to_dict(orient="records"),
        "sensor_candidates": sensor_candidates.to_dict(orient="records"),
        "runtime_sec": time.time() - started,
        "notes": [
            "Stage is a different processing-stage type A or B; detailed physical meaning is undisclosed.",
            "Chamber is the chamber in the machine used for wafer processing.",
            "Sensor ranges are observed associations within fixed Stage-Chamber groups, not causal recipe setpoints.",
        ],
    }
    (REV / "revision_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
