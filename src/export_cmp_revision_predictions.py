"""Refit selected regime-specific models and export independent test predictions."""

import json

import pandas as pd
from catboost import CatBoostRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from analyze_cmp_mrr import OUT, RANDOM_STATE
from analyze_cmp_mrr_sensitivity import flag_extreme_labels
from analyze_cmp_regime_revision import MAJOR, PHYSICAL, REV, regime_name


train = pd.read_csv(OUT / "CMP_training_wafer_stage_features.csv")
valid = pd.read_csv(OUT / "CMP_validation_wafer_stage_features.csv")
test = pd.read_csv(OUT / "CMP_test_wafer_stage_features.csv")
flagged, _ = flag_extreme_labels(train)
train = train.drop(index=flagged.index).copy()

excluded = {"WAFER_ID", "AVG_REMOVAL_RATE", "SOURCE_FILE", "FILE_INDEX", "START_TIMESTAMP",
            "STAGE", "CHAMBER", "MACHINE_ID", "MACHINE_DATA"}
features = [
    c for c in train.columns if c not in excluded
    and (c in {"N_SAMPLES", "DURATION_SEC"}
         or c.endswith("__median") or c.endswith("__std") or c.endswith("__range"))
]
comparison = pd.read_csv(REV / "model_comparison_by_regime.csv")
rows = []


def estimator(name):
    if name == "Random Forest":
        return RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)
    if name == "Extra Trees":
        return ExtraTreesRegressor(random_state=RANDOM_STATE, n_jobs=-1)
    if name == "HistGradientBoosting":
        return HistGradientBoostingRegressor(random_state=RANDOM_STATE)
    if name == "XGBoost":
        return XGBRegressor(objective="reg:squarederror", random_state=RANDOM_STATE, n_jobs=1, tree_method="hist")
    if name == "CatBoost":
        return CatBoostRegressor(loss_function="RMSE", random_seed=RANDOM_STATE, verbose=False,
                                 thread_count=1, allow_writing_files=False)
    return Ridge()


for stage, chamber in MAJOR:
    regime = regime_name(stage, chamber)
    local = comparison[comparison.REGIME == regime].sort_values(["VALID_RMSE", "VALID_MAE"]).iloc[0]
    name = local.MODEL
    params = {k.replace("model__", ""): v for k, v in json.loads(local.BEST_PARAMS).items()}
    model = estimator(name).set_params(**params)
    num_steps = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
    if name == "Ridge":
        num_steps.append(("scale", StandardScaler()))
    pipe = Pipeline([
        ("preprocess", ColumnTransformer([("numeric", Pipeline(num_steps), features)])),
        ("model", model),
    ])
    tv = pd.concat([
        train[(train.STAGE == stage) & (train.CHAMBER == chamber)],
        valid[(valid.STAGE == stage) & (valid.CHAMBER == chamber)],
    ], ignore_index=True)
    te = test[(test.STAGE == stage) & (test.CHAMBER == chamber)].copy()
    pipe.fit(tv[features], tv.AVG_REMOVAL_RATE)
    te["PREDICTED_MRR"] = pipe.predict(te[features])
    te["REGIME"] = regime
    rows.append(te[["REGIME", "WAFER_ID", "AVG_REMOVAL_RATE", "PREDICTED_MRR"]])

pd.concat(rows, ignore_index=True).to_csv(REV / "test_predictions_by_regime.csv", index=False, encoding="utf-8-sig")
