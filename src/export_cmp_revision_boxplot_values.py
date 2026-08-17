"""Export raw values used by native PowerPoint Box-Whisker charts."""

import pandas as pd

from analyze_cmp_mrr import OUT
from analyze_cmp_mrr_sensitivity import flag_extreme_labels
from analyze_cmp_regime_revision import MAJOR, PHYSICAL, REV, regime_name


train = pd.read_csv(OUT / "CMP_training_wafer_stage_features.csv")
flagged, _ = flag_extreme_labels(train)
train = train.drop(index=flagged.index).copy()
raw = pd.read_csv(OUT.parent / "cmp_integration" / "CMP_training_sensor_rows_combined.csv")
means = raw.groupby(["WAFER_ID", "STAGE"], as_index=False)[PHYSICAL].mean()
train = train.merge(means, on=["WAFER_ID", "STAGE"], how="left", validate="one_to_one")
importance = pd.read_csv(REV / "sensor_importance_by_regime.csv")

mrr_rows, sensor_rows = [], []
for stage, chamber in MAJOR:
    regime = regime_name(stage, chamber)
    group = train[(train.STAGE == stage) & (train.CHAMBER == chamber)].copy()
    for value in group.AVG_REMOVAL_RATE:
        mrr_rows.append({"REGIME": regime, "MRR": value})
    group["MRR_QUARTILE"] = pd.qcut(
        group.AVG_REMOVAL_RATE, 4, labels=["Q1 낮음", "Q2", "Q3", "Q4 높음"]
    )
    top = importance[importance.REGIME == regime].sort_values("RANK").head(3).SENSOR
    for sensor in top:
        for row in group[["MRR_QUARTILE", sensor]].itertuples(index=False):
            sensor_rows.append({
                "REGIME": regime, "SENSOR": sensor,
                "MRR_QUARTILE": str(row[0]), "SENSOR_MEAN": row[1],
            })

pd.DataFrame(mrr_rows).to_csv(REV / "mrr_boxplot_values.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(sensor_rows).to_csv(REV / "sensor_boxplot_values.csv", index=False, encoding="utf-8-sig")
