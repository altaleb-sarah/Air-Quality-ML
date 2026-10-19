"""Time-aware evaluation for contemporaneous CO concentration estimation."""
from pathlib import Path
import argparse
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TARGET = "CO(GT)"
SENSOR_COLUMNS = [
    "PT08.S1(CO)", "C6H6(GT)", "PT08.S2(NMHC)", "NOx(GT)",
    "PT08.S3(NOx)", "NO2(GT)", "PT08.S4(NO2)", "PT08.S5(O3)",
    "T", "RH", "AH",
]


def load_data(path: str | Path) -> pd.DataFrame:
    data = pd.read_csv(path, na_values=["", "-200", -200])
    data = data.loc[:, ~data.columns.str.startswith("Unnamed")]
    if not {"Date", "Time", TARGET}.issubset(data.columns):
        raise ValueError("CSV must contain Date, Time and CO(GT) columns.")
    date = data["Date"].astype(str)
    time = data["Time"].astype(str).str.replace(".", ":", regex=False)
    data["timestamp"] = pd.to_datetime(
        date + " " + time, format="%d/%m/%Y %H:%M:%S", errors="coerce"
    )
    if data["timestamp"].isna().any():
        raise ValueError("Invalid Date/Time values found.")
    for name in [TARGET, *SENSOR_COLUMNS]:
        if name not in data:
            raise ValueError(f"Missing expected column: {name}")
        data[name] = pd.to_numeric(
            data[name].astype("string").str.replace(",", ".", regex=False),
            errors="coerce",
        ).replace(-200, np.nan)
    if data["timestamp"].duplicated().any():
        raise ValueError("Duplicate timestamps found.")
    return (
        data.sort_values("timestamp")
        .dropna(subset=[TARGET])
        .reset_index(drop=True)
    )


def build_models():
    numeric = SENSOR_COLUMNS
    preprocess = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median", keep_empty_features=True), numeric)
    ])
    ridge_preprocess = ColumnTransformer([
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
        ]), numeric)
    ])
    return {
        "Dummy (median)": Pipeline([
            ("preprocess", preprocess),
            ("model", DummyRegressor(strategy="median")),
        ]),
        "Ridge": Pipeline([
            ("preprocess", ridge_preprocess),
            ("model", Ridge(alpha=1.0)),
        ]),
        "Random Forest": Pipeline([
            ("preprocess", preprocess),
            ("model", RandomForestRegressor(
                n_estimators=150, min_samples_leaf=2, random_state=42, n_jobs=-1
            )),
        ]),
        "Histogram Gradient Boosting": Pipeline([
            ("preprocess", preprocess),
            ("model", HistGradientBoostingRegressor(random_state=42)),
        ]),
    }


def evaluate(data: pd.DataFrame, test_fraction: float = 0.2, cv_splits: int = 5):
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be strictly between 0 and 1.")
    cutoff = int(len(data) * (1 - test_fraction))
    train, test = data.iloc[:cutoff], data.iloc[cutoff:]
    if len(train) <= cv_splits or len(test) == 0:
        raise ValueError("Not enough observations for requested split and CV.")
    x_train, y_train = train[SENSOR_COLUMNS], train[TARGET]
    x_test, y_test = test[SENSOR_COLUMNS], test[TARGET]
    rows = []
    models = build_models()
    for name, pipeline in models.items():
        cv = cross_validate(
            pipeline, x_train, y_train, cv=TimeSeriesSplit(n_splits=cv_splits),
            scoring={"mae": "neg_mean_absolute_error",
                     "rmse": "neg_root_mean_squared_error"},
            n_jobs=1,
        )
        rows.append({
            "model": name,
            "cv_mae": float(-cv["test_mae"].mean()),
            "cv_rmse": float(-cv["test_rmse"].mean()),
        })
    comparison = pd.DataFrame(rows).sort_values("cv_rmse")
    best_name = comparison.iloc[0]["model"]
    best_pipeline = models[best_name]
    best_pipeline.fit(x_train, y_train)
    predictions = best_pipeline.predict(x_test)
    metrics = {
        "selected_model": best_name,
        "train_rows": len(train),
        "test_rows": len(test),
        "train_end": str(train["timestamp"].iloc[-1]),
        "test_start": str(test["timestamp"].iloc[0]),
        "test_end": str(test["timestamp"].iloc[-1]),
        "test_mae": float(mean_absolute_error(y_test, predictions)),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "test_r2": float(r2_score(y_test, predictions)),
        "note": "Contemporaneous estimation, not future-hour forecasting.",
    }
    return comparison, best_pipeline, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="AirQuality.csv")
    parser.add_argument("--output", default="artifacts")
    args = parser.parse_args()
    data = load_data(args.data)
    comparison, model, metrics = evaluate(data)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output / "cv_comparison.csv", index=False)
    joblib.dump(model, output / "best_pipeline.joblib")
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(comparison.to_string(index=False))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
