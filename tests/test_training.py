"""Focused tests for data parsing, time ordering and model evaluation."""
import pandas as pd
import numpy as np
from train import SENSOR_COLUMNS, TARGET, load_data, evaluate


def test_original_csv_loads_without_improperly_dropping_missing_features():
    df = load_data("AirQuality.csv")
    assert len(df) > 7000
    assert df["timestamp"].is_monotonic_increasing
    assert df[TARGET].notna().all()
    assert df[SENSOR_COLUMNS].isna().any().any()
    assert df[SENSOR_COLUMNS].min().min() > -200


def test_evaluation_uses_future_holdout():
    # Limit work by testing model protocol on a smaller chronological sample.
    df = load_data("AirQuality.csv").iloc[:180].copy()
    results, model, metrics = evaluate(df, cv_splits=2)
    assert len(results) == 4
    assert metrics["train_end"] < metrics["test_start"]
    assert np.isfinite(metrics["test_rmse"])
    assert model is not None
