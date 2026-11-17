from __future__ import annotations

from pathlib import Path

import pandas as pd

from drift_detect.contracts import validate_credit_data
from drift_detect.drift import calculate_drift_metrics, score_frame, simulate_production_drift
from drift_detect.training import train_baseline


def test_simulation_preserves_labels_and_shifts_features(valid_frame: pd.DataFrame) -> None:
    production = simulate_production_drift(valid_frame)
    validate_credit_data(production)
    assert production["age"].mean() > valid_frame["age"].mean() + 8
    assert production["credit_amount"].mean() > valid_frame["credit_amount"].mean() * 1.7
    unchanged = list(set(valid_frame.columns) - {"age", "credit_amount"})
    pd.testing.assert_frame_equal(production[unchanged], valid_frame[unchanged])


def test_drift_detects_injected_features(valid_frame: pd.DataFrame, tmp_path: Path) -> None:
    model_path = tmp_path / "model.joblib"
    train_baseline(valid_frame, model_path, tmp_path / "metrics.json")
    production = simulate_production_drift(valid_frame)
    reference_scored = valid_frame.merge(score_frame(valid_frame, model_path), on="record_id")
    production_scored = production.merge(score_frame(production, model_path), on="record_id")
    metrics = {item.column: item for item in calculate_drift_metrics(
        reference_scored, production_scored)}
    assert metrics["age"].psi_drift_detected
    assert metrics["credit_amount"].psi_drift_detected
    assert not metrics["duration"].psi_drift_detected
    predictions = score_frame(production, model_path)
    assert set(predictions["predicted_risk"]) <= {"good", "bad"}
    assert predictions["bad_risk_probability"].between(0, 1).all()
