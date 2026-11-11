from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from drift_detect.contracts import validate_credit_data
from drift_detect.drift import calculate_drift_metrics, score_frame, simulate_production_drift
from drift_detect.training import train_baseline


def test_simulation_shifts_two_features_without_breaking_contract(
    valid_frame: pd.DataFrame,
) -> None:
    production = simulate_production_drift(valid_frame)

    validate_credit_data(production)
    assert len(production) == len(valid_frame)
    assert production["age"].mean() > valid_frame["age"].mean() + 8
    assert production["credit_amount"].mean() > valid_frame["credit_amount"].mean() * 1.7
    unchanged = set(valid_frame.columns) - {"age", "credit_amount"}
    pd.testing.assert_frame_equal(production[list(unchanged)], valid_frame[list(unchanged)])


def test_drift_metrics_detect_injected_features(valid_frame: pd.DataFrame, tmp_path: Path) -> None:
    model_path = tmp_path / "model.joblib"
    train_baseline(valid_frame, model_path, tmp_path / "metrics.json")
    production = simulate_production_drift(valid_frame)
    reference_scored = valid_frame.merge(score_frame(valid_frame, model_path), on="customer_id")
    production_scored = production.merge(score_frame(production, model_path), on="customer_id")

    metrics = {
        item.column: item
        for item in calculate_drift_metrics(reference_scored, production_scored)
    }

    assert metrics["age"].psi_drift_detected
    assert metrics["age"].ks_drift_detected
    assert metrics["credit_amount"].psi_drift_detected
    assert metrics["credit_amount"].ks_drift_detected
    assert not metrics["duration"].psi_drift_detected


def test_scoring_uses_persisted_pipeline(valid_frame: pd.DataFrame, tmp_path: Path) -> None:
    model_path = tmp_path / "model.joblib"
    train_baseline(valid_frame, model_path, tmp_path / "metrics.json")

    predictions = score_frame(valid_frame, model_path)

    assert len(predictions) == len(valid_frame)
    assert set(predictions["anomaly_label"]) <= {"anomaly", "inlier"}
    assert predictions["anomaly_score"].notna().all()
    assert joblib.load(model_path) is not None
