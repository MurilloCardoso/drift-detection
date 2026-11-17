"""Verifica os resultados reproduzíveis da fonte Statlog versionada."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from drift_detect.drift import calculate_drift_metrics, score_frame, simulate_production_drift
from drift_detect.training import train_baseline

ROOT = Path(__file__).parents[1]
DATA_SHA256 = "af2607959c93df2416fadd6a7e73bb21f7e7c640d810b00ac0c81c6e83dcc2bd"


def test_reference_dataset_and_results_are_reproducible(
    valid_frame: pd.DataFrame, dataset_path: Path, tmp_path: Path,
) -> None:
    assert hashlib.sha256(dataset_path.read_bytes()).hexdigest() == DATA_SHA256

    model_path = tmp_path / "model.joblib"
    actual = train_baseline(valid_frame, model_path, tmp_path / "metrics.json").to_dict()
    expected = json.loads(
        (ROOT / "artifacts/reports/statlog_baseline_metrics.json").read_text(encoding="utf-8")
    )
    for key in ("dataset_rows", "train_rows", "test_rows", "positive_class", "random_state"):
        assert actual[key] == expected[key]
    for key in ("accuracy", "balanced_accuracy", "f1", "roc_auc"):
        assert actual[key] == pytest.approx(expected[key], abs=1e-5)

    production = simulate_production_drift(valid_frame)
    reference_scored = valid_frame.merge(score_frame(valid_frame, model_path), on="record_id")
    production_scored = production.merge(score_frame(production, model_path), on="record_id")
    metrics = calculate_drift_metrics(reference_scored, production_scored)
    detected = sorted(
        metric.column for metric in metrics
        if metric.psi_drift_detected or metric.ks_drift_detected is True
    )
    expected_drift = json.loads(
        (ROOT / "artifacts/reports/statlog_drift_summary.json").read_text(encoding="utf-8")
    )
    assert len(production) == expected_drift["production_rows"]
    assert detected == expected_drift["drifted_columns"]
    for frame, key in (
        (reference_scored, "reference_bad_prediction_rate"),
        (production_scored, "production_bad_prediction_rate"),
    ):
        rate = (frame["predicted_risk"] == "bad").mean()
        assert rate == pytest.approx(expected_drift[key], abs=1e-5)
