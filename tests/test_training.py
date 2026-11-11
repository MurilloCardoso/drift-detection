from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from drift_detect.training import train_baseline


def test_training_creates_loadable_pipeline_and_metrics(
    valid_frame: pd.DataFrame, tmp_path: Path
) -> None:
    model_path = tmp_path / "model.joblib"
    metrics_path = tmp_path / "metrics.json"

    metrics = train_baseline(valid_frame, model_path, metrics_path, random_state=11)

    assert model_path.is_file()
    assert metrics_path.is_file()
    assert metrics.model_type == "IsolationForest"
    assert metrics.contract_status == "passed"
    assert metrics.dataset_rows == 1_000
    assert 0.04 <= metrics.anomaly_rate <= 0.06
    assert len(joblib.load(model_path).named_steps) == 2
    assert json.loads(metrics_path.read_text(encoding="utf-8"))["dataset_rows"] == 1_000
