from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from drift_detect.training import train_baseline


def test_training_evaluates_holdout_and_persists_classifier(
    valid_frame: pd.DataFrame, tmp_path: Path,
) -> None:
    model_path = tmp_path / "model.joblib"
    metrics_path = tmp_path / "metrics.json"
    metrics = train_baseline(valid_frame, model_path, metrics_path, random_state=11)
    assert metrics.model_type == "LogisticRegression"
    assert (metrics.train_rows, metrics.test_rows) == (800, 200)
    assert metrics.positive_class == 2
    assert 0 <= metrics.roc_auc <= 1
    assert list(joblib.load(model_path).named_steps) == ["preprocessor", "classifier"]
    assert json.loads(metrics_path.read_text(encoding="utf-8"))["f1"] == metrics.f1
