"""Treinamento do baseline não supervisionado de crédito."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from drift_detect.contracts import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    validate_credit_data,
)


@dataclass(frozen=True, slots=True)
class TrainingMetrics:
    generated_at_utc: str
    model_type: str
    contract_status: str
    dataset_rows: int
    anomaly_count: int
    anomaly_rate: float
    contamination: float
    score_min: float
    score_mean: float
    score_max: float
    random_state: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_pipeline(random_state: int = 42, contamination: float = 0.05) -> Pipeline:
    """Constrói o pré-processamento e o detector de anomalias."""

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, NUMERIC_FEATURES),
            ("categorical", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    detector = IsolationForest(
        contamination=contamination,
        n_estimators=300,
        random_state=random_state,
        n_jobs=-1,
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("detector", detector)])


def train_baseline(
    frame: pd.DataFrame,
    model_path: Path,
    metrics_path: Path,
    random_state: int = 42,
    contamination: float = 0.05,
) -> TrainingMetrics:
    """Valida, treina e persiste o baseline sem inventar uma variável-alvo."""

    validated, _ = validate_credit_data(frame)
    features = validated[FEATURE_COLUMNS]
    pipeline = build_pipeline(random_state=random_state, contamination=contamination)
    predictions = pipeline.fit_predict(features)
    scores = pipeline.decision_function(features)
    anomaly_count = int((predictions == -1).sum())

    metrics = TrainingMetrics(
        generated_at_utc=datetime.now(UTC).isoformat(),
        model_type="IsolationForest",
        contract_status="passed",
        dataset_rows=len(validated),
        anomaly_count=anomaly_count,
        anomaly_rate=round(anomaly_count / len(validated), 6),
        contamination=contamination,
        score_min=round(float(scores.min()), 6),
        score_mean=round(float(scores.mean()), 6),
        score_max=round(float(scores.max()), 6),
        random_state=random_state,
    )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    metrics_path.write_text(
        json.dumps(metrics.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return metrics
