"""Classificador supervisionado para risco de crédito Statlog."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
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
    train_rows: int
    test_rows: int
    positive_class: int
    positive_class_meaning: str
    accuracy: float
    balanced_accuracy: float
    f1: float
    roc_auc: float
    random_state: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_pipeline(random_state: int = 42) -> Pipeline:
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler())])
    categorical = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                            ("encoder", OneHotEncoder(handle_unknown="ignore"))])
    preprocessor = ColumnTransformer([
        ("numeric", numeric, NUMERIC_FEATURES),
        ("categorical", categorical, CATEGORICAL_FEATURES),
    ])
    return Pipeline([("preprocessor", preprocessor),
                     ("classifier", LogisticRegression(max_iter=1000, random_state=random_state))])


def train_baseline(
    frame: pd.DataFrame, model_path: Path, metrics_path: Path, random_state: int = 42,
) -> TrainingMetrics:
    """Avalia em teste estratificado e persiste modelo ajustado somente no treino."""
    validated, _ = validate_credit_data(frame)
    features = validated[FEATURE_COLUMNS]
    target = (validated["credit_risk"] == 2).astype(int)
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=random_state, stratify=target,
    )
    pipeline = build_pipeline(random_state)
    pipeline.fit(x_train, y_train)
    predicted = pipeline.predict(x_test)
    probabilities = pipeline.predict_proba(x_test)[:, 1]
    metrics = TrainingMetrics(
        generated_at_utc=datetime.now(UTC).isoformat(), model_type="LogisticRegression",
        contract_status="passed", dataset_rows=len(validated), train_rows=len(x_train),
        test_rows=len(x_test), positive_class=2, positive_class_meaning="bad credit risk",
        accuracy=round(float(accuracy_score(y_test, predicted)), 6),
        balanced_accuracy=round(float(balanced_accuracy_score(y_test, predicted)), 6),
        f1=round(float(f1_score(y_test, predicted)), 6),
        roc_auc=round(float(roc_auc_score(y_test, probabilities)), 6),
        random_state=random_state,
    )
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    metrics_path.write_text(json.dumps(metrics.to_dict(), indent=2, ensure_ascii=False) + "\n",
                            encoding="utf-8")
    return metrics
