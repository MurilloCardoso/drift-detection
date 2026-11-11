"""Simulação reproduzível e detecção estatística de data drift."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from drift_detect.contracts import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    validate_credit_data,
)

PREDICTION_COLUMNS = ["anomaly_score", "anomaly_label"]
PSI_THRESHOLD = 0.1
KS_ALPHA = 0.05


@dataclass(frozen=True, slots=True)
class DriftMetric:
    column: str
    column_type: str
    psi: float
    psi_drift_detected: bool
    ks_statistic: float | None = None
    ks_p_value: float | None = None
    ks_drift_detected: bool | None = None


def simulate_production_drift(reference: pd.DataFrame) -> pd.DataFrame:
    """Desloca duas variáveis numéricas mantendo o lote dentro do contrato."""

    validated, _ = validate_credit_data(reference)
    production = validated.copy()
    production["age"] = np.minimum(
        100, np.rint(production["age"].to_numpy(dtype=float) * 1.20 + 5)
    ).astype(int)
    production["credit_amount"] = np.minimum(
        2_000_000,
        np.rint(production["credit_amount"].to_numpy(dtype=float) * 1.80 + 1_000),
    ).astype(int)
    validated_production, _ = validate_credit_data(production)
    return validated_production


def score_frame(frame: pd.DataFrame, model_path: Path) -> pd.DataFrame:
    """Aplica o baseline persistido e retorna rótulo e score de anomalia."""

    if not model_path.is_file():
        raise FileNotFoundError(f"Modelo baseline não encontrado: {model_path}")
    validated, _ = validate_credit_data(frame)
    model = joblib.load(model_path)
    features = validated[FEATURE_COLUMNS]
    raw_labels = model.predict(features)
    scores = model.decision_function(features)
    return pd.DataFrame(
        {
            "customer_id": validated["customer_id"].to_numpy(),
            "anomaly_score": scores,
            "anomaly_label": np.where(raw_labels == -1, "anomaly", "inlier"),
        }
    )


def _numeric_psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    reference_values = reference.dropna().to_numpy(dtype=float)
    current_values = current.dropna().to_numpy(dtype=float)
    edges = np.unique(np.quantile(reference_values, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0
    edges = edges.astype(float)
    edges[0], edges[-1] = -np.inf, np.inf
    reference_counts = np.histogram(reference_values, bins=edges)[0]
    current_counts = np.histogram(current_values, bins=edges)[0]
    return _psi_from_counts(reference_counts, current_counts)


def _categorical_psi(reference: pd.Series, current: pd.Series) -> float:
    reference_values = reference.fillna("<missing>").astype(str)
    current_values = current.fillna("<missing>").astype(str)
    categories = sorted(set(reference_values) | set(current_values))
    reference_counts = reference_values.value_counts().reindex(categories, fill_value=0)
    current_counts = current_values.value_counts().reindex(categories, fill_value=0)
    return _psi_from_counts(reference_counts.to_numpy(), current_counts.to_numpy())


def _psi_from_counts(reference_counts: np.ndarray, current_counts: np.ndarray) -> float:
    epsilon = 1e-6
    reference_share = np.clip(reference_counts / reference_counts.sum(), epsilon, None)
    current_share = np.clip(current_counts / current_counts.sum(), epsilon, None)
    contribution = (current_share - reference_share) * np.log(
        current_share / reference_share
    )
    return float(np.sum(contribution))


def calculate_drift_metrics(
    reference: pd.DataFrame, current: pd.DataFrame
) -> list[DriftMetric]:
    """Calcula PSI e KS com limites explícitos e resultados serializáveis."""

    metrics: list[DriftMetric] = []
    for column in NUMERIC_FEATURES + ["anomaly_score"]:
        psi = _numeric_psi(reference[column], current[column])
        ks = ks_2samp(reference[column].dropna(), current[column].dropna())
        metrics.append(
            DriftMetric(
                column=column,
                column_type="numeric",
                psi=round(psi, 6),
                psi_drift_detected=psi >= PSI_THRESHOLD,
                ks_statistic=round(float(ks.statistic), 6),
                ks_p_value=round(float(ks.pvalue), 10),
                ks_drift_detected=bool(ks.pvalue < KS_ALPHA),
            )
        )
    for column in CATEGORICAL_FEATURES + ["anomaly_label"]:
        psi = _categorical_psi(reference[column], current[column])
        metrics.append(
            DriftMetric(
                column=column,
                column_type="categorical",
                psi=round(psi, 6),
                psi_drift_detected=psi >= PSI_THRESHOLD,
            )
        )
    return metrics


def _evidently_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    html_path: Path,
    json_path: Path,
) -> None:
    try:
        from evidently import DataDefinition, Dataset, Report
        from evidently.presets import DataDriftPreset
    except ImportError as error:
        raise RuntimeError(
            "Evidently não está instalado. Execute `uv sync --extra dev` antes da Etapa 2."
        ) from error

    definition = DataDefinition(
        id_column="customer_id",
        numerical_columns=NUMERIC_FEATURES + ["anomaly_score"],
        categorical_columns=CATEGORICAL_FEATURES + ["anomaly_label"],
    )
    reference_dataset = Dataset.from_pandas(reference, data_definition=definition)
    current_dataset = Dataset.from_pandas(current, data_definition=definition)
    report = Report(
        [DataDriftPreset(method="psi", threshold=PSI_THRESHOLD, drift_share=0.1)],
        include_tests=True,
    )
    snapshot = report.run(current_dataset, reference_dataset)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(html_path))
    snapshot.save_json(str(json_path))


def run_drift_analysis(
    reference: pd.DataFrame,
    model_path: Path,
    production_path: Path,
    predictions_path: Path,
    html_path: Path,
    evidently_json_path: Path,
    summary_path: Path,
) -> dict[str, object]:
    """Executa a Etapa 2 e persiste dados, predições e evidências de drift."""

    validated_reference, _ = validate_credit_data(reference)
    production = simulate_production_drift(validated_reference)
    reference_predictions = score_frame(validated_reference, model_path)
    production_predictions = score_frame(production, model_path)

    reference_analysis = validated_reference.merge(reference_predictions, on="customer_id")
    production_analysis = production.merge(production_predictions, on="customer_id")
    metrics = calculate_drift_metrics(reference_analysis, production_analysis)

    production_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    production.to_csv(production_path, index=False)
    production_predictions.to_csv(predictions_path, index=False)
    _evidently_report(
        reference_analysis,
        production_analysis,
        html_path,
        evidently_json_path,
    )

    detected = sorted(
        metric.column
        for metric in metrics
        if metric.psi_drift_detected or metric.ks_drift_detected is True
    )
    summary: dict[str, object] = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "reference_rows": len(validated_reference),
        "production_rows": len(production),
        "simulation": {
            "age": "round(min(100, age * 1.20 + 5))",
            "credit_amount": "round(min(2000000, credit_amount * 1.80 + 1000))",
        },
        "thresholds": {"psi": PSI_THRESHOLD, "ks_alpha": KS_ALPHA},
        "drifted_columns": detected,
        "reference_anomaly_rate": round(
            float((reference_predictions["anomaly_label"] == "anomaly").mean()), 6
        ),
        "production_anomaly_rate": round(
            float((production_predictions["anomaly_label"] == "anomaly").mean()), 6
        ),
        "metrics": [asdict(metric) for metric in metrics],
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary
