"""Configuração central e imutável dos caminhos do projeto."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ProjectPaths:
    """Caminhos utilizados pelas etapas 1 e 2, relativos à raiz do repositório."""

    root: Path

    @property
    def reference_data(self) -> Path:
        return self.root / "dataset" / "german_credit_data.csv"

    @property
    def invalid_batch(self) -> Path:
        return self.root / "data" / "invalid" / "german_credit_invalid_batch.csv"

    @property
    def model(self) -> Path:
        return self.root / "artifacts" / "model" / "german_credit_anomaly_pipeline.joblib"

    @property
    def metrics(self) -> Path:
        return self.root / "artifacts" / "reports" / "baseline_metrics.json"

    @property
    def validation_report(self) -> Path:
        return self.root / "artifacts" / "reports" / "validation_report.json"

    @property
    def failure_cases(self) -> Path:
        return self.root / "artifacts" / "reports" / "invalid_failure_cases.csv"

    @property
    def production_data(self) -> Path:
        return self.root / "data" / "production" / "german_credit_drifted.csv"

    @property
    def production_predictions(self) -> Path:
        return self.root / "artifacts" / "predictions" / "production_predictions.csv"

    @property
    def drift_report_html(self) -> Path:
        return self.root / "artifacts" / "reports" / "drift_report.html"

    @property
    def drift_report_json(self) -> Path:
        return self.root / "artifacts" / "reports" / "evidently_drift_report.json"

    @property
    def drift_summary(self) -> Path:
        return self.root / "artifacts" / "reports" / "drift_summary.json"


def default_paths() -> ProjectPaths:
    """Retorna caminhos ancorados na raiz do checkout atual."""

    return ProjectPaths(root=Path.cwd().resolve())

