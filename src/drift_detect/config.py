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
        return self.root / "dataset" / "statlog" / "german.data"

    @property
    def invalid_batch(self) -> Path:
        return self.root / "data" / "invalid" / "statlog_invalid_batch.csv"

    @property
    def model(self) -> Path:
        return self.root / "artifacts" / "model" / "statlog_credit_classifier.joblib"

    @property
    def metrics(self) -> Path:
        return self.root / "artifacts" / "reports" / "statlog_baseline_metrics.json"

    @property
    def validation_report(self) -> Path:
        return self.root / "artifacts" / "reports" / "statlog_validation_report.json"

    @property
    def failure_cases(self) -> Path:
        return self.root / "artifacts" / "reports" / "statlog_invalid_failure_cases.csv"

    @property
    def production_data(self) -> Path:
        return self.root / "data" / "production" / "statlog_drifted.csv"

    @property
    def production_predictions(self) -> Path:
        return self.root / "artifacts" / "predictions" / "statlog_production_predictions.csv"

    @property
    def drift_report_html(self) -> Path:
        return self.root / "artifacts" / "reports" / "statlog_drift_report.html"

    @property
    def drift_report_json(self) -> Path:
        return self.root / "artifacts" / "reports" / "statlog_evidently_drift_report.json"

    @property
    def drift_summary(self) -> Path:
        return self.root / "artifacts" / "reports" / "statlog_drift_summary.json"

    @property
    def pipeline_log(self) -> Path:
        return self.root / "artifacts" / "observability" / "statlog_pipeline.jsonl"

    @property
    def monitoring_history(self) -> Path:
        return self.root / "artifacts" / "observability" / "statlog_metrics_history.jsonl"

    @property
    def monitoring_dashboard(self) -> Path:
        return self.root / "artifacts" / "reports" / "statlog_monitoring_dashboard.html"


def default_paths() -> ProjectPaths:
    """Retorna caminhos ancorados na raiz do checkout atual."""

    return ProjectPaths(root=Path.cwd().resolve())

