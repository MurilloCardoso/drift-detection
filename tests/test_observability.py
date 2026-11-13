from __future__ import annotations

import json
from pathlib import Path

from drift_detect.observability import load_monitoring_history, render_monitoring_dashboard


def test_history_ignores_incomplete_json_line(tmp_path: Path) -> None:
    history_path = tmp_path / "history.jsonl"
    history_path.write_text(
        json.dumps({"run_id": "ok", "status": "success"}) + "\n{incomplete\n",
        encoding="utf-8",
    )

    assert load_monitoring_history(history_path) == [{"run_id": "ok", "status": "success"}]


def test_dashboard_exposes_operational_and_drift_metrics(tmp_path: Path) -> None:
    output = tmp_path / "dashboard.html"
    history = [
        {
            "run_id": "abc123",
            "started_at_utc": "2026-01-01T00:00:00+00:00",
            "status": "success",
            "duration_seconds": 1.25,
            "rows_processed": 1000,
            "drift_alert_count": 1,
            "drifted_columns": ["age"],
            "reference_anomaly_rate": 0.05,
            "production_anomaly_rate": 0.12,
            "tasks": [
                {
                    "name": "data_validation",
                    "status": "success",
                    "duration_seconds": 0.1,
                    "rows_processed": 1000,
                }
            ],
            "drift_metrics": [
                {
                    "column": "age",
                    "psi": 0.42,
                    "ks_p_value": 0.001,
                    "psi_drift_detected": True,
                    "ks_drift_detected": True,
                }
            ],
        }
    ]

    render_monitoring_dashboard(history, output)
    dashboard = output.read_text(encoding="utf-8")

    assert "abc123" in dashboard
    assert "data_validation" in dashboard
    assert "age" in dashboard
    assert "1.250s" in dashboard
    assert "12.00%" in dashboard
    assert 'id="statusFilter"' in dashboard
