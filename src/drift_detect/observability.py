"""Observabilidade operacional local para o pipeline de dados e drift."""

# A legibilidade do template HTML/CSS é melhor preservada sem quebrar tags e regras.
# ruff: noqa: E501

from __future__ import annotations

import html
import json
import logging
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from drift_detect.config import ProjectPaths
from drift_detect.contracts import validate_csv
from drift_detect.drift import run_drift_analysis
from drift_detect.training import train_baseline

LOGGER = logging.getLogger("drift_detect.observability")


def _utc_now() -> datetime:
    return datetime.now(UTC)


class JsonLineFormatter(logging.Formatter):
    """Serializa eventos de log em JSON Lines, um objeto por linha."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "timestamp_utc": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for field in ("run_id", "stage", "event"):
            value = getattr(record, field, None)
            if value is not None:
                payload[field] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def add_json_file_handler(log_path: Path) -> None:
    """Adiciona uma saída JSONL persistente ao logger raiz uma única vez."""

    resolved = log_path.resolve()
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        if isinstance(handler, logging.FileHandler) and Path(handler.baseFilename) == resolved:
            return
    resolved.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(resolved, encoding="utf-8")
    handler.setFormatter(JsonLineFormatter())
    root_logger.addHandler(handler)


def _append_json_line(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False) + "\n")


def load_monitoring_history(path: Path) -> list[dict[str, Any]]:
    """Lê o histórico tolerando linhas vazias ou interrompidas."""

    if not path.is_file():
        return []
    history: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            LOGGER.warning("Linha inválida ignorada no histórico de monitoramento")
            continue
        if isinstance(item, dict):
            history.append(item)
    return history


def _duration(started: float) -> float:
    return round(time.perf_counter() - started, 6)


def run_monitored_pipeline(paths: ProjectPaths) -> dict[str, object]:
    """Executa validação, prontidão do modelo e drift registrando SLIs operacionais."""

    run_id = uuid.uuid4().hex[:12]
    started_at = _utc_now()
    pipeline_started = time.perf_counter()
    stage_started = pipeline_started
    tasks: list[dict[str, object]] = []
    current_stage = "validation"
    record: dict[str, object] = {
        "run_id": run_id,
        "started_at_utc": started_at.isoformat(),
        "status": "running",
        "tasks": tasks,
    }
    LOGGER.info(
        "Execução monitorada iniciada",
        extra={"run_id": run_id, "event": "pipeline_started"},
    )

    try:
        stage_started = time.perf_counter()
        reference, validation = validate_csv(paths.reference_data)
        tasks.append(
            {
                "name": "data_validation",
                "status": "success",
                "duration_seconds": _duration(stage_started),
                "rows_processed": len(reference),
                "contract": validation.contract,
            }
        )
        LOGGER.info(
            "Contrato aprovado para %d registros",
            len(reference),
            extra={"run_id": run_id, "stage": current_stage, "event": "stage_succeeded"},
        )

        current_stage = "model_readiness"
        stage_started = time.perf_counter()
        model_action = "reused"
        if not paths.model.is_file():
            train_baseline(reference, paths.model, paths.metrics)
            model_action = "trained"
        tasks.append(
            {
                "name": current_stage,
                "status": "success",
                "duration_seconds": _duration(stage_started),
                "action": model_action,
            }
        )
        LOGGER.info(
            "Modelo pronto (%s)",
            model_action,
            extra={"run_id": run_id, "stage": current_stage, "event": "stage_succeeded"},
        )

        current_stage = "drift_analysis"
        stage_started = time.perf_counter()
        drift = run_drift_analysis(
            reference=reference,
            model_path=paths.model,
            production_path=paths.production_data,
            predictions_path=paths.production_predictions,
            html_path=paths.drift_report_html,
            evidently_json_path=paths.drift_report_json,
            summary_path=paths.drift_summary,
        )
        drifted_columns = list(drift["drifted_columns"])
        tasks.append(
            {
                "name": current_stage,
                "status": "success",
                "duration_seconds": _duration(stage_started),
                "rows_processed": drift["production_rows"],
                "drift_alert_count": len(drifted_columns),
            }
        )
        record.update(
            {
                "status": "success",
                "rows_processed": drift["production_rows"],
                "drift_alert_count": len(drifted_columns),
                "drifted_columns": drifted_columns,
                "reference_anomaly_rate": drift["reference_anomaly_rate"],
                "production_anomaly_rate": drift["production_anomaly_rate"],
                "drift_metrics": drift["metrics"],
            }
        )
        LOGGER.warning(
            "Análise concluída com %d alertas de drift: %s",
            len(drifted_columns),
            ", ".join(drifted_columns) or "nenhum",
            extra={"run_id": run_id, "stage": current_stage, "event": "drift_evaluated"},
        )
    except Exception as error:
        tasks.append(
            {
                "name": current_stage,
                "status": "failed",
                "duration_seconds": _duration(stage_started),
                "error_type": type(error).__name__,
            }
        )
        record.update(
            {
                "status": "failed",
                "failed_stage": current_stage,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "drift_alert_count": 0,
            }
        )
        LOGGER.exception(
            "Falha na etapa %s",
            current_stage,
            extra={"run_id": run_id, "stage": current_stage},
        )
        raise
    finally:
        record["finished_at_utc"] = _utc_now().isoformat()
        record["duration_seconds"] = _duration(pipeline_started)
        _append_json_line(paths.monitoring_history, record)
        render_monitoring_dashboard(
            load_monitoring_history(paths.monitoring_history), paths.monitoring_dashboard
        )
        LOGGER.info(
            "Métricas e dashboard atualizados",
            extra={"run_id": run_id, "event": "artifacts_updated"},
        )
    return record


def _format_rate(value: object) -> str:
    if isinstance(value, int | float):
        return f"{float(value) * 100:.2f}%"
    return "-"


def render_monitoring_dashboard(history: list[dict[str, Any]], output_path: Path) -> None:
    """Gera dashboard HTML autocontido a partir do histórico de execuções."""

    latest = history[-1] if history else {}
    successful = sum(item.get("status") == "success" for item in history)
    success_rate = successful / len(history) if history else 0.0
    latest_status = str(latest.get("status", "sem execuções"))
    latest_columns = latest.get("drifted_columns", [])
    if not isinstance(latest_columns, list):
        latest_columns = []

    run_rows = []
    for item in reversed(history[-20:]):
        columns = item.get("drifted_columns", [])
        columns_text = ", ".join(map(str, columns)) if isinstance(columns, list) else "-"
        run_status = html.escape(str(item.get("status", "unknown")))
        run_rows.append(
            f"<tr data-status=\"{run_status}\">"
            f"<td><code>{html.escape(str(item.get('run_id', '-')))}</code></td>"
            f"<td>{html.escape(str(item.get('started_at_utc', '-')))}</td>"
            f"<td><span class=\"status {run_status}\">"
            f"{html.escape(str(item.get('status', '-')))}</span></td>"
            f"<td>{float(item.get('duration_seconds', 0)):.3f}s</td>"
            f"<td>{html.escape(str(item.get('rows_processed', '-')))}</td>"
            f"<td>{html.escape(str(item.get('drift_alert_count', 0)))}</td>"
            f"<td>{html.escape(columns_text or '-')}</td>"
            "</tr>"
        )

    task_rows = []
    latest_tasks = latest.get("tasks", [])
    if isinstance(latest_tasks, list):
        for task in latest_tasks:
            if not isinstance(task, dict):
                continue
            task_rows.append(
                "<tr>"
                f"<td>{html.escape(str(task.get('name', '-')))}</td>"
                f"<td><span class=\"status {html.escape(str(task.get('status', 'unknown')))}\">"
                f"{html.escape(str(task.get('status', '-')))}</span></td>"
                f"<td>{float(task.get('duration_seconds', 0)):.3f}s</td>"
                f"<td>{html.escape(str(task.get('rows_processed', '-')))}</td>"
                "</tr>"
            )

    metric_rows = []
    latest_metrics = latest.get("drift_metrics", [])
    if isinstance(latest_metrics, list):
        for metric in latest_metrics:
            if not isinstance(metric, dict):
                continue
            detected = bool(
                metric.get("psi_drift_detected") or metric.get("ks_drift_detected")
            )
            metric_rows.append(
                "<tr>"
                f"<td>{html.escape(str(metric.get('column', '-')))}</td>"
                f"<td>{float(metric.get('psi', 0)):.4f}</td>"
                f"<td>{html.escape(str(metric.get('ks_p_value', '-')))}</td>"
                f"<td><span class=\"status {'alert' if detected else 'success'}\">"
                f"{'alerta' if detected else 'estável'}</span></td>"
                "</tr>"
            )

    dashboard = f"""<!doctype html>
<html lang="pt-BR"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Drift Detect - Monitoramento</title><style>
:root{{--bg:#0b1020;--panel:#151c31;--line:#2a3556;--text:#f3f6ff;--muted:#aab5d1;--green:#3ddc97;--red:#ff6b7a;--blue:#76a9ff}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--text);font:14px system-ui,sans-serif}} main{{max-width:1180px;margin:auto;padding:32px 20px 60px}}
h1{{margin:0 0 6px;font-size:30px}} h2{{margin:28px 0 12px;font-size:19px}} .subtitle{{color:var(--muted)}} .grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-top:24px}}
.card,.table-wrap{{background:var(--panel);border:1px solid var(--line);border-radius:12px}} .card{{padding:18px}} .label{{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.08em}} .value{{font-size:25px;font-weight:700;margin-top:8px}}
.table-wrap{{overflow:auto}} table{{width:100%;border-collapse:collapse}} th,td{{padding:12px 14px;text-align:left;border-bottom:1px solid var(--line);white-space:nowrap}} th{{color:var(--muted);font-size:12px;text-transform:uppercase}} tr:last-child td{{border:0}}
.status{{padding:4px 8px;border-radius:99px;background:#27304a}} .status.success{{color:var(--green);background:#14382e}} .status.failed,.status.alert{{color:var(--red);background:#421f2a}} code{{color:var(--blue)}} .note{{color:var(--muted);margin-top:28px}} @media(max-width:800px){{.grid{{grid-template-columns:repeat(2,1fr)}}}}
.toolbar{{display:flex;justify-content:flex-end;margin:-42px 0 12px}} select{{background:var(--panel);color:var(--text);border:1px solid var(--line);border-radius:8px;padding:7px 10px}}
</style></head><body><main>
<h1>Saúde do pipeline</h1><div class="subtitle">Observabilidade local das etapas de validação, modelo e detecção de drift.</div>
<section class="grid">
<div class="card"><div class="label">Último status</div><div class="value">{html.escape(latest_status)}</div></div>
<div class="card"><div class="label">Taxa de sucesso</div><div class="value">{success_rate * 100:.1f}%</div></div>
<div class="card"><div class="label">Latência total</div><div class="value">{float(latest.get('duration_seconds', 0)):.2f}s</div></div>
<div class="card"><div class="label">Alertas de drift</div><div class="value">{html.escape(str(latest.get('drift_alert_count', 0)))}</div></div>
</section>
<h2>Última execução</h2><div class="table-wrap"><table><thead><tr><th>Tarefa</th><th>Status</th><th>Latência</th><th>Volume</th></tr></thead><tbody>{''.join(task_rows) or '<tr><td colspan="4">Sem dados</td></tr>'}</tbody></table></div>
<h2>Alertas estatísticos</h2><div class="table-wrap"><table><thead><tr><th>Feature</th><th>PSI</th><th>KS p-value</th><th>Estado</th></tr></thead><tbody>{''.join(metric_rows) or '<tr><td colspan="4">Sem dados</td></tr>'}</tbody></table></div>
<h2>Histórico (20 execuções)</h2><div class="toolbar"><label>Status: <select id="statusFilter"><option value="all">todos</option><option value="success">sucesso</option><option value="failed">falha</option></select></label></div><div class="table-wrap"><table><thead><tr><th>Run ID</th><th>Início UTC</th><th>Status</th><th>Latência</th><th>Volume</th><th>Alertas</th><th>Features</th></tr></thead><tbody id="runHistory">{''.join(run_rows) or '<tr><td colspan="7">Sem execuções</td></tr>'}</tbody></table></div>
<p class="note">Taxas de anomalia: referência {_format_rate(latest.get('reference_anomaly_rate'))} · produção {_format_rate(latest.get('production_anomaly_rate'))}. Features em alerta: {html.escape(', '.join(map(str, latest_columns)) or 'nenhuma')}.</p>
</main><script>
const filter = document.getElementById('statusFilter');
filter.addEventListener('change', () => document.querySelectorAll('#runHistory tr[data-status]').forEach(row => {{
  row.hidden = filter.value !== 'all' && row.dataset.status !== filter.value;
}}));
</script></body></html>"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(dashboard, encoding="utf-8")
