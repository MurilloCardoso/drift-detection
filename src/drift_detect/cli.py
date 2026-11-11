"""Interface de linha de comando das etapas 1 e 2."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from pandera.errors import SchemaErrors

from drift_detect.config import ProjectPaths, default_paths
from drift_detect.contracts import failure_cases_from, validate_csv
from drift_detect.data import write_invalid_batch
from drift_detect.drift import run_drift_analysis
from drift_detect.training import train_baseline

LOGGER = logging.getLogger("drift_detect")


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def prepare(paths: ProjectPaths) -> None:
    write_invalid_batch(paths.reference_data, paths.invalid_batch)
    LOGGER.info("Dataset de referência preservado: %s", paths.reference_data)
    LOGGER.info("Lote propositalmente inválido criado: %s", paths.invalid_batch)


def validate(path: Path, report_path: Path | None = None) -> dict[str, object]:
    _, result = validate_csv(path)
    payload = result.to_dict()
    if report_path:
        _write_json(report_path, payload)
    LOGGER.info("Contrato aprovado: %s", json.dumps(payload, ensure_ascii=False))
    return payload


def train(paths: ProjectPaths, data_path: Path | None = None) -> dict[str, object]:
    source = data_path or paths.reference_data
    frame, _ = validate_csv(source)
    metrics = train_baseline(frame, paths.model, paths.metrics)
    LOGGER.info("Modelo salvo em %s", paths.model)
    LOGGER.info(
        "Anomalias=%d | taxa=%.2f%%",
        metrics.anomaly_count,
        metrics.anomaly_rate * 100,
    )
    return metrics.to_dict()


def demonstrate_rejection(paths: ProjectPaths) -> int:
    try:
        validate_csv(paths.invalid_batch)
    except SchemaErrors as error:
        cases = failure_cases_from(error)
        paths.failure_cases.parent.mkdir(parents=True, exist_ok=True)
        cases.to_csv(paths.failure_cases, index=False)
        LOGGER.warning("INGESTÃO BLOQUEADA: %d violações encontradas", len(cases))
        LOGGER.info("Evidências salvas em %s", paths.failure_cases)
        return len(cases)
    raise RuntimeError("O lote inválido foi aceito; a demonstração do contrato falhou.")


def run_demo(paths: ProjectPaths) -> None:
    LOGGER.info("Iniciando demonstração reproduzível da Etapa 1")
    prepare(paths)
    validation = validate(paths.reference_data, paths.validation_report)
    metrics = train(paths)
    rejected_rules = demonstrate_rejection(paths)
    LOGGER.info(
        "Etapa 1 concluída | referência=%s | anomalias=%d | violações bloqueadas=%d",
        validation["status"],
        metrics["anomaly_count"],
        rejected_rules,
    )


def run_drift(paths: ProjectPaths) -> dict[str, object]:
    """Executa a simulação, as predições e os testes da Etapa 2."""

    reference, _ = validate_csv(paths.reference_data)
    if not paths.model.is_file():
        LOGGER.info("Modelo baseline ausente; treinando antes da análise de drift")
        train_baseline(reference, paths.model, paths.metrics)
    summary = run_drift_analysis(
        reference=reference,
        model_path=paths.model,
        production_path=paths.production_data,
        predictions_path=paths.production_predictions,
        html_path=paths.drift_report_html,
        evidently_json_path=paths.drift_report_json,
        summary_path=paths.drift_summary,
    )
    LOGGER.info("Dataset de produção salvo em %s", paths.production_data)
    LOGGER.info("Relatório Evidently salvo em %s", paths.drift_report_html)
    LOGGER.info("Colunas com drift: %s", ", ".join(summary["drifted_columns"]))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="drift-detect",
        description=(
            "Validação de dados, baseline e detecção de drift — Tech Challenge Fase 4."
        ),
    )
    parser.add_argument("--verbose", action="store_true", help="Ativa logs de depuração.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "prepare", help="Valida o German Credit Data e cria o lote inválido."
    )

    validate_parser = subparsers.add_parser("validate", help="Valida um CSV pelo contrato.")
    validate_parser.add_argument("path", type=Path)

    train_parser = subparsers.add_parser("train", help="Treina o detector baseline.")
    train_parser.add_argument("--data", type=Path)

    subparsers.add_parser("demo", help="Executa a Etapa 1 completa.")
    subparsers.add_parser(
        "drift", help="Executa a simulação e a detecção de drift da Etapa 2."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _configure_logging(args.verbose)
    paths = default_paths()

    try:
        if args.command == "prepare":
            prepare(paths)
        elif args.command == "validate":
            validate(args.path)
        elif args.command == "train":
            train(paths, data_path=args.data)
        elif args.command == "demo":
            run_demo(paths)
        elif args.command == "drift":
            run_drift(paths)
        return 0
    except SchemaErrors as error:
        LOGGER.error("INGESTÃO BLOQUEADA pelo contrato: %d violações", len(error.failure_cases))
        LOGGER.error("%s", failure_cases_from(error).to_string(index=False))
        return 2
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        LOGGER.error("%s", error)
        return 1


if __name__ == "__main__":
    sys.exit(main())
