"""Preparação do lote inválido usado para demonstrar o contrato."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from drift_detect.contracts import validate_csv


def generate_invalid_batch(reference: pd.DataFrame) -> pd.DataFrame:
    """Injeta falhas independentes para provar que o contrato bloqueia a ingestão."""

    if len(reference) < 12:
        raise ValueError("São necessárias ao menos 12 linhas para construir o lote inválido.")

    invalid = reference.head(12).copy()
    invalid.loc[0, "age"] = 17
    invalid.loc[1, "sex"] = "unknown"
    invalid.loc[2, "job"] = 9
    invalid.loc[3, "housing"] = "hotel"
    invalid.loc[4, "saving_accounts"] = "invalid"
    invalid.loc[5, "checking_account"] = "invalid"
    invalid.loc[6, "credit_amount"] = -500
    invalid.loc[7, "duration"] = 0
    invalid.loc[8, "purpose"] = "unknown"
    invalid.loc[9, "customer_id"] = invalid.loc[10, "customer_id"]
    return pd.concat([invalid, invalid.iloc[[11]]], ignore_index=True)


def write_invalid_batch(reference_path: Path, invalid_path: Path) -> None:
    """Valida o dataset de referência e grava apenas o lote de rejeição."""

    reference, _ = validate_csv(reference_path)
    invalid = generate_invalid_batch(reference)
    invalid_path.parent.mkdir(parents=True, exist_ok=True)
    invalid.to_csv(invalid_path, index=False)
