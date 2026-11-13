"""Contrato executável para o German Credit Data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaErrors

NUMERIC_FEATURES = ["age", "credit_amount", "duration"]
CATEGORICAL_FEATURES = [
    "sex",
    "job",
    "housing",
    "saving_accounts",
    "checking_account",
    "purpose",
]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

RAW_COLUMN_MAPPING = {
    "Unnamed: 0": "customer_id",
    "Age": "age",
    "Sex": "sex",
    "Job": "job",
    "Housing": "housing",
    "Saving accounts": "saving_accounts",
    "Checking account": "checking_account",
    "Credit amount": "credit_amount",
    "Duration": "duration",
    "Purpose": "purpose",
}


def _has_no_exact_duplicates(frame: pd.DataFrame) -> bool:
    return not bool(frame.duplicated().any())


GERMAN_CREDIT_SCHEMA = pa.DataFrameSchema(
    {
        "customer_id": pa.Column(
            int,
            checks=pa.Check.ge(0),
            nullable=False,
            unique=True,
            description="Índice único do registro no dataset original.",
        ),
        "age": pa.Column(int, checks=pa.Check.in_range(18, 100), nullable=False),
        "sex": pa.Column(str, checks=pa.Check.isin(["female", "male"]), nullable=False),
        "job": pa.Column(int, checks=pa.Check.isin([0, 1, 2, 3]), nullable=False),
        "housing": pa.Column(
            str, checks=pa.Check.isin(["free", "own", "rent"]), nullable=False
        ),
        "saving_accounts": pa.Column(
            str,
            checks=pa.Check.isin(["little", "moderate", "quite rich", "rich"]),
            nullable=True,
        ),
        "checking_account": pa.Column(
            str,
            checks=pa.Check.isin(["little", "moderate", "rich"]),
            nullable=True,
        ),
        "credit_amount": pa.Column(
            int, checks=pa.Check.in_range(1, 2_000_000), nullable=False
        ),
        "duration": pa.Column(int, checks=pa.Check.in_range(1, 120), nullable=False),
        "purpose": pa.Column(
            str,
            checks=pa.Check.isin(
                [
                    "business",
                    "car",
                    "domestic appliances",
                    "education",
                    "furniture/equipment",
                    "radio/TV",
                    "repairs",
                    "vacation/others",
                ]
            ),
            nullable=False,
        ),
    },
    checks=[
        pa.Check(_has_no_exact_duplicates, error="o lote contém linhas exatamente duplicadas")
    ],
    strict=True,
    coerce=False,
    ordered=True,
    name="german_credit_contract_v1",
)


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """Resumo serializável de uma validação aprovada."""

    rows: int
    columns: int
    duplicate_rows: int
    missing_values: int
    contract: str

    def to_dict(self) -> dict[str, int | str]:
        return {
            "status": "passed",
            "contract": self.contract,
            "rows": self.rows,
            "columns": self.columns,
            "duplicate_rows": self.duplicate_rows,
            "missing_values": self.missing_values,
        }


def normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Converte os cabeçalhos do CSV original para nomes internos estáveis."""

    return frame.rename(columns=RAW_COLUMN_MAPPING).copy()


def validate_credit_data(frame: pd.DataFrame) -> tuple[pd.DataFrame, ValidationResult]:
    """Normaliza e valida um lote, acumulando os erros antes de rejeitá-lo."""

    normalized = normalize_columns(frame)
    validated = GERMAN_CREDIT_SCHEMA.validate(normalized, lazy=True)
    result = ValidationResult(
        rows=len(validated),
        columns=len(validated.columns),
        duplicate_rows=int(validated.duplicated().sum()),
        missing_values=int(validated.isna().sum().sum()),
        contract=str(GERMAN_CREDIT_SCHEMA.name),
    )
    return validated, result


def validate_csv(path: Path) -> tuple[pd.DataFrame, ValidationResult]:
    """Lê o CSV fornecido pelo usuário e aplica o contrato completo."""

    if not path.is_file():
        raise FileNotFoundError(f"Dataset não encontrado: {path}")
    frame = pd.read_csv(path)
    return validate_credit_data(frame)


def failure_cases_from(error: SchemaErrors) -> pd.DataFrame:
    """Converte falhas Pandera em evidência estável para auditoria."""

    columns = ["schema_context", "column", "check", "check_number", "failure_case", "index"]
    cases = error.failure_cases.copy()
    for column in columns:
        if column not in cases:
            cases[column] = None
    return cases[columns].sort_values(
        by=["column", "check", "index"], na_position="last", ignore_index=True
    )
