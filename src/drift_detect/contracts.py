"""Contrato independente para o Statlog German Credit oficial."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaErrors

RAW_COLUMNS = [
    "checking_status", "duration", "credit_history", "purpose", "credit_amount",
    "savings_status", "employment", "installment_rate", "personal_status_sex",
    "other_debtors", "residence_since", "property", "age", "other_installment_plans",
    "housing", "existing_credits", "job", "dependents", "telephone", "foreign_worker",
    "credit_risk",
]
NUMERIC_FEATURES = [
    "duration", "credit_amount", "installment_rate", "residence_since", "age",
    "existing_credits", "dependents",
]
CATEGORICAL_FEATURES = [column for column in RAW_COLUMNS[:-1] if column not in NUMERIC_FEATURES]
FEATURE_COLUMNS = RAW_COLUMNS[:-1]

_CODES = {
    "checking_status": ["A11", "A12", "A13", "A14"],
    "credit_history": [f"A3{i}" for i in range(5)],
    "purpose": ["A40", "A41", "A42", "A43", "A44", "A45", "A46", "A48", "A49", "A410"],
    "savings_status": [f"A6{i}" for i in range(1, 6)],
    "employment": [f"A7{i}" for i in range(1, 6)],
    "personal_status_sex": [f"A9{i}" for i in range(1, 6)],
    "other_debtors": ["A101", "A102", "A103"],
    "property": ["A121", "A122", "A123", "A124"],
    "other_installment_plans": ["A141", "A142", "A143"],
    "housing": ["A151", "A152", "A153"],
    "job": ["A171", "A172", "A173", "A174"],
    "telephone": ["A191", "A192"],
    "foreign_worker": ["A201", "A202"],
}

SCHEMA = pa.DataFrameSchema(
    {
        "record_id": pa.Column(int, checks=pa.Check.ge(0), unique=True),
        **{
            column: pa.Column(int, checks=pa.Check.in_range(*bounds))
            for column, bounds in {
                "duration": (1, 120), "credit_amount": (1, 2_000_000),
                "installment_rate": (1, 4), "residence_since": (1, 4),
                "age": (18, 100), "existing_credits": (1, 4), "dependents": (1, 2),
            }.items()
        },
        **{column: pa.Column(str, checks=pa.Check.isin(codes)) for column, codes in _CODES.items()},
        "credit_risk": pa.Column(int, checks=pa.Check.isin([1, 2])),
    },
    checks=pa.Check(lambda frame: not frame[FEATURE_COLUMNS + ["credit_risk"]].duplicated().any(),
                    error="linhas exatamente duplicadas"),
    strict=True, ordered=False, coerce=False, name="statlog_german_credit_contract_v1",
)


@dataclass(frozen=True, slots=True)
class ValidationResult:
    rows: int
    columns: int
    duplicate_rows: int
    missing_values: int
    contract: str

    def to_dict(self) -> dict[str, int | str]:
        return {"status": "passed", "contract": self.contract, "rows": self.rows,
                "columns": self.columns, "duplicate_rows": self.duplicate_rows,
                "missing_values": self.missing_values}


def validate_credit_data(frame: pd.DataFrame) -> tuple[pd.DataFrame, ValidationResult]:
    """Valida somente o formato Statlog, nunca o CSV legado."""
    validated = SCHEMA.validate(frame.copy(), lazy=True)
    return validated, ValidationResult(len(validated), len(validated.columns),
                                       int(validated.duplicated().sum()),
                                       int(validated.isna().sum().sum()), str(SCHEMA.name))


def validate_csv(path: Path) -> tuple[pd.DataFrame, ValidationResult]:
    """Lê o arquivo original german.data ou um CSV Statlog já normalizado."""
    if not path.is_file():
        raise FileNotFoundError(f"Dataset não encontrado: {path}")
    if path.suffix.lower() == ".data":
        frame = pd.read_csv(path, sep=r"\s+", header=None, names=RAW_COLUMNS)
        frame.insert(0, "record_id", range(len(frame)))
    else:
        frame = pd.read_csv(path)
    return validate_credit_data(frame)


def failure_cases_from(error: SchemaErrors) -> pd.DataFrame:
    columns = ["schema_context", "column", "check", "check_number", "failure_case", "index"]
    cases = error.failure_cases.copy()
    for column in columns:
        if column not in cases:
            cases[column] = None
    return cases[columns].sort_values(by=["column", "check", "index"],
                                     na_position="last", ignore_index=True)
