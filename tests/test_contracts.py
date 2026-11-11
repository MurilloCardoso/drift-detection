from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from pandera.errors import SchemaErrors

from drift_detect.contracts import validate_credit_data, validate_csv
from drift_detect.data import generate_invalid_batch


def test_contract_accepts_german_credit_dataset(dataset_path: Path) -> None:
    validated, result = validate_csv(dataset_path)

    assert len(validated) == 1_000
    assert list(validated.columns) == [
        "customer_id",
        "age",
        "sex",
        "job",
        "housing",
        "saving_accounts",
        "checking_account",
        "credit_amount",
        "duration",
        "purpose",
    ]
    assert result.missing_values == 577
    assert result.duplicate_rows == 0
    assert result.contract == "german_credit_contract_v1"


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("age", 17),
        ("sex", "unknown"),
        ("job", 9),
        ("housing", "hotel"),
        ("saving_accounts", "invalid"),
        ("checking_account", "invalid"),
        ("credit_amount", -1),
        ("duration", 0),
        ("purpose", "unknown"),
    ],
)
def test_contract_rejects_invalid_values(
    valid_frame: pd.DataFrame, column: str, value: object
) -> None:
    batch = valid_frame.head(20).copy()
    batch.loc[batch.index[0], column] = value

    with pytest.raises(SchemaErrors):
        validate_credit_data(batch)


def test_contract_allows_documented_missing_accounts(valid_frame: pd.DataFrame) -> None:
    batch = valid_frame.head(20).copy()
    batch.loc[batch.index[0], ["saving_accounts", "checking_account"]] = None

    validated, _ = validate_credit_data(batch)

    assert pd.isna(validated.loc[batch.index[0], "saving_accounts"])
    assert pd.isna(validated.loc[batch.index[0], "checking_account"])


def test_contract_rejects_duplicate_customer_id(valid_frame: pd.DataFrame) -> None:
    batch = valid_frame.head(20).copy()
    batch.loc[batch.index[0], "customer_id"] = batch.loc[batch.index[1], "customer_id"]

    with pytest.raises(SchemaErrors):
        validate_credit_data(batch)


def test_invalid_demo_batch_accumulates_multiple_failures(valid_frame: pd.DataFrame) -> None:
    invalid = generate_invalid_batch(valid_frame)

    with pytest.raises(SchemaErrors) as captured:
        validate_credit_data(invalid)

    assert len(captured.value.failure_cases) >= 10
