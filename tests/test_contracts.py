from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from pandera.errors import SchemaErrors

from drift_detect.contracts import FEATURE_COLUMNS, validate_credit_data, validate_csv
from drift_detect.data import generate_invalid_batch


def test_official_zip_has_independent_contract(dataset_path: Path) -> None:
    frame, result = validate_csv(dataset_path)
    assert len(frame) == 1000
    assert len(FEATURE_COLUMNS) == 20
    assert result.columns == 22
    assert result.missing_values == 0
    assert frame["credit_risk"].value_counts().to_dict() == {1: 700, 2: 300}
    assert result.contract == "statlog_german_credit_contract_v1"


@pytest.mark.parametrize(("column", "value"), [
    ("age", 17), ("duration", 0), ("credit_amount", -1),
    ("checking_status", "invalid"), ("credit_risk", 3),
])
def test_contract_rejects_invalid_values(valid_frame: pd.DataFrame, column: str,
                                         value: object) -> None:
    batch = valid_frame.head(20).copy()
    batch.loc[batch.index[0], column] = value
    with pytest.raises(SchemaErrors):
        validate_credit_data(batch)


def test_contract_rejects_legacy_columns(valid_frame: pd.DataFrame) -> None:
    with pytest.raises(SchemaErrors):
        validate_credit_data(valid_frame.head(20).rename(columns={"record_id": "customer_id"}))


def test_invalid_batch_accumulates_failures(valid_frame: pd.DataFrame) -> None:
    with pytest.raises(SchemaErrors) as captured:
        validate_credit_data(generate_invalid_batch(valid_frame))
    assert len(captured.value.failure_cases) >= 10
