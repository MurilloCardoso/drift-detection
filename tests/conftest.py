from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from drift_detect.contracts import validate_csv


@pytest.fixture(scope="session")
def dataset_path() -> Path:
    return Path(__file__).parents[1] / "dataset" / "german_credit_data.csv"


@pytest.fixture(scope="session")
def valid_frame(dataset_path: Path) -> pd.DataFrame:
    frame, _ = validate_csv(dataset_path)
    return frame
