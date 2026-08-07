"""
data_loader.py

Load and validate time series datasets.
"""

from pathlib import Path
import pandas as pd


REQUIRED_COLUMNS = [
    "Month",
    "Passengers"
]


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load a CSV file after validating it.

    Parameters
    ----------
    file_path : str
        Path to CSV file.

    Returns
    -------
    pd.DataFrame
    """

    file = Path(file_path)

    if not file.exists():
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    df = pd.read_csv(file)

    if df.empty:
        raise ValueError("Dataset is empty.")

    missing_columns = [
        col
        for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    return df