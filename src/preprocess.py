"""
preprocess.py

Preprocessing utilities for Prophet forecasting.
"""

import pandas as pd


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values automatically.

    Steps:
    1. Convert target column to numeric (invalid values become NaN).
    2. Fill gaps using linear interpolation.
    3. Forward fill any remaining leading/trailing gaps.
    4. Backward fill any that are still left.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing a "Passengers" column.

    Returns
    -------
    pd.DataFrame
        DataFrame with missing values handled.
    """
    df = df.copy()

    # Convert target column to numeric
    df["Passengers"] = pd.to_numeric(
        df["Passengers"],
        errors="coerce"
    )

    # Time interpolation
    df["Passengers"] = df["Passengers"].interpolate(
        method="linear"
    )

    # Forward fill
    df["Passengers"] = df["Passengers"].ffill()

    # Backward fill
    df["Passengers"] = df["Passengers"].bfill()

    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the dataset for Prophet.

    Steps:
    1. Validate required columns.
    2. Convert date column to datetime.
    3. Handle missing values.
    4. Sort data by date.
    5. Rename columns to Prophet format (ds, y).

    Parameters
    ----------
    df : pd.DataFrame
        Raw input DataFrame.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame ready for Prophet.
    """

    # Validate required columns
    required_columns = ["Month", "Passengers"]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Create a copy
    df = df.copy()

    # Convert Month column to datetime
    df["Month"] = pd.to_datetime(df["Month"])

    # Handle missing values (numeric coercion + interpolation + fill)
    df = handle_missing_values(df)

    # Drop any rows still missing Month after datetime conversion
    df = df.dropna(subset=["Month"])

    # Sort chronologically
    df = df.sort_values(by="Month")

    # Reset index
    df.reset_index(drop=True, inplace=True)

    # Rename columns for Prophet
    df.rename(
        columns={
            "Month": "ds",
            "Passengers": "y"
        },
        inplace=True
    )

    return df