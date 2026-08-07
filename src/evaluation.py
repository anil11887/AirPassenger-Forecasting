"""
evaluation.py

Evaluation metrics for time series forecasting models.
"""

import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)


def calculate_mae(
    actual: pd.Series,
    predicted: pd.Series
) -> float:
    """
    Calculate Mean Absolute Error (MAE).

    Parameters
    ----------
    actual : pd.Series
        Actual values.

    predicted : pd.Series
        Predicted values.

    Returns
    -------
    float
        MAE score.
    """

    return mean_absolute_error(actual, predicted)


def calculate_rmse(
    actual: pd.Series,
    predicted: pd.Series
) -> float:
    """
    Calculate Root Mean Squared Error (RMSE).

    Parameters
    ----------
    actual : pd.Series
        Actual values.

    predicted : pd.Series
        Predicted values.

    Returns
    -------
    float
        RMSE score.
    """

    mse = mean_squared_error(actual, predicted)

    return np.sqrt(mse)


def calculate_mape(
    actual: pd.Series,
    predicted: pd.Series
) -> float:
    """
    Calculate Mean Absolute Percentage Error (MAPE).

    Parameters
    ----------
    actual : pd.Series
        Actual values.

    predicted : pd.Series
        Predicted values.

    Returns
    -------
    float
        MAPE percentage.
    """

    actual = np.array(actual)
    predicted = np.array(predicted)

    mask = actual != 0

    mape = np.mean(
        np.abs((actual[mask] - predicted[mask]) / actual[mask])
    ) * 100

    return mape


def evaluate_model(
    actual: pd.Series,
    predicted: pd.Series
) -> dict:
    """
    Calculate all evaluation metrics.

    Parameters
    ----------
    actual : pd.Series

    predicted : pd.Series

    Returns
    -------
    dict
        Dictionary containing MAE, RMSE, and MAPE.
    """

    return {
        "MAE": calculate_mae(actual, predicted),
        "RMSE": calculate_rmse(actual, predicted),
        "MAPE": calculate_mape(actual, predicted),
    }