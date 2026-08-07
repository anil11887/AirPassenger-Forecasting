"""
prophet_model.py

Utilities for creating, training, and forecasting
using Facebook Prophet.
"""

from prophet import Prophet
import pandas as pd
from src.utils import detect_frequency


def create_model(
    yearly_seasonality=True,
    weekly_seasonality=False,
    daily_seasonality=False,
    country=None,
    monthly=False,
    quarterly=False,
    changepoint_prior_scale=0.05,
    seasonality_prior_scale=10.0
) -> Prophet:
    """
    Create a Prophet model.

    Parameters
    ----------
    yearly_seasonality : bool
    weekly_seasonality : bool
    daily_seasonality : bool
    country : str, optional
        ISO country code (e.g. "US", "IN") to add built-in
        country-specific holidays. None = no holidays added.
    monthly : bool
        Add a custom monthly seasonality component (period=30.5 days).
    quarterly : bool
        Add a custom quarterly seasonality component (period=91.25 days).
    changepoint_prior_scale : float
        Flexibility of the trend — higher values allow more
        abrupt trend changes (more overfitting risk).
    seasonality_prior_scale : float
        Strength of seasonality — higher values allow larger
        seasonal fluctuations

    Returns
    -------
    Prophet
        Configured Prophet model.
    """

    model = Prophet(
        yearly_seasonality=yearly_seasonality,
        weekly_seasonality=weekly_seasonality,
        daily_seasonality=daily_seasonality,
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_prior_scale=seasonality_prior_scale
    )

    if country:
        model.add_country_holidays(
            country_name=country
        )

    if monthly:
        model.add_seasonality(
            name="monthly",
            period=30.5,
            fourier_order=5
        )

    if quarterly:
        model.add_seasonality(
            name="quarterly",
            period=91.25,
            fourier_order=8
        )

    return model


def train_model(
    model: Prophet,
    df: pd.DataFrame
) -> Prophet:
    """
    Train a Prophet model.

    Parameters
    ----------
    model : Prophet
        Prophet model instance (from create_model()).
    df : pd.DataFrame
        DataFrame containing 'ds' and 'y'.

    Returns
    -------
    Prophet
        Trained Prophet model.
    """

    model.fit(df)

    return model


def generate_forecast(
    model: Prophet,
    df: pd.DataFrame,
    forecast_periods: int = 30
) -> pd.DataFrame:
    """
    Generate future forecasts, auto-detecting frequency from df.

    Parameters
    ----------
    model : Prophet
        Trained Prophet model.
    df : pd.DataFrame
        The same preprocessed DataFrame used for fitting (needs 'ds').
    forecast_periods : int
        Number of future periods to forecast.

    Returns
    -------
    pd.DataFrame
        Forecast DataFrame.
    """

    freq = detect_frequency(df)

    future = model.make_future_dataframe(
        periods=forecast_periods,
        freq=freq
    )

    forecast = model.predict(future)

    return forecast