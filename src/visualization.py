"""
visualization.py

Visualization utilities for Prophet forecasting.
"""

import matplotlib.pyplot as plt
import pandas as pd
from prophet import Prophet


def plot_forecast(
    model: Prophet,
    forecast: pd.DataFrame,
    figsize=(12, 6)
):
    """
    Plot the Prophet forecast.

    Parameters
    ----------
    model : Prophet
        Trained Prophet model.

    forecast : pd.DataFrame
        Forecast DataFrame returned by model.predict().

    figsize : tuple
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
        Forecast figure.
    """

    fig = model.plot(forecast)

    fig.set_size_inches(figsize)

    plt.title("Air Passenger Forecast")
    plt.xlabel("Date")
    plt.ylabel("Passengers")

    plt.grid(True)

    return fig


def plot_trend(
    forecast: pd.DataFrame,
    figsize=(12, 5)
):
    """
    Plot trend component.

    Parameters
    ----------
    forecast : pd.DataFrame
        Prophet forecast DataFrame.

    figsize : tuple
        Figure size.
    """

    plt.figure(figsize=figsize)

    plt.plot(
        forecast["ds"],
        forecast["trend"],
        linewidth=2
    )

    plt.title("Trend")
    plt.xlabel("Date")
    plt.ylabel("Trend")

    plt.grid(True)

    plt.show()


def plot_seasonality(
    model: Prophet,
    forecast: pd.DataFrame,
    figsize=(12, 8)
):
    """
    Plot Prophet seasonal components.

    Parameters
    ----------
    model : Prophet

    forecast : pd.DataFrame

    figsize : tuple
        Figure size.
    """

    fig = model.plot_components(forecast)

    fig.set_size_inches(figsize)

    return fig


def plot_actual_vs_forecast(
    actual: pd.DataFrame,
    forecast: pd.DataFrame,
    figsize=(12, 6)
):
    """
    Compare actual values with Prophet predictions.

    Parameters
    ----------
    actual : pd.DataFrame
        DataFrame containing ds and y.

    forecast : pd.DataFrame
        Forecast DataFrame.
    """

    plt.figure(figsize=figsize)

    plt.plot(
        actual["ds"],
        actual["y"],
        label="Actual",
        linewidth=2
    )

    plt.plot(
        forecast["ds"][:len(actual)],
        forecast["yhat"][:len(actual)],
        label="Predicted",
        linewidth=2
    )

    plt.title("Actual vs Forecast")

    plt.xlabel("Date")
    plt.ylabel("Passengers")

    plt.legend()

    plt.grid(True)

    plt.show()