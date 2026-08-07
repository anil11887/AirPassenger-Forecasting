import os
import joblib
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from prophet import Prophet

from src.preprocess import preprocess_data
from src.prophet_model import (
    create_model,
    train_model,
    generate_forecast
)
from src.evaluation import evaluate_model

st.set_page_config(
    page_title="Prophet Forecasting",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Time Series Forecasting with Prophet")

st.markdown("---")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_PATH = os.path.join(OUTPUT_DIR, "model.pkl")

# --- Sidebar Controls ---
st.sidebar.header("Forecast Settings")

forecast_days = st.sidebar.slider(
    "Forecast Periods",
    1,
    365,
    30
)

country = st.sidebar.selectbox(
    "Holiday Country",
    [
        "None",
        "IN",
        "US",
        "GB"
    ]
)

monthly = st.sidebar.checkbox(
    "Monthly Seasonality"
)

quarterly = st.sidebar.checkbox(
    "Quarterly Seasonality"
)

st.sidebar.header("Advanced Parameters")

changepoint = st.sidebar.slider(
    "Change Point Prior Scale",
    0.001,
    1.0,
    0.05
)

seasonality = st.sidebar.slider(
    "Seasonality Prior Scale",
    1.0,
    30.0,
    10.0
)

# --- Main Page ---
uploaded_file = st.file_uploader(
    "Upload CSV",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")

    st.dataframe(df.head())

    columns = df.columns.tolist()

    date_column = st.selectbox(
        "Choose Date Column",
        columns
    )

    target_column = st.selectbox(
        "Choose Target Column",
        columns
    )

    st.subheader("Model Options")

    mode = st.radio(
        "Choose an action",
        ["Train New Model", "Load Saved Model"]
    )

    # ----------------------------------------------------------------
    # TRAIN NEW MODEL
    # ----------------------------------------------------------------
    if mode == "Train New Model":

        if st.button("Train Model"):

            try:

                data = df[[date_column, target_column]].copy()

                data.columns = ["Month", "Passengers"]

                data = preprocess_data(data)

                selected_country = None if country == "None" else country

                model = create_model(
                    country=selected_country,
                    monthly=monthly,
                    quarterly=quarterly,
                    changepoint_prior_scale=changepoint,
                    seasonality_prior_scale=seasonality
                )

                model = train_model(model, data)

                forecast = generate_forecast(
                    model,
                    data,
                    forecast_periods=forecast_days
                )

                st.success("Model Trained Successfully!")

                # --- Save trained model ---
                os.makedirs(OUTPUT_DIR, exist_ok=True)
                joblib.dump(model, MODEL_PATH)
                st.info(f"Model saved to {MODEL_PATH}")

                st.subheader("Forecast")

                st.dataframe(
                    forecast[
                        [
                            "ds",
                            "yhat",
                            "yhat_lower",
                            "yhat_upper"
                        ]
                    ]
                )

                st.subheader("Forecast Plot")

                fig = model.plot(forecast)

                st.pyplot(fig)

                st.subheader("Confidence Interval")

                plt.figure(figsize=(12, 6))

                plt.plot(
                    forecast["ds"],
                    forecast["yhat"],
                    label="Forecast"
                )

                plt.fill_between(
                    forecast["ds"],
                    forecast["yhat_lower"],
                    forecast["yhat_upper"],
                    alpha=0.25
                )

                plt.legend()

                st.pyplot(plt)

                st.subheader("Trend & Seasonality")

                fig2 = model.plot_components(forecast)

                st.pyplot(fig2)

                historical = forecast.iloc[:len(data)]

                metrics = evaluate_model(
                    data["y"],
                    historical["yhat"]
                )

                st.subheader("Evaluation Metrics")

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "RMSE",
                    f"{metrics['RMSE']:.2f}"
                )

                col2.metric(
                    "MAE",
                    f"{metrics['MAE']:.2f}"
                )

                col3.metric(
                    "MAPE",
                    f"{metrics['MAPE']:.2f}%"
                )

                csv = forecast.to_csv(index=False)

                st.download_button(
                    "Download Forecast CSV",
                    csv,
                    file_name="forecast.csv",
                    mime="text/csv"
                )

                with open(MODEL_PATH, "rb") as f:
                    st.download_button(
                        "Download Trained Model",
                        f,
                        file_name="model.pkl",
                        mime="application/octet-stream"
                    )

            except Exception as e:

                st.error(e)

    # ----------------------------------------------------------------
    # LOAD SAVED MODEL
    # ----------------------------------------------------------------
    else:

        if st.button("Load Model"):

            try:

                if not os.path.exists(MODEL_PATH):

                    st.error(
                        f"No saved model found at {MODEL_PATH}. "
                        "Train and save one first."
                    )

                else:

                    model = joblib.load(MODEL_PATH)

                    st.success("Model loaded successfully!")

                    data = df[[date_column, target_column]].copy()

                    data.columns = ["Month", "Passengers"]

                    data = preprocess_data(data)

                    forecast = generate_forecast(
                        model,
                        data,
                        forecast_periods=forecast_days
                    )

                    st.subheader("Forecast")

                    st.dataframe(
                        forecast[
                            [
                                "ds",
                                "yhat",
                                "yhat_lower",
                                "yhat_upper"
                            ]
                        ]
                    )

                    st.subheader("Forecast Plot")

                    fig = model.plot(forecast)

                    st.pyplot(fig)

                    st.subheader("Confidence Interval")

                    plt.figure(figsize=(12, 6))

                    plt.plot(
                        forecast["ds"],
                        forecast["yhat"],
                        label="Forecast"
                    )

                    plt.fill_between(
                        forecast["ds"],
                        forecast["yhat_lower"],
                        forecast["yhat_upper"],
                        alpha=0.25
                    )

                    plt.legend()

                    st.pyplot(plt)

                    csv = forecast.to_csv(index=False)

                    st.download_button(
                        "Download Forecast CSV",
                        csv,
                        file_name="forecast.csv",
                        mime="text/csv"
                    )

            except Exception as e:

                st.error(e)
