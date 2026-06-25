import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from xgboost import XGBRegressor

st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    layout="wide"
)

st.title("📈 Sales Forecasting Dashboard")

uploaded_file = st.file_uploader(
    "Upload Superstore CSV",
    type=["csv"]
)

if uploaded_file is not None:

    # -------------------------
    # LOAD DATA
    # -------------------------

    try:
        df = pd.read_csv(uploaded_file, encoding="utf-8")

    except:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, encoding="latin1")

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # -------------------------
    # DATE CONVERSION
    # -------------------------

    df["Order Date"] = pd.to_datetime(
        df["Order Date"],
        errors="coerce"
    )

    df = df.dropna(subset=["Order Date"])

    # -------------------------
    # MONTHLY SALES
    # -------------------------

    df["YearMonth"] = df["Order Date"].dt.to_period("M")

    monthly_sales = (
        df.groupby("YearMonth")["Sales"]
        .sum()
        .reset_index()
    )

    monthly_sales["Order Date"] = (
        monthly_sales["YearMonth"]
        .dt.to_timestamp()
    )

    # -------------------------
    # MONTHLY SALES TREND
    # -------------------------

    st.subheader("Monthly Sales Trend")

    fig, ax = plt.subplots(figsize=(12,5))

    ax.plot(
        monthly_sales["Order Date"],
        monthly_sales["Sales"],
        marker="o"
    )

    ax.set_title("Monthly Sales Trend")

    st.pyplot(fig)

    # -------------------------
    # FEATURE ENGINEERING
    # -------------------------

    monthly_sales["month"] = (
        monthly_sales["Order Date"].dt.month
    )

    monthly_sales["quarter"] = (
        monthly_sales["Order Date"].dt.quarter
    )

    monthly_sales["year"] = (
        monthly_sales["Order Date"].dt.year
    )

    monthly_sales["month_sin"] = np.sin(
        2*np.pi*monthly_sales["month"]/12
    )

    monthly_sales["month_cos"] = np.cos(
        2*np.pi*monthly_sales["month"]/12
    )

    monthly_sales["lag1"] = (
        monthly_sales["Sales"].shift(1)
    )

    monthly_sales["lag3"] = (
        monthly_sales["Sales"].shift(3)
    )

    monthly_sales["lag6"] = (
        monthly_sales["Sales"].shift(6)
    )

    monthly_sales["rolling3"] = (
        monthly_sales["Sales"]
        .rolling(3)
        .mean()
    )

    monthly_sales["rolling6"] = (
        monthly_sales["Sales"]
        .rolling(6)
        .mean()
    )

    monthly_sales.dropna(inplace=True)

    FEATURES = [
        "month",
        "quarter",
        "year",
        "month_sin",
        "month_cos",
        "lag1",
        "lag3",
        "lag6",
        "rolling3",
        "rolling6"
    ]

    X = monthly_sales[FEATURES]
    y = monthly_sales["Sales"]

    split = int(len(monthly_sales) * 0.8)

    X_train = X[:split]
    X_test = X[split:]

    y_train = y[:split]
    y_test = y[split:]

    # -------------------------
    # MODEL
    # -------------------------

    model = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    # -------------------------
    # METRICS
    # -------------------------

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    st.subheader("Model Performance")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "MAE",
        f"{mae:,.2f}"
    )

    col2.metric(
        "RMSE",
        f"{rmse:,.2f}"
    )

    col3.metric(
        "R²",
        f"{r2:.4f}"
    )

    # -------------------------
    # ACTUAL VS PREDICTED
    # -------------------------

    st.subheader("Actual vs Predicted")

    fig2, ax2 = plt.subplots(figsize=(12,5))

    ax2.plot(
        monthly_sales["Order Date"].iloc[split:],
        y_test,
        marker="o",
        label="Actual"
    )

    ax2.plot(
        monthly_sales["Order Date"].iloc[split:],
        predictions,
        marker="o",
        label="Predicted"
    )

    ax2.legend()

    st.pyplot(fig2)

    # -------------------------
    # FEATURE IMPORTANCE
    # -------------------------

    st.subheader("Feature Importance")

    importance = pd.DataFrame({
        "Feature": FEATURES,
        "Importance": model.feature_importances_
    })

    importance = importance.sort_values(
        "Importance",
        ascending=True
    )

    fig3, ax3 = plt.subplots(figsize=(8,5))

    ax3.barh(
        importance["Feature"],
        importance["Importance"]
    )

    st.pyplot(fig3)

    # -------------------------
    # FUTURE FORECAST
    # -------------------------

    final_model = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    final_model.fit(X, y)

    future_months = 12

    last_date = monthly_sales["Order Date"].max()

    future_dates = pd.date_range(
        start=last_date,
        periods=future_months + 1,
        freq="MS"
    )[1:]

    future_predictions = []

    temp_df = monthly_sales.copy()

    for date in future_dates:

        month = date.month
        quarter = date.quarter
        year = date.year

        month_sin = np.sin(
            2*np.pi*month/12
        )

        month_cos = np.cos(
            2*np.pi*month/12
        )

        lag1 = temp_df["Sales"].iloc[-1]
        lag3 = temp_df["Sales"].iloc[-3]
        lag6 = temp_df["Sales"].iloc[-6]

        rolling3 = (
            temp_df["Sales"]
            .tail(3)
            .mean()
        )

        rolling6 = (
            temp_df["Sales"]
            .tail(6)
            .mean()
        )

        row = pd.DataFrame({
            "month":[month],
            "quarter":[quarter],
            "year":[year],
            "month_sin":[month_sin],
            "month_cos":[month_cos],
            "lag1":[lag1],
            "lag3":[lag3],
            "lag6":[lag6],
            "rolling3":[rolling3],
            "rolling6":[rolling6]
        })

        pred = final_model.predict(row)[0]

        future_predictions.append(pred)

        temp_df = pd.concat([
            temp_df,
            pd.DataFrame({
                "Sales":[pred]
            })
        ])

    forecast_df = pd.DataFrame({
        "Date": future_dates,
        "Forecast": future_predictions
    })

    st.subheader("Future 12-Month Forecast")

    fig4, ax4 = plt.subplots(figsize=(12,5))

    ax4.plot(
        monthly_sales["Order Date"],
        monthly_sales["Sales"],
        label="Historical Sales"
    )

    ax4.plot(
        forecast_df["Date"],
        forecast_df["Forecast"],
        linestyle="--",
        linewidth=3,
        label="Forecast"
    )

    ax4.legend()

    st.pyplot(fig4)

    # -------------------------
    # BUSINESS INSIGHTS
    # -------------------------

    st.subheader("Business Insights")

    st.success(f"""
    • Model achieved an R² score of {r2:.2f}

    • Month and recent sales history are strong predictors

    • Sales demonstrate seasonal patterns

    • Forecasts can support inventory planning

    • Forecasts can support staffing and budgeting decisions
    """)
