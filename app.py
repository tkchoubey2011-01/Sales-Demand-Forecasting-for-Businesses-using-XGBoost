import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("Sales Forecasting Dashboard")

uploaded_file = st.file_uploader(
    "Upload Superstore CSV",
    type=["csv"]
)

if uploaded_file is not None:

    try:
        df = pd.read_csv(uploaded_file, encoding="utf-8")

    except UnicodeDecodeError:

        uploaded_file.seek(0)

        try:
            df = pd.read_csv(
                uploaded_file,
                encoding="latin1"
            )

        except:

            uploaded_file.seek(0)

            df = pd.read_csv(
                uploaded_file,
                encoding="cp1252"
            )

    st.subheader("Dataset Preview")

    st.dataframe(df.head())

    df["Order Date"] = pd.to_datetime(
        df["Order Date"],
        errors="coerce"
    )

    df = df.dropna(subset=["Order Date"])

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

    fig, ax = plt.subplots(figsize=(10,4))

    ax.plot(
        monthly_sales["Order Date"],
        monthly_sales["Sales"]
    )

    ax.set_title("Monthly Sales Trend")

    st.pyplot(fig)
