import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("Sales Forecasting Dashboard")

uploaded_file = st.file_uploader(
    "Upload Superstore CSV",
    type=["csv"]
)

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")

    st.dataframe(df.head())

    df["Order Date"] = pd.to_datetime(df["Order Date"])

    monthly_sales = (
        df.groupby(
            pd.Grouper(
                key="Order Date",
                freq="M"
            )
        )["Sales"]
        .sum()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10,4))

    ax.plot(
        monthly_sales["Order Date"],
        monthly_sales["Sales"]
    )

    ax.set_title("Monthly Sales Trend")

    st.pyplot(fig)