import streamlit as st
import pandas as pd
import numpy as np
try:
    import plotly.express as px
except Exception as e:
    st.error(f"Plotly Error: {e}")
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Retail Analytics Dashboard")

# ==================================================
# LOAD DATA
# ==================================================

@st.cache_data
def load_data():

    df = pd.read_csv(
        'data/Assignment-1_Data.csv'
        
    )

    return df

df = load_data()

# ==================================================
# DATA PREPARATION
# ==================================================

if "Date" in df.columns:
    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

if (
    "Qty" in df.columns and
    "Price" in df.columns
):

    df["Revenue"] = (
        df["Qty"] * df["Price"]
    )

# ==================================================
# KPI SECTION
# ==================================================

st.subheader("Business Overview")

col1, col2, col3, col4 = st.columns(4)

total_revenue = (
    df["Revenue"].sum()
    if "Revenue" in df.columns
    else 0
)

total_customers = (
    df["CustomerID"].nunique()
    if "CustomerID" in df.columns
    else 0
)

total_products = (
    df["Itemname"].nunique()
    if "Itemname" in df.columns
    else 0
)

total_transactions = (
    df["BillNo"].nunique()
    if "BillNo" in df.columns
    else 0
)

col1.metric(
    "Total Revenue",
    f"${total_revenue:,.0f}"
)

col2.metric(
    "Customers",
    f"{total_customers:,}"
)

col3.metric(
    "Products",
    f"{total_products:,}"
)

col4.metric(
    "Transactions",
    f"{total_transactions:,}"
)

st.divider()

# ==================================================
# MONTHLY SALES TREND
# ==================================================

st.subheader("📈 Revenue Trend")

if "Date" in df.columns:

    monthly = (
        df
        .groupby(
            df["Date"].dt.to_period("M")
        )["Revenue"]
        .sum()
        .reset_index()
    )

    monthly["Date"] = monthly["Date"].astype(str)

    fig = px.line(
        monthly,
        x="Date",
        y="Revenue",
        markers=True,
        title="Monthly Revenue Trend"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ==================================================
# TOP PRODUCTS
# ==================================================

col1, col2 = st.columns(2)

with col1:

    st.subheader("🏆 Top Products")

    if (
        "Itemname" in df.columns and
        "Revenue" in df.columns
    ):

        top_products = (

            df.groupby(
                "Itemname"
            )["Revenue"]

            .sum()

            .sort_values(
                ascending=False
            )

            .head(10)

            .reset_index()

        )

        fig = px.bar(
            top_products,
            x="Revenue",
            y="Itemname",
            orientation="h"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

# ==================================================
# TOP CUSTOMERS
# ==================================================

with col2:

    st.subheader("👑 Top Customers")

    if (
        "CustomerID" in df.columns and
        "Revenue" in df.columns
    ):

        top_customers = (

            df.groupby(
                "CustomerID"
            )["Revenue"]

            .sum()

            .sort_values(
                ascending=False
            )

            .head(10)

            .reset_index()

        )

        fig = px.bar(
            top_customers,
            x="CustomerID",
            y="Revenue"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

# ==================================================
# COUNTRY ANALYSIS
# ==================================================

if "Country" in df.columns:

    st.subheader("🌍 Revenue by Country")

    country_sales = (

        df.groupby(
            "Country"
        )["Revenue"]

        .sum()

        .sort_values(
            ascending=False
        )

        .head(15)

        .reset_index()

    )

    fig = px.bar(
        country_sales,
        x="Country",
        y="Revenue"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ==================================================
# PRODUCT PERFORMANCE
# ==================================================

st.subheader("📦 Product Performance")

if (
    "Itemname" in df.columns and
    "Qty" in df.columns
):

    product_perf = (

        df.groupby(
            "Itemname"
        )

        .agg({

            "Qty":"sum",
            "Revenue":"sum"

        })

        .reset_index()

        .sort_values(
            "Revenue",
            ascending=False
        )

        .head(20)

    )

    st.dataframe(
        product_perf,
        use_container_width=True
    )

# ==================================================
# CUSTOMER ANALYSIS
# ==================================================

st.subheader("👥 Customer Analysis")

if (
    "CustomerID" in df.columns and
    "Revenue" in df.columns
):

    customer_df = (

        df.groupby(
            "CustomerID"
        )

        .agg({

            "Revenue":"sum",

            "BillNo":"nunique"

        })

        .reset_index()

    )

    customer_df.columns = [

        "CustomerID",
        "TotalSpend",
        "Orders"

    ]

    fig = px.scatter(

        customer_df,

        x="Orders",

        y="TotalSpend",

        size="TotalSpend",

        title="Orders vs Spend"

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ==================================================
# CORRELATION ANALYSIS
# ==================================================

st.subheader("🔥 Correlation Analysis")

numeric_df = df.select_dtypes(
    include=np.number
)

if len(numeric_df.columns) > 1:

    corr = numeric_df.corr()

    fig = px.imshow(
        corr,
        text_auto=True,
        aspect="auto"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ==================================================
# ROOT CAUSE ANALYSIS
# ==================================================

st.subheader("🎯 Key Driver Analysis")

required_cols = [
    "Qty",
    "Price",
    "Revenue"
]

if all(
    col in df.columns
    for col in required_cols
):

    X = df[[
        "Qty",
        "Price"
    ]]

    y = df["Revenue"]

    model = RandomForestRegressor(
        random_state=42
    )

    model.fit(X, y)

    importance = pd.DataFrame({

        "Feature":
        X.columns,

        "Importance":
        model.feature_importances_

    })

    importance = (

        importance

        .sort_values(
            "Importance",
            ascending=False
        )

    )

    fig = px.bar(

        importance,

        x="Importance",

        y="Feature",

        orientation="h",

        title="Revenue Drivers"

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.dataframe(
        importance,
        use_container_width=True
    )

# ==================================================
# RAW DATA
# ==================================================

with st.expander("View Dataset"):

    st.dataframe(
        df.head(100),
        use_container_width=True
    )

# ==================================================
# DOWNLOAD
# ==================================================

csv = df.to_csv(
    index=False
).encode("utf-8")

st.download_button(

    label="⬇ Download Dataset",

    data=csv,

    file_name="retail_data.csv",

    mime="text/csv"

)
