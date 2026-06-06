import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
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
    try:
        df = pd.read_csv(
            "data/Assignment-1_Data.csv.csv",
            sep=";"
        )
    except:
        df = pd.read_csv(
            "data/Assignment-1_Data.csv.csv"
        )

    return df

df = load_data()

# ==================================================
# CLEAN COLUMN NAMES
# ==================================================

df.columns = df.columns.str.strip()

# ==================================================
# AUTO DETECT COLUMNS
# ==================================================

qty_col = None
price_col = None
date_col = None
customer_col = None
product_col = None
bill_col = None
country_col = None

for col in df.columns:

    c = col.lower()

    if c in ["qty", "quantity"]:
        qty_col = col

    elif c in ["price", "unitprice", "sellingprice"]:
        price_col = col

    elif c == "date":
        date_col = col

    elif c in ["customerid", "customer_id"]:
        customer_col = col

    elif c in ["itemname", "item_name", "product", "description"]:
        product_col = col

    elif c in ["billno", "invoice", "invoiceno"]:
        bill_col = col

    elif c == "country":
        country_col = col

# ==================================================
# DATE
# ==================================================

if date_col:
    df[date_col] = pd.to_datetime(
        df[date_col],
        errors="coerce"
    )

# ==================================================
# REVENUE
# ==================================================

if qty_col and price_col:

    df[qty_col] = pd.to_numeric(
        df[qty_col],
        errors="coerce"
    )

    df[price_col] = pd.to_numeric(
        df[price_col],
        errors="coerce"
    )

    df["Revenue"] = (
        df[qty_col].fillna(0)
        *
        df[price_col].fillna(0)
    )

else:

    st.error(
        f"""
        Revenue cannot be created.

        Quantity Column Found: {qty_col}
        Price Column Found: {price_col}

        Dataset Columns:
        {list(df.columns)}
        """
    )
    st.stop()

# ==================================================
# KPI SECTION
# ==================================================

st.subheader("Business Overview")

col1, col2, col3, col4 = st.columns(4)

total_revenue = df["Revenue"].sum()

total_customers = (
    df[customer_col].nunique()
    if customer_col else 0
)

total_products = (
    df[product_col].nunique()
    if product_col else 0
)

total_transactions = (
    df[bill_col].nunique()
    if bill_col else 0
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
# MONTHLY TREND
# ==================================================

if date_col:

    st.subheader("📈 Revenue Trend")

    monthly = (
        df.groupby(
            df[date_col].dt.to_period("M")
        )["Revenue"]
        .sum()
        .reset_index()
    )

    monthly[date_col] = (
        monthly[date_col]
        .astype(str)
    )

    fig = px.line(
        monthly,
        x=date_col,
        y="Revenue",
        markers=True
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ==================================================
# TOP PRODUCTS
# ==================================================

col1, col2 = st.columns(2)

if product_col:

    with col1:

        st.subheader("🏆 Top Products")

        top_products = (
            df.groupby(product_col)["Revenue"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )

        fig = px.bar(
            top_products,
            x="Revenue",
            y=product_col,
            orientation="h"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

# ==================================================
# TOP CUSTOMERS
# ==================================================

if customer_col:

    with col2:

        st.subheader("👑 Top Customers")

        top_customers = (
            df.groupby(customer_col)["Revenue"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )

        fig = px.bar(
            top_customers,
            x=customer_col,
            y="Revenue"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

# ==================================================
# COUNTRY ANALYSIS
# ==================================================

if country_col:

    st.subheader("🌍 Revenue by Country")

    country_sales = (
        df.groupby(country_col)["Revenue"]
        .sum()
        .sort_values(ascending=False)
        .head(15)
        .reset_index()
    )

    fig = px.bar(
        country_sales,
        x=country_col,
        y="Revenue"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ==================================================
# PRODUCT PERFORMANCE
# ==================================================

if product_col:

    st.subheader("📦 Product Performance")

    product_perf = (
        df.groupby(product_col)
        .agg({
            qty_col: "sum",
            "Revenue": "sum"
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

if customer_col and bill_col:

    st.subheader("👥 Customer Analysis")

    customer_df = (
        df.groupby(customer_col)
        .agg({
            "Revenue": "sum",
            bill_col: "nunique"
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
        size="TotalSpend"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ==================================================
# CORRELATION
# ==================================================

st.subheader("🔥 Correlation Analysis")

numeric_df = df.select_dtypes(include=np.number)

if len(numeric_df.columns) > 1:

    corr = numeric_df.corr()

    fig = px.imshow(
        corr,
        text_auto=True
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ==================================================
# ROOT CAUSE ANALYSIS
# ==================================================

if qty_col and price_col:

    st.subheader("🎯 Revenue Driver Analysis")

    X = df[[qty_col, price_col]].fillna(0)
    y = df["Revenue"]

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    model.fit(X, y)

    importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    })

    fig = px.bar(
        importance,
        x="Importance",
        y="Feature",
        orientation="h"
    )

    st.plotly_chart(
        fig,
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

csv = df.to_csv(index=False).encode("utf-8")

st.download_button(
    "⬇ Download Dataset",
    csv,
    "retail_data.csv",
    "text/csv"
)
