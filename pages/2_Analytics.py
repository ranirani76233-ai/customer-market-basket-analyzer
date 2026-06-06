import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="Analytics",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Advanced Retail Analytics")

# ==================================================
# LOAD DATA
# ==================================================

@st.cache_data
def load_data():

    df = pd.read_csv(
        'data/Assignment-1_Data.csv.csv'      
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
# SALES ANALYTICS
# ==================================================

st.header("💰 Sales Analytics")

col1, col2 = st.columns(2)

with col1:

    revenue_stats = pd.DataFrame({

        "Metric": [
            "Total Revenue",
            "Average Revenue",
            "Maximum Sale",
            "Minimum Sale"
        ],

        "Value": [

            round(df["Revenue"].sum(), 2),

            round(df["Revenue"].mean(), 2),

            round(df["Revenue"].max(), 2),

            round(df["Revenue"].min(), 2)

        ]
    })

    st.dataframe(
        revenue_stats,
        use_container_width=True
    )

with col2:

    fig = px.histogram(
        df,
        x="Revenue",
        nbins=50,
        title="Revenue Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ==================================================
# DAILY REVENUE TREND
# ==================================================

if "Date" in df.columns:

    st.subheader("📅 Daily Revenue Trend")

    daily_sales = (

        df.groupby(
            df["Date"].dt.date
        )["Revenue"]

        .sum()

        .reset_index()

    )

    fig = px.line(

        daily_sales,

        x="Date",

        y="Revenue",

        markers=True

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ==================================================
# PRODUCT ANALYTICS
# ==================================================

st.header("📦 Product Analytics")

product_analysis = (

    df.groupby("Itemname")

    .agg({

        "Qty": "sum",

        "Revenue": "sum"

    })

    .reset_index()

    .sort_values(
        "Revenue",
        ascending=False
    )

)

top_products = product_analysis.head(15)

fig = px.bar(

    top_products,

    x="Revenue",

    y="Itemname",

    orientation="h",

    title="Top Revenue Products"

)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ==================================================
# CUSTOMER ANALYTICS
# ==================================================

st.header("👥 Customer Analytics")

customer_df = (

    df.groupby("CustomerID")

    .agg({

        "Revenue": "sum",

        "BillNo": "nunique",

        "Qty": "sum"

    })

    .reset_index()

)

customer_df.columns = [

    "CustomerID",

    "TotalSpend",

    "Orders",

    "ItemsPurchased"

]

fig = px.scatter(

    customer_df,

    x="Orders",

    y="TotalSpend",

    size="ItemsPurchased",

    title="Customer Spend Analysis"

)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ==================================================
# RFM ANALYSIS
# ==================================================

st.header("🎯 RFM Analysis")

if "Date" in df.columns:

    snapshot_date = df["Date"].max()

    rfm = (

        df.groupby("CustomerID")

        .agg({

            "Date": lambda x:
            (snapshot_date - x.max()).days,

            "BillNo": "nunique",

            "Revenue": "sum"

        })

        .reset_index()

    )

    rfm.columns = [

        "CustomerID",

        "Recency",

        "Frequency",

        "Monetary"

    ]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Avg Recency",
            round(rfm["Recency"].mean(), 1)
        )

    with col2:
        st.metric(
            "Avg Frequency",
            round(rfm["Frequency"].mean(), 1)
        )

    with col3:
        st.metric(
            "Avg Monetary",
            round(rfm["Monetary"].mean(), 2)
        )

    fig = px.scatter(

        rfm,

        x="Frequency",

        y="Monetary",

        color="Recency",

        title="RFM Customer Distribution"

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ==================================================
# TOP CUSTOMERS
# ==================================================

st.header("🏆 Top Customers")

top_customers = (

    customer_df

    .sort_values(
        "TotalSpend",
        ascending=False
    )

    .head(20)

)

st.dataframe(
    top_customers,
    use_container_width=True
)

# ==================================================
# COUNTRY ANALYSIS
# ==================================================

if "Country" in df.columns:

    st.header("🌍 Country Analysis")

    country_sales = (

        df.groupby("Country")

        .agg({

            "Revenue": "sum"

        })

        .reset_index()

        .sort_values(
            "Revenue",
            ascending=False
        )

        .head(20)

    )

    fig = px.bar(

        country_sales,

        x="Country",

        y="Revenue",

        title="Revenue by Country"

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ==================================================
# ROOT CAUSE ANALYSIS
# ==================================================

st.header("🔍 Revenue Driver Analysis")

features = ["Qty", "Price"]

if all(
    col in df.columns
    for col in features
):

    X = df[features]

    y = df["Revenue"]

    model = RandomForestRegressor(
        random_state=42
    )

    model.fit(X, y)

    importance = pd.DataFrame({

        "Feature":
        features,

        "Importance":
        model.feature_importances_

    })

    importance = importance.sort_values(
        "Importance",
        ascending=False
    )

    fig = px.bar(

        importance,

        x="Importance",

        y="Feature",

        orientation="h",

        title="Revenue Key Drivers"

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ==================================================
# CORRELATION MATRIX
# ==================================================

st.header("🔥 Correlation Analysis")

numeric_df = df.select_dtypes(
    include=np.number
)

if len(numeric_df.columns) > 1:

    corr = numeric_df.corr()

    fig = px.imshow(

        corr,

        text_auto=True,

        aspect="auto",

        title="Correlation Heatmap"

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ==================================================
# BUSINESS INSIGHTS
# ==================================================

st.header("💡 Business Recommendations")

st.success(
    """
    1. Focus promotions on high-revenue products.

    2. Retain high-value customers identified through RFM analysis.

    3. Bundle products frequently purchased together.

    4. Target countries contributing the highest revenue.

    5. Improve repeat purchases through loyalty campaigns.
    """
)

# ==================================================
# EXPORT ANALYTICS
# ==================================================

st.header("📥 Download Analytics")

csv = customer_df.to_csv(
    index=False
).encode("utf-8")

st.download_button(

    label="Download Customer Analytics",

    data=csv,

    file_name="customer_analytics.csv",

    mime="text/csv"
)
