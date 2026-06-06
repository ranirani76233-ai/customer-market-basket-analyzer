import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="Advanced Retail Analytics",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Advanced Retail Analytics")

# ==================================================
# LOAD DATA
# ==================================================

@st.cache_data
def load_data():
    try:
        df = pd.read_csv(
            "data/Assignment-1_Data.csv.csv",
            sep=None,
            engine="python",
            on_bad_lines="skip"
        )
        return df

    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return pd.DataFrame()


df = load_data()

if df.empty:
    st.stop()

# ==================================================
# CLEAN COLUMNS
# ==================================================

df.columns = df.columns.str.strip()

st.write("Detected Columns:", list(df.columns))

# ==================================================
# AUTO DETECT IMPORTANT COLUMNS
# ==================================================

qty_col = None
price_col = None
date_col = None

for col in df.columns:

    col_lower = col.lower()

    if col_lower in ["qty", "quantity"]:
        qty_col = col

    if col_lower in ["price", "unitprice"]:
        price_col = col

    if col_lower == "date":
        date_col = col

# ==================================================
# CREATE REVENUE
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
        Revenue could not be calculated.

        Quantity Column Found: {qty_col}
        Price Column Found: {price_col}
        """
    )

    st.stop()

# ==================================================
# KPI SECTION
# ==================================================

st.subheader("Business Overview")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Total Revenue",
    f"${df['Revenue'].sum():,.0f}"
)

col2.metric(
    "Total Records",
    len(df)
)

col3.metric(
    "Columns",
    len(df.columns)
)

# ==================================================
# REVENUE DISTRIBUTION
# ==================================================

st.subheader("Revenue Distribution")

fig = px.histogram(
    df,
    x="Revenue",
    nbins=40
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ==================================================
# MONTHLY TREND
# ==================================================

if date_col:

    df[date_col] = pd.to_datetime(
        df[date_col],
        errors="coerce"
    )

    monthly = (
        df.groupby(
            df[date_col].dt.to_period("M")
        )["Revenue"]
        .sum()
        .reset_index()
    )

    monthly[date_col] = monthly[
        date_col
    ].astype(str)

    st.subheader("Monthly Revenue Trend")

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
# NUMERIC ANALYSIS
# ==================================================

st.subheader("Correlation Analysis")

numeric_df = df.select_dtypes(
    include=np.number
)

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
# DATA PREVIEW
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
    file_name="analytics_data.csv",
    mime="text/csv"
)
