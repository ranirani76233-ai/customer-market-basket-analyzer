import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Market Basket Analysis", layout="wide")

st.title("🛒 Market Basket Analysis (Fixed CSV Format)")

# ==================================================
# LOAD DATA (FIXED: semicolon delimiter)
# ==================================================

@st.cache_data
def load_data():
    try:
        df = pd.read_csv(
            "data/Assignment-1_Data.csv.csv",
            sep=";"   # 🔥 IMPORTANT FIX
        )
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return pd.DataFrame()

df = load_data()

if df.empty:
    st.error("Dataset not loaded")
    st.stop()

# ==================================================
# CLEAN COLUMNS
# ==================================================

df.columns = df.columns.str.strip()

st.write("Detected Columns:", df.columns.tolist())

# ==================================================
# REQUIRED COLUMNS (NOW FIXED)
# ==================================================

invoice_col = "BillNo"
product_col = "Itemname"

# safety check
if invoice_col not in df.columns or product_col not in df.columns:
    st.error("Required columns not found after fixing delimiter")
    st.write(df.columns.tolist())
    st.stop()

# ==================================================
# CLEAN DATA
# ==================================================

df = df[[invoice_col, product_col]].dropna()
df = df.drop_duplicates()

# limit for performance
if len(df) > 20000:
    df = df.sample(20000, random_state=42)

# ==================================================
# TOP PRODUCTS
# ==================================================

st.subheader("🔥 Top Products")

top_products = (
    df[product_col]
    .value_counts()
    .head(10)
    .reset_index()
)

top_products.columns = ["Product", "Count"]

st.dataframe(top_products, use_container_width=True)

# ==================================================
# BASKET MATRIX
# ==================================================

basket = (
    df.groupby([invoice_col, product_col])
    .size()
    .unstack(fill_value=0)
)

basket = basket.astype(bool).astype(int)

# ==================================================
# CO-OCCURRENCE MATRIX
# ==================================================

co_matrix = basket.T.dot(basket)
np.fill_diagonal(co_matrix.values, 0)

co_df = pd.DataFrame(co_matrix)

# ==================================================
# RECOMMENDER SYSTEM
# ==================================================

st.subheader("🎯 Product Recommendations")

product = st.selectbox("Select Product", co_df.columns)

recommendations = (
    co_df[product]
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
)

recommendations.columns = ["Product", "Score"]

st.dataframe(recommendations, use_container_width=True)

# ==================================================
# INSIGHT
# ==================================================

if len(recommendations) > 1:
    st.success(
        f"Customers who buy **{product}** also buy **{recommendations.iloc[1]['Product']}**"
    )

# ==================================================
# RAW DATA
# ==================================================

with st.expander("View Dataset"):
    st.dataframe(df.head(100), use_container_width=True)
