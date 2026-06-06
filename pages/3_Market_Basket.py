import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Market Basket", layout="wide")

st.title("🛒 Market Basket Analysis (Bulletproof Version)")

# ==================================================
# SAFE DATA LOADER (NO CRASH EVER)
# ==================================================

@st.cache_data
def load_data():
    file_paths = [
        "data/Assignment-1_Data.csv.csv",
        "data/Assignment-1_Data.csv"
    ]

    for path in file_paths:
        try:
            df = pd.read_csv(path, sep=";", engine="python")
            st.success(f"Loaded file: {path}")
            return df
        except:
            continue

    return pd.DataFrame()

df = load_data()

if df.empty:
    st.error("❌ Dataset could not be loaded. Check file path.")
    st.stop()

# ==================================================
# CLEAN COLUMNS
# ==================================================

df.columns = df.columns.str.strip()

st.write("📌 Columns detected:")
st.write(df.columns.tolist())

# ==================================================
# REQUIRED COLUMNS CHECK (SAFE)
# ==================================================

required_cols = ["BillNo", "Itemname"]

missing = [col for col in required_cols if col not in df.columns]

if missing:
    st.error(f"Missing columns: {missing}")
    st.stop()

invoice_col = "BillNo"
product_col = "Itemname"

# ==================================================
# CLEAN DATA
# ==================================================

df = df[[invoice_col, product_col]].dropna()
df = df.drop_duplicates()

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

if basket.shape[1] < 2:
    st.error("Not enough product variety for analysis")
    st.stop()

# ==================================================
# CO-OCCURRENCE MATRIX
# ==================================================

co_matrix = basket.T.dot(basket)

np.fill_diagonal(co_matrix.values, 0)

co_df = pd.DataFrame(co_matrix)

# ==================================================
# RECOMMENDER
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
# SAFE INSIGHT
# ==================================================

if len(recommendations) > 1:
    st.success(
        f"""
Customers who buy **{product}**
also buy **{recommendations.iloc[1]['Product']}**
"""
    )

# ==================================================
# DEBUG VIEW (VERY IMPORTANT)
# ==================================================

with st.expander("🔍 Debug Data"):
    st.write("Shape:", df.shape)
    st.dataframe(df.head(50))
