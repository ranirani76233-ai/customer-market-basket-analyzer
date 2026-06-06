import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Market Basket Analysis",
    page_icon="🛒",
    layout="wide"
)

st.title("🛒 Market Basket Analysis (Stable Version)")

# ==================================================
# SAFE DATA LOADING
# ==================================================

@st.cache_data
def load_data():
    try:
        # FIX: auto-detect delimiter + ignore bad rows
        df = pd.read_csv(
            "data/Assignment-1_Data.csv.csv",
            engine="python",
            on_bad_lines="skip"
        )
        return df
    except Exception as e:
        st.error(f"CSV Load Error: {e}")
        return pd.DataFrame()

df = load_data()

if df.empty:
    st.error("Dataset not loaded or empty")
    st.stop()

# ==================================================
# CLEAN COLUMNS
# ==================================================

df.columns = df.columns.str.strip()

st.write("Detected Columns:", df.columns.tolist())

# ==================================================
# AUTO DETECT REQUIRED COLUMNS
# ==================================================

invoice_col = None
product_col = None

for col in df.columns:
    c = col.lower()

    if c in ["billno", "invoice", "invoiceno", "invoice_no"]:
        invoice_col = col

    if c in ["itemname", "product", "description", "stockcode"]:
        product_col = col

# ==================================================
# VALIDATION
# ==================================================

if not invoice_col or not product_col:
    st.error("Required columns not found in dataset")
    st.stop()

# ==================================================
# CLEAN DATA
# ==================================================

df = df[[invoice_col, product_col]].dropna()
df = df.drop_duplicates()

# LIMIT DATA (VERY IMPORTANT FOR STREAMLIT CLOUD)
if len(df) > 20000:
    df = df.sample(20000, random_state=42)

# ==================================================
# POPULAR PRODUCTS (ALWAYS WORKS)
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
# CREATE BASKET MATRIX (SAFE)
# ==================================================

basket = (
    df.groupby([invoice_col, product_col])
    .size()
    .unstack(fill_value=0)
)

basket = basket.astype(bool).astype(int)

# ==================================================
# SAFETY CHECK
# ==================================================

if basket.shape[1] < 2:
    st.warning("Not enough product diversity for associations")
    st.stop()

# ==================================================
# SIMPLE CO-OCCURRENCE (NO APRIORI = NO CRASH)
# ==================================================

st.subheader("🎯 Product Recommendations")

co_matrix = basket.T.dot(basket)

np.fill_diagonal(co_matrix.values, 0)

co_df = pd.DataFrame(co_matrix)

# ==================================================
# PRODUCT SELECTOR
# ==================================================

selected_product = st.selectbox("Select Product", co_df.columns)

recommendations = (
    co_df[selected_product]
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
)

recommendations.columns = ["Product", "Score"]

st.dataframe(recommendations, use_container_width=True)

# ==================================================
# SAFE INSIGHT (NO ERROR POSSIBLE)
# ==================================================

top_match = recommendations.iloc[1] if len(recommendations) > 1 else None

st.subheader("💡 Insight")

if top_match is not None:
    st.success(
        f"Customers who buy **{selected_product}** also buy **{top_match['Product']}**"
    )
else:
    st.info("Not enough data for strong relationships")

# ==================================================
# OPTIONAL HEATMAP VIEW
# ==================================================

st.subheader("📊 Product Relationship Matrix (Sample)")

st.dataframe(co_df.iloc[:10, :10], use_container_width=True)

# ==================================================
# RAW DATA VIEW
# ==================================================

with st.expander("View Dataset"):
    st.dataframe(df.head(100), use_container_width=True)
