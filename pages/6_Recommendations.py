import streamlit as st
import pandas as pd
import numpy as np

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="Recommendations",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Product Recommendation System")

# ==================================================
# SAFE DATA LOADING
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

if df is None or df.empty:
    st.error("Dataset is empty or not loaded properly.")
    st.stop()

# ==================================================
# CLEAN COLUMNS
# ==================================================

df.columns = df.columns.str.strip()

st.write("Columns detected:", df.columns.tolist())

# ==================================================
# AUTO DETECT COLUMNS (SAFE)
# ==================================================

invoice_col = None
product_col = None

for col in df.columns:

    c = col.lower()

    if c in ["billno", "invoice", "invoiceno", "bill no"]:
        invoice_col = col

    if c in ["itemname", "product", "description", "stockcode"]:
        product_col = col

# ==================================================
# VALIDATION
# ==================================================

if not invoice_col or not product_col:

    st.error("Required columns not found!")
    st.stop()

# ==================================================
# CLEAN DATA
# ==================================================

df = df[[invoice_col, product_col]].dropna()
df = df.drop_duplicates()

# LIMIT DATA (IMPORTANT FIX FOR CRASH)
if len(df) > 30000:
    df = df.sample(30000, random_state=42)

# ==================================================
# POPULAR PRODUCTS
# ==================================================

st.subheader("🔥 Top Products")

top_products = (
    df.groupby(product_col)
    .size()
    .reset_index(name="Count")
    .sort_values("Count", ascending=False)
    .head(10)
)

st.dataframe(top_products, use_container_width=True)

# ==================================================
# CREATE SMALL BASKET (SAFE)
# ==================================================

basket = (
    df.groupby([invoice_col, product_col])
    .size()
    .unstack(fill_value=0)
)

# convert to binary safely
basket = basket.apply(lambda x: (x > 0).astype(int))

# ==================================================
# LIMIT FEATURES (IMPORTANT FIX)
# ==================================================

# keep only top 50 products to avoid crash
top_cols = basket.sum().sort_values(ascending=False).head(50).index
basket = basket[top_cols]

# ==================================================
# CO-OCCURRENCE MATRIX
# ==================================================

co_matrix = basket.T.dot(basket)

np.fill_diagonal(co_matrix.values, 0)

co_df = pd.DataFrame(co_matrix)

# ==================================================
# PRODUCT SELECTOR (SAFE)
# ==================================================

if co_df.shape[0] == 0:
    st.error("Not enough data for recommendations.")
    st.stop()

selected_product = st.selectbox("Select Product", co_df.columns)

# ==================================================
# RECOMMENDATIONS
# ==================================================

st.subheader("🎯 Recommended Products")

if selected_product in co_df.columns:

    recommendations = (
        co_df[selected_product]
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    recommendations.columns = ["Product", "Score"]

    st.dataframe(recommendations, use_container_width=True)

# ==================================================
# FREQUENT PAIRS
# ==================================================

st.subheader("📦 Frequently Bought Together")

pairs = []

cols = list(co_df.columns)

for i in range(len(cols)):

    for j in range(i + 1, len(cols)):

        score = co_df.loc[cols[i], cols[j]]

        if score > 0:
            pairs.append((cols[i], cols[j], score))

pairs_df = pd.DataFrame(pairs, columns=["Product 1", "Product 2", "Score"])

if not pairs_df.empty:

    st.dataframe(
        pairs_df.sort_values("Score", ascending=False).head(15),
        use_container_width=True
    )

else:

    st.warning("No strong product pairs found.")

# ==================================================
# RAW DATA
# ==================================================

with st.expander("View Data"):
    st.dataframe(df.head(100), use_container_width=True)
