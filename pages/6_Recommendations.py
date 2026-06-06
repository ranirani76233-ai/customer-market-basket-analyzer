import streamlit as st
import pandas as pd
import numpy as np

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="Product Recommendations",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Product Recommendation System")

# ==================================================
# LOAD DATA (SAFE)
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
        st.error(f"Dataset Error: {e}")
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
# AUTO DETECT COLUMNS
# ==================================================

invoice_col = None
product_col = None
qty_col = None

for col in df.columns:

    c = col.lower()

    if c in ["billno", "invoice", "invoiceno"]:
        invoice_col = col

    if c in ["itemname", "product", "description"]:
        product_col = col

    if c in ["qty", "quantity"]:
        qty_col = col

# ==================================================
# VALIDATION
# ==================================================

if invoice_col is None or product_col is None:

    st.error(f"""
    Required columns missing!

    Invoice Column: {invoice_col}
    Product Column: {product_col}

    Available Columns:
    {df.columns.tolist()}
    """)
    st.stop()

# ==================================================
# CLEAN DATA
# ==================================================

df = df[[invoice_col, product_col]].dropna()
df = df.drop_duplicates()

# ==================================================
# POPULARITY BASED RECOMMENDATION
# ==================================================

st.subheader("🔥 Top Selling Products")

top_products = (
    df.groupby(product_col)
    .size()
    .reset_index(name="Count")
    .sort_values("Count", ascending=False)
    .head(10)
)

st.dataframe(top_products, use_container_width=True)

# ==================================================
# SIMPLE CO-OCCURRENCE MATRIX
# ==================================================

st.subheader("🤝 Product Association (Simple Model)")

basket = (
    df.groupby([invoice_col, product_col])
    .size()
    .unstack(fill_value=0)
)

basket = basket.applymap(lambda x: 1 if x > 0 else 0)

co_matrix = basket.T.dot(basket)

np.fill_diagonal(co_matrix.values, 0)

co_df = pd.DataFrame(co_matrix)

# ==================================================
# PRODUCT SELECTOR
# ==================================================

selected_product = st.selectbox(
    "Select a product",
    co_df.columns
)

# ==================================================
# RECOMMENDATIONS
# ==================================================

if selected_product:

    recommendations = (
        co_df[selected_product]
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    recommendations.columns = ["Product", "Score"]

    st.subheader(f"🎯 Recommended products for: {selected_product}")

    st.dataframe(recommendations, use_container_width=True)

# ==================================================
# MOST COMMON PAIRS
# ==================================================

st.subheader("📦 Frequently Bought Together")

pairs = []

for i in range(len(co_df.columns)):

    for j in range(i + 1, len(co_df.columns)):

        prod1 = co_df.columns[i]
        prod2 = co_df.columns[j]

        score = co_df.loc[prod1, prod2]

        if score > 0:
            pairs.append((prod1, prod2, score))

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

with st.expander("View Dataset"):

    st.dataframe(df.head(100), use_container_width=True)
