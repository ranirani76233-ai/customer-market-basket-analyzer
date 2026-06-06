import streamlit as st
import pandas as pd
import numpy as np

from mlxtend.frequent_patterns import apriori

# ==================================================
# CONFIG
# ==================================================

st.set_page_config(
    page_title="Market Basket Analysis",
    page_icon="🛒",
    layout="wide"
)

st.title("🛒 Market Basket Analysis (Smart Hybrid)")

# ==================================================
# LOAD DATA
# ==================================================

@st.cache_data
def load_data():
    try:
        return pd.read_csv(
            "data/Assignment-1_Data.csv.csv",
            sep=None,
            engine="python",
            on_bad_lines="skip"
        )
    except:
        return pd.DataFrame()

df = load_data()

if df.empty:
    st.error("Dataset not found")
    st.stop()

df.columns = df.columns.str.strip()

# ==================================================
# DETECT COLUMNS
# ==================================================

invoice_col = None
product_col = None

for col in df.columns:
    c = col.lower()

    if c in ["billno", "invoice", "invoiceno", "invoice_no"]:
        invoice_col = col

    if c in ["itemname", "product", "description", "stockcode"]:
        product_col = col

if not invoice_col or not product_col:
    st.error(f"Missing required columns: {df.columns.tolist()}")
    st.stop()

# ==================================================
# CLEAN DATA
# ==================================================

df = df[[invoice_col, product_col]].dropna()
df = df.drop_duplicates()

# LIMIT SIZE FOR SPEED
if len(df) > 20000:
    df = df.sample(20000, random_state=42)

# ==================================================
# TOP PRODUCTS (ALWAYS WORKS BASELINE)
# ==================================================

st.subheader("🔥 Top Products (Popularity Based)")

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

basket = basket.apply(lambda x: (x > 0).astype(int))

# ==================================================
# SIMPLE CO-OCCURRENCE MATRIX (FALLBACK ENGINE)
# ==================================================

co_matrix = basket.T.dot(basket)

np.fill_diagonal(co_matrix.values, 0)

co_df = pd.DataFrame(co_matrix)

# ==================================================
# PRODUCT SELECTOR
# ==================================================

st.subheader("🎯 Product Recommendations")

selected_product = st.selectbox("Choose a product", co_df.columns)

# ==================================================
# RECOMMENDATIONS (ALWAYS WORKS)
# ==================================================

recommendations = (
    co_df[selected_product]
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
)

recommendations.columns = ["Product", "Score"]

st.dataframe(recommendations, use_container_width=True)

# ==================================================
# FREQUENT PAIRS (OPTIONAL INSIGHT)
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
    st.info("No strong product pairs found, showing popularity-based insights instead.")

# ==================================================
# INSIGHT (ALWAYS SHOWS)
# ==================================================

top_pair = pairs_df.iloc[0] if not pairs_df.empty else None

st.subheader("💡 Insight")

if top_pair is not None:
    st.success(f"""
Customers who buy **{top_pair['Product 1']}**
also buy **{top_pair['Product 2']}**
""")
else:
    st.info("Dataset shows weak co-occurrence. Recommendations are based on popularity model.")
