import streamlit as st
import pandas as pd
import numpy as np

from mlxtend.frequent_patterns import apriori

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="Market Basket Analysis",
    page_icon="🛒",
    layout="wide"
)

st.title("🛒 Market Basket Analysis (Smart Recommendation System)")

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
    st.error("Dataset not loaded")
    st.stop()

# ==================================================
# CLEAN COLUMNS
# ==================================================

df.columns = df.columns.str.strip()

# ==================================================
# AUTO DETECT COLUMNS
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
    st.error(f"Missing columns: {df.columns.tolist()}")
    st.stop()

# ==================================================
# CLEAN DATA
# ==================================================

df = df[[invoice_col, product_col]].dropna()
df = df.drop_duplicates()

# limit dataset for performance
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
    st.error("Not enough product diversity")
    st.stop()

# ==================================================
# APRIORI (OPTIONAL INSIGHTS)
# ==================================================

st.subheader("📦 Frequent Itemsets (Insights Only)")

frequent_itemsets = apriori(
    basket,
    min_support=0.01,
    use_colnames=True
)

if not frequent_itemsets.empty:
    st.dataframe(
        frequent_itemsets.sort_values("support", ascending=False).head(10),
        use_container_width=True
    )
else:
    st.info("No strong frequent itemsets found (this is normal for sparse data).")

# ==================================================
# SMART RECOMMENDATION ENGINE (FIXED)
# ==================================================

st.subheader("🎯 Product Recommendation System")

# Convert to numpy for speed
basket_np = basket.values

# Jaccard similarity
intersection = basket.T.dot(basket)

sum_vals = basket.sum(axis=0).values.reshape(-1, 1)
union = sum_vals + sum_vals.T - intersection

similarity = intersection / (union + 1e-9)

co_df = pd.DataFrame(similarity, index=basket.columns, columns=basket.columns)

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

recommendations.columns = ["Product", "Similarity Score"]

st.dataframe(recommendations, use_container_width=True)

# ==================================================
# INSIGHT (SAFE)
# ==================================================

top_match = recommendations.iloc[1]  # skip itself

st.success(f"""
Customers who buy **{selected_product}**
are also likely to buy **{top_match['Product']}**
(Score: {top_match['Similarity Score']:.2f})
""")

# ==================================================
# FREQUENT PAIRS (OPTIONAL)
# ==================================================

st.subheader("📊 Product Relationship Heatmap (Top View)")

st.dataframe(co_df.iloc[:10, :10], use_container_width=True)

# ==================================================
# DATA VIEW
# ==================================================

with st.expander("View Dataset"):
    st.dataframe(df.head(100), use_container_width=True)
