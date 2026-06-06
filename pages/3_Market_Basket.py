import streamlit as st
import pandas as pd
import numpy as np

from mlxtend.frequent_patterns import apriori, association_rules

# ==================================================
# CONFIG
# ==================================================

st.set_page_config(
    page_title="Market Basket Analysis",
    page_icon="🛒",
    layout="wide"
)

st.title("🛒 Market Basket Analysis")

# ==================================================
# LOAD DATA (ULTRA SAFE)
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
    except:
        return pd.DataFrame()

df = load_data()

if df is None or df.empty:
    st.error("Dataset not found or empty.")
    st.stop()

# ==================================================
# CLEAN COLUMNS
# ==================================================

df.columns = df.columns.str.strip()

# ==================================================
# FIND COLUMNS SAFELY
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
    st.error(f"Missing columns:\n{df.columns.tolist()}")
    st.stop()

# ==================================================
# CLEAN DATA
# ==================================================

df = df[[invoice_col, product_col]].dropna()
df = df.drop_duplicates()

# LIMIT DATA (VERY IMPORTANT FIX)
if len(df) > 25000:
    df = df.sample(25000, random_state=42)

# ==================================================
# KEEP TOP PRODUCTS ONLY (CRITICAL FIX)
# ==================================================

top_products = df[product_col].value_counts().head(40).index
df = df[df[product_col].isin(top_products)]

# ==================================================
# CREATE BASKET
# ==================================================

basket = (
    df.groupby([invoice_col, product_col])
    .size()
    .unstack(fill_value=0)
)

basket = basket.apply(lambda x: (x > 0).astype(int))

# ==================================================
# SAFETY CHECK
# ==================================================

if basket.shape[1] < 2:
    st.error("Not enough product combinations for analysis.")
    st.stop()

# ==================================================
# SIDEBAR SETTINGS (SAFE RANGE)
# ==================================================

st.sidebar.header("Settings")

min_support = st.sidebar.slider("Min Support", 0.001, 0.05, 0.01)
min_confidence = st.sidebar.slider("Min Confidence", 0.1, 1.0, 0.3)

# ==================================================
# FREQUENT ITEMSETS
# ==================================================

st.subheader("📦 Frequent Itemsets")

frequent_itemsets = apriori(
    basket,
    min_support=min_support,
    use_colnames=True
)

if frequent_itemsets.empty:
    st.warning("No frequent itemsets found. Lower support.")
    st.stop()

st.dataframe(
    frequent_itemsets.sort_values("support", ascending=False).head(20),
    use_container_width=True
)

# ==================================================
# ASSOCIATION RULES (SAFE + PROTECTED)
# ==================================================

st.subheader("🔗 Association Rules")

try:
    rules = association_rules(
        frequent_itemsets,
        metric="confidence",
        min_threshold=min_confidence
    )
except:
    st.error("Could not generate rules. Try lowering support/confidence.")
    st.stop()

if rules.empty:
    st.warning("No association rules found. Try reducing thresholds.")
    st.stop()

rules = rules.sort_values("lift", ascending=False)

st.dataframe(
    rules[[
        "antecedents",
        "consequents",
        "support",
        "confidence",
        "lift"
    ]].head(20),
    use_container_width=True
)

# ==================================================
# VISUAL
# ==================================================

st.subheader("📊 Relationship Chart")

st.scatter_chart(
    rules[["confidence", "lift"]].head(200)
)

# ==================================================
# INSIGHT
# ==================================================

top_rule = rules.iloc[0]

st.success(f"""
Top Insight:

If customers buy:
{top_rule['antecedents']}

They also buy:
{top_rule['consequents']}

Confidence: {top_rule['confidence']:.2f}
Lift: {top_rule['lift']:.2f}
""")

# ==================================================
# DATA PREVIEW
# ==================================================

with st.expander("View Data"):
    st.dataframe(df.head(100), use_container_width=True)
