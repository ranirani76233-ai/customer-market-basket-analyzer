import streamlit as st
import pandas as pd
import numpy as np

from mlxtend.frequent_patterns import apriori, association_rules

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="Market Basket Analysis",
    page_icon="🛒",
    layout="wide"
)

st.title("🛒 Market Basket Analysis")

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
    st.error("Dataset is empty or not found.")
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

# ==================================================
# VALIDATION
# ==================================================

if not invoice_col or not product_col:
    st.error(f"""
    Required columns not found!

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

# LIMIT DATA (PREVENT CRASH)
if len(df) > 30000:
    df = df.sample(30000, random_state=42)

# ==================================================
# CREATE BASKET MATRIX
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
    st.error("Not enough products for Market Basket Analysis")
    st.stop()

# ==================================================
# SIDEBAR SETTINGS
# ==================================================

st.sidebar.header("Settings")

min_support = st.sidebar.slider("Min Support", 0.01, 0.2, 0.02)
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
    st.warning("No frequent itemsets found. Try lowering support.")
    st.stop()

st.dataframe(
    frequent_itemsets.sort_values("support", ascending=False).head(20),
    use_container_width=True
)

# ==================================================
# ASSOCIATION RULES (SAFE FIX)
# ==================================================

st.subheader("🔗 Association Rules")

try:
    rules = association_rules(
        frequent_itemsets,
        metric="confidence",
        min_threshold=min_confidence
    )
except Exception as e:
    st.error(f"Rules generation failed: {e}")
    st.stop()

if rules.empty:
    st.warning("No association rules found. Try lowering thresholds.")
    st.stop()

# ==================================================
# CLEAN RULES
# ==================================================

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
# VISUALIZATION
# ==================================================

st.subheader("📊 Lift vs Confidence")

st.scatter_chart(
    rules[["confidence", "lift"]].head(200)
)

# ==================================================
# INSIGHTS
# ==================================================

best_rule = rules.iloc[0]

st.success(f"""
Top Insight:

If customers buy:
{best_rule['antecedents']}

They also buy:
{best_rule['consequents']}

Confidence: {best_rule['confidence']:.2f}
Lift: {best_rule['lift']:.2f}
""")

# ==================================================
# RAW DATA
# ==================================================

with st.expander("View Dataset"):
    st.dataframe(df.head(100), use_container_width=True)
