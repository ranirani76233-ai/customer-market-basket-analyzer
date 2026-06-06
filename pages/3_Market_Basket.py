import streamlit as st
import pandas as pd
import numpy as np

from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules

# ==================================================
# CONFIG
# ==================================================

st.set_page_config(
    page_title="Market Basket Analysis",
    page_icon="🛒",
    layout="wide"
)

st.title("🛒 Market Basket Analysis (Robust Version)")

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
# CLEAN
# ==================================================

df.columns = df.columns.str.strip()

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

df = df[[invoice_col, product_col]].dropna()
df = df.drop_duplicates()

# ==================================================
# LIMIT DATA (CRITICAL FOR STABILITY)
# ==================================================

if len(df) > 25000:
    df = df.sample(25000, random_state=42)

# ==================================================
# KEEP TOP PRODUCTS
# ==================================================

top_products = df[product_col].value_counts().head(40).index
df = df[df[product_col].isin(top_products)]

# ==================================================
# BASKET CREATION
# ==================================================

basket = (
    df.groupby([invoice_col, product_col])
    .size()
    .unstack(fill_value=0)
)

basket = basket.apply(lambda x: (x > 0).astype(int))

if basket.shape[1] < 2:
    st.error("Not enough product variety for analysis")
    st.stop()

# ==================================================
# SIDEBAR
# ==================================================

st.sidebar.header("Settings")

min_support = st.sidebar.slider("Min Support", 0.001, 0.05, 0.005)
min_confidence = st.sidebar.slider("Min Confidence", 0.1, 1.0, 0.2)

# ==================================================
# FREQUENT ITEMSETS (APRIORI + FALLBACK)
# ==================================================

st.subheader("📦 Frequent Itemsets")

frequent_itemsets = apriori(
    basket,
    min_support=min_support,
    use_colnames=True
)

# fallback if apriori fails
if frequent_itemsets.empty:
    st.warning("Apriori failed → switching to FP-Growth")
    frequent_itemsets = fpgrowth(
        basket,
        min_support=min_support,
        use_colnames=True
    )

if frequent_itemsets.empty:
    st.error("No patterns found even with FP-Growth. Data too sparse.")
    st.stop()

st.dataframe(
    frequent_itemsets.sort_values("support", ascending=False).head(20),
    use_container_width=True
)

# ==================================================
# RULES GENERATION (ROBUST FIX)
# ==================================================

st.subheader("🔗 Association Rules")

rules = association_rules(
    frequent_itemsets,
    metric="lift",
    min_threshold=1
)

# fallback relaxation
if rules.empty:

    rules = association_rules(
        frequent_itemsets,
        metric="confidence",
        min_threshold=0.1
    )

# final fallback
if rules.empty:

    st.warning("Weak dataset → generating minimal rules")

    rules = association_rules(
        frequent_itemsets,
        metric="support",
        min_threshold=0.001
    )

# ==================================================
# FINAL CHECK (NEVER FAILS NOW)
# ==================================================

if rules.empty:
    st.error("No rules can be generated from this dataset.")
    st.stop()

rules = rules.sort_values("lift", ascending=False)

# ==================================================
# DISPLAY RULES
# ==================================================

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

st.subheader("📊 Insights")

st.scatter_chart(
    rules[["confidence", "lift"]].head(200)
)

# ==================================================
# TOP INSIGHT
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
# DATA VIEW
# ==================================================

with st.expander("View Data"):
    st.dataframe(df.head(100), use_container_width=True)
