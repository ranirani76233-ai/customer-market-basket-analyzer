import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
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
        st.error(f"Error loading dataset: {e}")
        return pd.DataFrame()


df = load_data()

if df.empty:
    st.error("Dataset is empty or failed to load.")
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

for col in df.columns:

    c = col.lower()

    if c in ["billno", "invoice", "invoiceno", "invoiceno"]:
        invoice_col = col

    if c in ["itemname", "product", "description", "stockcode"]:
        product_col = col

# ==================================================
# VALIDATION
# ==================================================

if invoice_col is None or product_col is None:

    st.error(
        f"""
        Required columns not found!

        Invoice Column: {invoice_col}
        Product Column: {product_col}

        Available Columns:
        {df.columns.tolist()}
        """
    )
    st.stop()

# ==================================================
# CLEAN DATA
# ==================================================

df = df[[invoice_col, product_col]].dropna()

# remove duplicates
df = df.drop_duplicates()

# ==================================================
# CREATE BASKET MATRIX
# ==================================================

basket = (
    df.groupby([invoice_col, product_col])
    .size()
    .unstack(fill_value=0)
)

basket = basket.applymap(lambda x: 1 if x > 0 else 0)

# ==================================================
# SIDEBAR SETTINGS
# ==================================================

st.sidebar.header("Settings")

min_support = st.sidebar.slider(
    "Min Support",
    0.01, 0.2, 0.02
)

min_confidence = st.sidebar.slider(
    "Min Confidence",
    0.1, 1.0, 0.3
)

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
    st.warning("No frequent itemsets found.")
    st.stop()

frequent_itemsets["itemsets"] = frequent_itemsets["itemsets"].astype(str)

st.dataframe(
    frequent_itemsets.sort_values("support", ascending=False).head(20),
    use_container_width=True
)

# ==================================================
# TOP ITEMSETS CHART
# ==================================================

top_itemsets = frequent_itemsets.sort_values("support", ascending=False).head(10)

fig = px.bar(
    top_itemsets,
    x="support",
    y="itemsets",
    orientation="h",
    title="Top Frequent Itemsets"
)

st.plotly_chart(fig, use_container_width=True)

# ==================================================
# ASSOCIATION RULES
# ==================================================

st.subheader("🔗 Association Rules")

rules = association_rules(
    frequent_itemsets,
    metric="confidence",
    min_threshold=min_confidence
)

if rules.empty:
    st.warning("No association rules found.")
    st.stop()

rules["antecedents"] = rules["antecedents"].astype(str)
rules["consequents"] = rules["consequents"].astype(str)

st.dataframe(
    rules[["antecedents", "consequents", "support", "confidence", "lift"]]
    .sort_values("lift", ascending=False),
    use_container_width=True
)

# ==================================================
# VISUALIZATION
# ==================================================

st.subheader("📊 Lift vs Confidence")

fig = px.scatter(
    rules,
    x="confidence",
    y="lift",
    size="support",
    hover_data=["antecedents", "consequents"]
)

st.plotly_chart(fig, use_container_width=True)

# ==================================================
# INSIGHTS
# ==================================================

st.subheader("💡 Business Insights")

top_rule = rules.sort_values("lift", ascending=False).iloc[0]

st.success(
    f"""
    If customers buy:
    {top_rule['antecedents']}

    They also buy:
    {top_rule['consequents']}

    Confidence: {top_rule['confidence']:.2f}
    Lift: {top_rule['lift']:.2f}
    """
)

# ==================================================
# RAW DATA
# ==================================================

with st.expander("View Dataset"):
    st.dataframe(df.head(100), use_container_width=True)
