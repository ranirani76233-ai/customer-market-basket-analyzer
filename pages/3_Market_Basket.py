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

    except Exception as e:

        st.error(f"Dataset Error: {e}")
        return pd.DataFrame()

    return df


df = load_data()

if df.empty:
    st.stop()

# ==================================================
# CLEAN COLUMN NAMES
# ==================================================

df.columns = df.columns.str.strip()

# ==================================================
# AUTO DETECT COLUMNS
# ==================================================

invoice_col = None
product_col = None

for col in df.columns:

    c = col.lower()

    if c in [
        "billno",
        "invoice",
        "invoiceno",
        "invoice_no"
    ]:
        invoice_col = col

    if c in [
        "itemname",
        "description",
        "product",
        "productname"
    ]:
        product_col = col

if invoice_col is None or product_col is None:

    st.error(
        f"""
        Required columns not found.

        Invoice Column: {invoice_col}
        Product Column: {product_col}

        Available Columns:
        {list(df.columns)}
        """
    )

    st.stop()

# ==================================================
# CREATE BASKET
# ==================================================

basket = (

    df.groupby(
        [invoice_col, product_col]
    )

    .size()

    .unstack()

    .fillna(0)

)

basket = basket.applymap(
    lambda x: 1 if x > 0 else 0
)

# ==================================================
# SETTINGS
# ==================================================

st.sidebar.header("Analysis Settings")

min_support = st.sidebar.slider(
    "Minimum Support",
    0.01,
    0.20,
    0.02,
    0.01
)

min_confidence = st.sidebar.slider(
    "Minimum Confidence",
    0.10,
    1.00,
    0.30,
    0.05
)

# ==================================================
# FREQUENT ITEMSETS
# ==================================================

st.subheader("📦 Frequent Itemsets")

frequent_items = apriori(
    basket,
    min_support=min_support,
    use_colnames=True
)

if frequent_items.empty:

    st.warning(
        "No frequent itemsets found."
    )

    st.stop()

frequent_items["Itemsets"] = (
    frequent_items["itemsets"]
    .astype(str)
)

st.dataframe(
    frequent_items.sort_values(
        "support",
        ascending=False
    ).head(20),
    use_container_width=True
)

# ==================================================
# TOP ITEMSETS CHART
# ==================================================

top_itemsets = (

    frequent_items

    .sort_values(
        "support",
        ascending=False
    )

    .head(10)

)

fig = px.bar(

    top_itemsets,

    x="support",

    y="Itemsets",

    orientation="h",

    title="Top Frequent Itemsets"

)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ==================================================
# ASSOCIATION RULES
# ==================================================

st.subheader("🔗 Association Rules")

rules = association_rules(
    frequent_items,
    metric="confidence",
    min_threshold=min_confidence
)

if rules.empty:

    st.warning(
        "No association rules found."
    )

else:

    rules["Antecedents"] = (
        rules["antecedents"]
        .astype(str)
    )

    rules["Consequents"] = (
        rules["consequents"]
        .astype(str)
    )

    display_rules = rules[[
        "Antecedents",
        "Consequents",
        "support",
        "confidence",
        "lift"
    ]]

    st.dataframe(
        display_rules.sort_values(
            "lift",
            ascending=False
        ),
        use_container_width=True
    )

# ==================================================
# LIFT VS CONFIDENCE
# ==================================================

if not rules.empty:

    st.subheader("📊 Lift vs Confidence")

    fig = px.scatter(

        rules,

        x="confidence",

        y="lift",

        size="support",

        hover_data=[
            "Antecedents",
            "Consequents"
        ]

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ==================================================
# PRODUCT RECOMMENDATIONS
# ==================================================

if not rules.empty:

    st.subheader("🎯 Product Recommendations")

    top_rules = (

        rules

        .sort_values(
            "lift",
            ascending=False
        )

        .head(10)

    )

    for _, row in top_rules.iterrows():

        st.success(

            f"""
            Customers buying
            {row['Antecedents']}
            also buy
            {row['Consequents']}

            Confidence:
            {row['confidence']:.2f}

            Lift:
            {row['lift']:.2f}
            """

        )

# ==================================================
# BUSINESS INSIGHTS
# ==================================================

st.subheader("💡 Market Basket Insights")

st.info(
    f"""
    Total Transactions: {basket.shape[0]}

    Total Products: {basket.shape[1]}

    Frequent Itemsets Found:
    {len(frequent_items)}

    Association Rules Found:
    {len(rules)}
    """
)

# ==================================================
# DATA PREVIEW
# ==================================================

with st.expander("View Source Data"):

    st.dataframe(
        df.head(100),
        use_container_width=True
    )
