import streamlit as st
import pandas as pd
import plotly.express as px

from mlxtend.frequent_patterns import (
    apriori,
    association_rules
)

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Market Basket Analysis",
    page_icon="🔗",
    layout="wide"
)

st.title("🔗 Market Basket Analysis")

# =====================================================
# LOAD DATA
# =====================================================

@st.cache_data
def load_data():

    df = pd.read_csv(
        'data/Assignment-1_Data.csv'
     
    )

    return df


df = load_data()

# =====================================================
# COLUMN CHECK
# =====================================================

required_columns = [
    "BillNo",
    "Itemname"
]

missing = [
    col for col in required_columns
    if col not in df.columns
]

if missing:

    st.error(
        f"Missing columns: {missing}"
    )

    st.stop()

# =====================================================
# SIDEBAR FILTERS
# =====================================================

st.sidebar.header("MBA Parameters")

min_support = st.sidebar.slider(
    "Minimum Support",
    0.001,
    0.10,
    0.01,
    0.001
)

min_confidence = st.sidebar.slider(
    "Minimum Confidence",
    0.1,
    1.0,
    0.30,
    0.05
)

min_lift = st.sidebar.slider(
    "Minimum Lift",
    1.0,
    10.0,
    1.0,
    0.1
)

# =====================================================
# CREATE BASKET
# =====================================================

basket = (

    df.groupby(
        ["BillNo", "Itemname"]
    )["Itemname"]

    .count()

    .unstack()

    .fillna(0)

)

basket = basket.applymap(
    lambda x: 1 if x > 0 else 0
)

# =====================================================
# APRIORI
# =====================================================

with st.spinner(
    "Running Apriori..."
):

    frequent_itemsets = apriori(

        basket,

        min_support=min_support,

        use_colnames=True

    )

# =====================================================
# ASSOCIATION RULES
# =====================================================

rules = association_rules(

    frequent_itemsets,

    metric="confidence",

    min_threshold=min_confidence

)

rules = rules[
    rules["lift"] >= min_lift
]

# =====================================================
# SUMMARY KPIs
# =====================================================

st.subheader("📊 Market Basket Summary")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Transactions",
    f"{basket.shape[0]:,}"
)

col2.metric(
    "Frequent Itemsets",
    f"{len(frequent_itemsets):,}"
)

col3.metric(
    "Association Rules",
    f"{len(rules):,}"
)

# =====================================================
# FREQUENT ITEMSETS
# =====================================================

st.subheader("🛒 Frequent Itemsets")

itemsets_display = frequent_itemsets.copy()

itemsets_display["itemsets"] = (

    itemsets_display["itemsets"]

    .apply(
        lambda x: ", ".join(list(x))
    )

)

st.dataframe(
    itemsets_display,
    use_container_width=True
)

# =====================================================
# ASSOCIATION RULES TABLE
# =====================================================

st.subheader("🔗 Association Rules")

if not rules.empty:

    rules_display = rules.copy()

    rules_display["antecedents"] = (

        rules_display["antecedents"]

        .apply(
            lambda x: ", ".join(list(x))
        )

    )

    rules_display["consequents"] = (

        rules_display["consequents"]

        .apply(
            lambda x: ", ".join(list(x))
        )

    )

    display_columns = [

        "antecedents",

        "consequents",

        "support",

        "confidence",

        "lift"

    ]

    st.dataframe(

        rules_display[
            display_columns
        ],

        use_container_width=True

    )

else:

    st.warning(
        "No rules found."
    )

# =====================================================
# PRODUCT SEARCH
# =====================================================

st.subheader("🔍 Product Recommendation Search")

products = sorted(
    df["Itemname"].unique()
)

selected_product = st.selectbox(

    "Select Product",

    products

)

if selected_product:

    recommendations = rules_display[

        rules_display[
            "antecedents"
        ].str.contains(
            selected_product,
            case=False,
            na=False
        )

    ]

    st.markdown(
        f"### Products Bought With: {selected_product}"
    )

    if not recommendations.empty:

        st.dataframe(

            recommendations[

                [
                    "antecedents",
                    "consequents",
                    "confidence",
                    "lift"
                ]

            ],

            use_container_width=True

        )

    else:

        st.info(
            "No recommendations found."
        )

# =====================================================
# ASSOCIATION MATRIX
# =====================================================

st.subheader(
    "📋 Product Association Matrix"
)

if not rules.empty:

    matrix_df = rules_display[

        [
            "antecedents",
            "consequents",
            "lift"
        ]

    ]

    matrix = matrix_df.pivot_table(

        index="antecedents",

        columns="consequents",

        values="lift",

        fill_value=0

    )

    st.dataframe(
        matrix,
        use_container_width=True
    )

# =====================================================
# HEATMAP
# =====================================================

st.subheader("🔥 Association Heatmap")

if not rules.empty:

    fig = px.imshow(

        matrix,

        aspect="auto",

        labels=dict(
            color="Lift"
        )

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

# =====================================================
# TOP RULES
# =====================================================

st.subheader(
    "🏆 Strongest Product Associations"
)

if not rules.empty:

    top_rules = (

        rules_display

        .sort_values(
            "lift",
            ascending=False
        )

        .head(20)

    )

    fig = px.bar(

        top_rules,

        x="lift",

        y="antecedents",

        color="confidence",

        orientation="h",

        title="Top Product Associations"

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# =====================================================
# DOWNLOAD RULES
# =====================================================

st.subheader("📥 Download Results")

csv = rules_display.to_csv(
    index=False
).encode("utf-8")

st.download_button(

    label="Download Association Rules",

    data=csv,

    file_name="association_rules.csv",

    mime="text/csv"

)

# =====================================================
# BUSINESS INSIGHTS
# =====================================================

st.subheader("💡 Business Insights")

st.success(
    """
    • Bundle products with high lift values.

    • Cross-sell products frequently purchased together.

    • Place associated products nearby in stores.

    • Use recommendations in e-commerce checkout.

    • Create combo offers using high-confidence rules.
    """
)
