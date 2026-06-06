import streamlit as st
import pandas as pd
import plotly.express as px

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Recommendations",
    page_icon="💡",
    layout="wide"
)

st.title("💡 Smart Recommendation Engine")

# =====================================================
# LOAD DATA
# =====================================================

@st.cache_data
def load_data():

    df = pd.read_csv(
        'data/Assignment-1_Data.csv'
       
    )

    return df

# =====================================================
# LOAD ASSOCIATION RULES
# =====================================================

@st.cache_data
def load_rules():

    try:

        rules = pd.read_csv(
            "reports/association_rules.csv"
        )

        return rules

    except:

        return pd.DataFrame()

# =====================================================
# LOAD CUSTOMER SEGMENTS
# =====================================================

@st.cache_data
def load_segments():

    try:

        segments = pd.read_csv(
            "reports/customer_segments.csv"
        )

        return segments

    except:

        return pd.DataFrame()

# =====================================================
# DATA
# =====================================================

df = load_data()
rules = load_rules()
segments = load_segments()

# =====================================================
# PRODUCT RECOMMENDATIONS
# =====================================================

st.header("🛒 Product Recommendations")

if not rules.empty:

    products = sorted(
        df["Itemname"].dropna().unique()
    )

    selected_product = st.selectbox(
        "Select Product",
        products
    )

    recommendations = rules[

        rules["antecedents"]
        .astype(str)
        .str.contains(
            selected_product,
            case=False,
            na=False
        )

    ]

    if not recommendations.empty:

        st.success(
            f"Recommended Products for: {selected_product}"
        )

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

        st.warning(
            "No recommendations found."
        )

else:

    st.warning(
        "Association rules file not found."
    )

# =====================================================
# TOP PRODUCT BUNDLES
# =====================================================

st.header("🎁 Top Product Bundles")

if not rules.empty:

    top_bundles = (

        rules

        .sort_values(
            "lift",
            ascending=False
        )

        .head(20)

    )

    fig = px.bar(

        top_bundles,

        x="lift",

        y="antecedents",

        color="confidence",

        orientation="h",

        title="Top Bundles by Lift"

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# =====================================================
# CROSS SELL OPPORTUNITIES
# =====================================================

st.header("🔗 Cross-Selling Opportunities")

if not rules.empty:

    cross_sell = (

        rules

        .sort_values(
            "confidence",
            ascending=False
        )

        .head(15)

    )

    st.dataframe(

        cross_sell[
            [
                "antecedents",
                "consequents",
                "confidence",
                "lift"
            ]
        ],

        use_container_width=True

    )

# =====================================================
# UPSELL OPPORTUNITIES
# =====================================================

st.header("📈 Upselling Opportunities")

if (
    "Itemname" in df.columns and
    "Price" in df.columns
):

    upsell_df = (

        df.groupby("Itemname")["Price"]

        .mean()

        .reset_index()

        .sort_values(
            "Price",
            ascending=False
        )

        .head(15)

    )

    fig = px.bar(

        upsell_df,

        x="Price",

        y="Itemname",

        orientation="h",

        title="Premium Products"

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# =====================================================
# CUSTOMER SEGMENT RECOMMENDATIONS
# =====================================================

st.header("🎯 Customer Segment Recommendations")

if not segments.empty:

    segment_list = sorted(
        segments["Segment_Name"].unique()
    )

    selected_segment = st.selectbox(

        "Select Customer Segment",

        segment_list

    )

    filtered_segment = segments[
        segments["Segment_Name"]
        == selected_segment
    ]

    st.dataframe(
        filtered_segment.head(50),
        use_container_width=True
    )

    recommendation_map = {

        "Champions":
        "Offer VIP rewards and loyalty programs.",

        "Loyal Customers":
        "Upsell premium products and bundles.",

        "Potential Customers":
        "Provide targeted promotions.",

        "At Risk":
        "Run retention campaigns.",

        "Lost Customers":
        "Send win-back offers."
    }

    recommendation = recommendation_map.get(

        selected_segment,

        "Create personalized campaigns."

    )

    st.success(recommendation)

else:

    st.warning(
        "Customer segmentation file not found."
    )

# =====================================================
# REVENUE OPPORTUNITIES
# =====================================================

st.header("💰 Revenue Opportunities")

if (
    "Qty" in df.columns and
    "Price" in df.columns
):

    df["Revenue"] = (
        df["Qty"] * df["Price"]
    )

    revenue_products = (

        df.groupby("Itemname")

        .agg({

            "Revenue":"sum"

        })

        .reset_index()

        .sort_values(
            "Revenue",
            ascending=False
        )

        .head(15)

    )

    fig = px.pie(

        revenue_products,

        names="Itemname",

        values="Revenue",

        hole=0.4,

        title="Revenue Contribution"

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# =====================================================
# ACTIONABLE INSIGHTS
# =====================================================

st.header("📋 Actionable Business Insights")

st.info(
    """
    1. Bundle products with high lift values.

    2. Promote frequently purchased combinations.

    3. Focus on high-value customer segments.

    4. Re-engage at-risk customers.

    5. Increase visibility of premium products.

    6. Use personalized recommendations in marketing.

    7. Optimize inventory for frequently purchased bundles.

    8. Launch loyalty programs for repeat buyers.
    """
)

# =====================================================
# EXPORT RECOMMENDATIONS
# =====================================================

st.header("📥 Export Recommendations")

if not rules.empty:

    csv = rules.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(

        "Download Recommendations",

        data=csv,

        file_name="recommendations.csv",

        mime="text/csv"

    )

# =====================================================
# SUMMARY DASHBOARD
# =====================================================

st.header("📊 Recommendation Summary")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Products",
    df["Itemname"].nunique()
    if "Itemname" in df.columns
    else 0
)

col2.metric(
    "Association Rules",
    len(rules)
)

col3.metric(
    "Customer Segments",
    segments["Segment_Name"].nunique()
    if not segments.empty
    else 0
)

st.success(
    "Recommendation engine successfully generated insights and opportunities."
)
