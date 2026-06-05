import pandas as pd


# =====================================================
# PRODUCT RECOMMENDATIONS
# =====================================================

def get_product_recommendations(
    rules_df,
    product_name,
    top_n=10
):
    """
    Get products frequently bought together
    """

    if rules_df.empty:
        return pd.DataFrame()

    recommendations = rules_df[

        rules_df["antecedents"]

        .astype(str)

        .str.contains(
            product_name,
            case=False,
            na=False
        )

    ]

    recommendations = recommendations.sort_values(
        by=["lift", "confidence"],
        ascending=False
    )

    return recommendations.head(top_n)


# =====================================================
# TOP PRODUCT BUNDLES
# =====================================================

def get_top_bundles(
    rules_df,
    top_n=20
):
    """
    Highest lift product bundles
    """

    if rules_df.empty:
        return pd.DataFrame()

    bundles = rules_df.sort_values(
        by="lift",
        ascending=False
    )

    return bundles.head(top_n)


# =====================================================
# CROSS SELL RECOMMENDATIONS
# =====================================================

def get_cross_sell_opportunities(
    rules_df,
    top_n=15
):
    """
    Highest confidence rules
    """

    if rules_df.empty:
        return pd.DataFrame()

    cross_sell = rules_df.sort_values(
        by="confidence",
        ascending=False
    )

    return cross_sell.head(top_n)


# =====================================================
# UPSELL PRODUCTS
# =====================================================

def get_upsell_products(
    df,
    top_n=15
):
    """
    Premium products based on average price
    """

    if (
        "Itemname" not in df.columns or
        "Price" not in df.columns
    ):
        return pd.DataFrame()

    upsell_df = (

        df.groupby("Itemname")["Price"]

        .mean()

        .reset_index()

        .sort_values(
            "Price",
            ascending=False
        )

    )

    return upsell_df.head(top_n)


# =====================================================
# CUSTOMER SEGMENT RECOMMENDATIONS
# =====================================================

def get_segment_recommendation(
    segment_name
):
    """
    Marketing actions for customer segments
    """

    recommendations = {

        "Champions":
        """
        Reward customers with VIP programs,
        exclusive offers and early access.
        """,

        "Loyal Customers":
        """
        Upsell premium products and
        bundle offers.
        """,

        "Potential Customers":
        """
        Encourage repeat purchases through
        personalized campaigns.
        """,

        "At Risk":
        """
        Run retention campaigns,
        discounts and reminders.
        """,

        "Lost Customers":
        """
        Launch win-back campaigns and
        special incentives.
        """,

        "Big Spenders":
        """
        Promote premium products,
        subscriptions and memberships.
        """,

        "Frequent Buyers":
        """
        Offer loyalty rewards and
        recurring purchase incentives.
        """,

        "Occasional Buyers":
        """
        Increase engagement through
        targeted offers.
        """,

        "New Customers":
        """
        Introduce onboarding promotions
        and welcome discounts.
        """,

        "Dormant Customers":
        """
        Send reactivation campaigns and
        personalized recommendations.
        """

    }

    return recommendations.get(
        segment_name,
        "Create personalized campaigns."
    )


# =====================================================
# TOP REVENUE PRODUCTS
# =====================================================

def get_top_revenue_products(
    df,
    top_n=10
):
    """
    Products generating highest revenue
    """

    if (
        "Itemname" not in df.columns or
        "Revenue" not in df.columns
    ):
        return pd.DataFrame()

    revenue_df = (

        df.groupby("Itemname")

        .agg({

            "Revenue": "sum"

        })

        .reset_index()

        .sort_values(
            "Revenue",
            ascending=False
        )

    )

    return revenue_df.head(top_n)


# =====================================================
# ASSOCIATION MATRIX
# =====================================================

def create_association_matrix(
    rules_df
):
    """
    Create lift matrix
    """

    if rules_df.empty:
        return pd.DataFrame()

    matrix = rules_df.pivot_table(

        index="antecedents",

        columns="consequents",

        values="lift",

        fill_value=0

    )

    return matrix


# =====================================================
# CUSTOMER SPECIFIC RECOMMENDATIONS
# =====================================================

def recommend_for_customer(
    customer_segment,
    rules_df=None
):
    """
    Personalized recommendation
    """

    recommendation = {

        "Champions":
        [
            "Premium Products",
            "VIP Membership",
            "Exclusive Bundles"
        ],

        "Loyal Customers":
        [
            "Bundle Offers",
            "Cross-Sell Products",
            "Reward Points"
        ],

        "Potential Customers":
        [
            "Discount Coupons",
            "Popular Products",
            "Seasonal Offers"
        ],

        "At Risk":
        [
            "Retention Discounts",
            "Special Promotions"
        ],

        "Lost Customers":
        [
            "Win-Back Campaign",
            "Limited-Time Offer"
        ]

    }

    return recommendation.get(
        customer_segment,
        ["Recommended Products"]
    )


# =====================================================
# BUSINESS INSIGHTS
# =====================================================

def generate_business_insights(
    rules_df,
    top_n=5
):
    """
    Generate actionable insights
    """

    insights = []

    if rules_df.empty:
        return insights

    top_rules = rules_df.sort_values(
        "lift",
        ascending=False
    ).head(top_n)

    for _, row in top_rules.iterrows():

        insight = (
            f"Customers purchasing "
            f"{row['antecedents']} "
            f"are highly likely to purchase "
            f"{row['consequents']} "
            f"(Lift={row['lift']:.2f})"
        )

        insights.append(insight)

    return insights


# =====================================================
# EXPORT RECOMMENDATIONS
# =====================================================

def export_recommendations(
    recommendations_df,
    filepath
):
    """
    Save recommendations to csv
    """

    try:

        recommendations_df.to_csv(
            filepath,
            index=False
        )

        return True

    except Exception:

        return False
