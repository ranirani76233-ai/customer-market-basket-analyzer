"""
market_basket.py

Market Basket Analysis Module

Features:
---------
1. Basket Creation
2. Apriori Algorithm
3. Frequent Itemsets
4. Association Rules
5. Product Recommendations
6. Product Association Matrix
"""

import pandas as pd

from mlxtend.frequent_patterns import (
    apriori,
    association_rules
)


# =====================================================
# CREATE BASKET MATRIX
# =====================================================

def create_basket(df):
    """
    Creates one-hot encoded transaction basket
    """

    basket = (
        df.groupby(
            ["BillNo", "Itemname"]
        )["Quantity"]
        .sum()
        .unstack()
        .fillna(0)
    )

    basket = basket.apply(
        lambda x: x > 0
    )

    return basket


# =====================================================
# FREQUENT ITEMSETS
# =====================================================

def generate_frequent_itemsets(
    basket,
    min_support=0.02
):
    """
    Run Apriori Algorithm
    """

    frequent_itemsets = apriori(
        basket,
        min_support=min_support,
        use_colnames=True
    )

    frequent_itemsets = (
        frequent_itemsets
        .sort_values(
            by="support",
            ascending=False
        )
    )

    return frequent_itemsets


# =====================================================
# ASSOCIATION RULES
# =====================================================

def generate_association_rules(
    frequent_itemsets,
    metric="lift",
    min_threshold=1
):
    """
    Generate Association Rules
    """

    rules = association_rules(
        frequent_itemsets,
        metric=metric,
        min_threshold=min_threshold
    )

    rules = rules.sort_values(
        by="lift",
        ascending=False
    )

    return rules


# =====================================================
# CLEAN RULES FOR DISPLAY
# =====================================================

def prepare_rules_display(rules):

    display_rules = rules.copy()

    display_rules["Antecedent"] = (
        display_rules["antecedents"]
        .apply(
            lambda x: ", ".join(list(x))
        )
    )

    display_rules["Consequent"] = (
        display_rules["consequents"]
        .apply(
            lambda x: ", ".join(list(x))
        )
    )

    return display_rules[
        [
            "Antecedent",
            "Consequent",
            "support",
            "confidence",
            "lift"
        ]
    ]


# =====================================================
# PRODUCT RECOMMENDATION
# =====================================================

def get_product_recommendations(
    rules,
    product_name
):
    """
    Return products frequently purchased
    with selected product
    """

    recommendations = []

    for _, row in rules.iterrows():

        antecedent = list(
            row["antecedents"]
        )

        if product_name in antecedent:

            recommendations.append({

                "Selected_Product":
                product_name,

                "Recommended_Product":
                ", ".join(
                    list(
                        row["consequents"]
                    )
                ),

                "Confidence":
                round(
                    row["confidence"],
                    3
                ),

                "Lift":
                round(
                    row["lift"],
                    3
                )
            })

    recommendations = pd.DataFrame(
        recommendations
    )

    if not recommendations.empty:

        recommendations = (
            recommendations
            .sort_values(
                by="Confidence",
                ascending=False
            )
        )

    return recommendations


# =====================================================
# ASSOCIATION MATRIX
# =====================================================

def create_association_matrix(rules):
    """
    Creates matrix table for heatmap
    """

    matrix_data = []

    for _, row in rules.iterrows():

        antecedent = ", ".join(
            list(row["antecedents"])
        )

        consequent = ", ".join(
            list(row["consequents"])
        )

        matrix_data.append({

            "Antecedent":
            antecedent,

            "Consequent":
            consequent,

            "Confidence":
            row["confidence"]
        })

    matrix_df = pd.DataFrame(
        matrix_data
    )

    matrix = matrix_df.pivot_table(

        index="Antecedent",

        columns="Consequent",

        values="Confidence",

        fill_value=0
    )

    return matrix


# =====================================================
# PRODUCT LIST
# =====================================================

def get_product_list(df):

    products = sorted(
        df["Itemname"]
        .dropna()
        .unique()
        .tolist()
    )

    return products


# =====================================================
# MARKET BASKET SUMMARY
# =====================================================

def market_basket_summary(
    basket,
    frequent_itemsets,
    rules
):

    summary = {

        "Transactions":
        basket.shape[0],

        "Products":
        basket.shape[1],

        "Frequent_Itemsets":
        len(frequent_itemsets),

        "Association_Rules":
        len(rules)
    }

    return summary


# =====================================================
# COMPLETE PIPELINE
# =====================================================

def run_market_basket_analysis(
    df,
    min_support=0.02,
    min_lift=1
):
    """
    Complete Market Basket Pipeline
    """

    basket = create_basket(df)

    frequent_itemsets = (
        generate_frequent_itemsets(
            basket,
            min_support=min_support
        )
    )

    rules = (
        generate_association_rules(
            frequent_itemsets,
            metric="lift",
            min_threshold=min_lift
        )
    )

    return (
        basket,
        frequent_itemsets,
        rules
    )


# =====================================================
# EXPORT RULES
# =====================================================

def export_rules(
    rules,
    output_path
):
    """
    Save association rules to CSV
    """

    rules.to_csv(
        output_path,
        index=False
    )


# =====================================================
# MAIN TEST
# =====================================================

if __name__ == "__main__":

    from src.data_preprocessing import (
        load_data,
        preprocess_data
    )

    FILE_PATH = (
        "data/raw/Assignment-1_Data.csv"
    )

    df = load_data(FILE_PATH)

    df = preprocess_data(df)

    basket, frequent_itemsets, rules = (
        run_market_basket_analysis(df)
    )

    print("\nBasket Shape")
    print(basket.shape)

    print("\nTop Frequent Itemsets")
    print(
        frequent_itemsets.head()
    )

    print("\nTop Rules")
    print(
        prepare_rules_display(
            rules
        ).head()
    )

    print(
        market_basket_summary(
            basket,
            frequent_itemsets,
            rules
        )
    )
