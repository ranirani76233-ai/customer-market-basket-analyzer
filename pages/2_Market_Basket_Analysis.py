import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Market Basket Analysis",
    page_icon="🛒",
    layout="wide"
)

st.title("🛒 Market Basket Analysis")

st.markdown("""
Discover product associations and customer buying patterns using
the Apriori Algorithm.
""")

# --------------------------------------------------
# FILE UPLOAD
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload Transaction Dataset",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.info(
        "Dataset should contain at least two columns:\n"
        "- TransactionID\n"
        "- Product"
    )

    transaction_col = st.selectbox(
        "Select Transaction ID Column",
        df.columns
    )

    product_col = st.selectbox(
        "Select Product Column",
        df.columns
    )

    min_support = st.slider(
        "Minimum Support",
        min_value=0.01,
        max_value=1.0,
        value=0.05,
        step=0.01
    )

    min_confidence = st.slider(
        "Minimum Confidence",
        min_value=0.01,
        max_value=1.0,
        value=0.30,
        step=0.01
    )

    if st.button("Generate Market Basket Analysis"):

        try:

            # --------------------------------------
            # CREATE BASKET MATRIX
            # --------------------------------------

            basket = (
                df.groupby(
                    [transaction_col, product_col]
                )[product_col]
                .count()
                .unstack()
                .fillna(0)
            )

            basket = basket.applymap(
                lambda x: 1 if x > 0 else 0
            )

            # --------------------------------------
            # APRIORI
            # --------------------------------------

            frequent_itemsets = apriori(
                basket,
                min_support=min_support,
                use_colnames=True
            )

            st.subheader("📦 Frequent Itemsets")

            st.dataframe(
                frequent_itemsets.sort_values(
                    "support",
                    ascending=False
                ),
                use_container_width=True
            )

            # --------------------------------------
            # ASSOCIATION RULES
            # --------------------------------------

            rules = association_rules(
                frequent_itemsets,
                metric="confidence",
                min_threshold=min_confidence
            )

            if len(rules) > 0:

                rules = rules[
                    [
                        "antecedents",
                        "consequents",
                        "support",
                        "confidence",
                        "lift"
                    ]
                ]

                rules["antecedents"] = rules[
                    "antecedents"
                ].apply(
                    lambda x: ", ".join(list(x))
                )

                rules["consequents"] = rules[
                    "consequents"
                ].apply(
                    lambda x: ", ".join(list(x))
                )

                st.subheader("🔗 Association Rules")

                st.dataframe(
                    rules.sort_values(
                        "lift",
                        ascending=False
                    ),
                    use_container_width=True
                )

                # ----------------------------------
                # TOP RECOMMENDATIONS
                # ----------------------------------

                st.subheader("🎯 Top Product Recommendations")

                top_rules = rules.sort_values(
                    "lift",
                    ascending=False
                ).head(10)

                st.dataframe(
                    top_rules,
                    use_container_width=True
                )

                st.subheader("📈 Lift Comparison")

                chart_data = top_rules[
                    ["antecedents", "lift"]
                ].set_index(
                    "antecedents"
                )

                st.bar_chart(chart_data)

            else:

                st.warning(
                    "No association rules found. "
                    "Try lowering support or confidence."
                )

        except Exception as e:

            st.error(
                f"Error: {str(e)}"
            )

else:

    st.info(
        "Upload a transaction dataset to begin analysis."
    )

# --------------------------------------------------
# SIDEBAR INFO
# --------------------------------------------------

st.sidebar.header("📚 Metrics Guide")

st.sidebar.markdown("""
**Support**
- Frequency of itemset occurrence.

**Confidence**
- Probability of buying B when A is purchased.

**Lift**
- Strength of association.
- Lift > 1 indicates a positive relationship.
""")
