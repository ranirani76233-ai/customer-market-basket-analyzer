st.subheader("📦 Frequent Itemsets")

from mlxtend.frequent_patterns import apriori, association_rules

# safe apriori
frequent_itemsets = apriori(
    basket,
    min_support=min_support,
    use_colnames=True
)

# ==================================================
# FIX 1: CHECK EMPTY ITEMSETS
# ==================================================

if frequent_itemsets is None or frequent_itemsets.empty:
    st.error("No frequent itemsets found. Try lowering Min Support.")
    st.stop()

# ==================================================
# SHOW ITEMSETS
# ==================================================

st.dataframe(
    frequent_itemsets.sort_values("support", ascending=False).head(20),
    use_container_width=True
)

# ==================================================
# ASSOCIATION RULES (SAFE FIX 🔥)
# ==================================================

st.subheader("🔗 Association Rules")

try:

    rules = association_rules(
        frequent_itemsets,
        metric="confidence",
        min_threshold=min_confidence
    )

except KeyError as e:

    st.error("Not enough item combinations to generate rules.")
    st.stop()

# ==================================================
# FIX 2: CHECK EMPTY RULES
# ==================================================

if rules is None or rules.empty:
    st.warning("No association rules found. Lower support/confidence.")
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
