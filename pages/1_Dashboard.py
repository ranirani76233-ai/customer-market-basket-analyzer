import streamlit as st
import pandas as pd
import plotly.express as px

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Customer Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("📊 Customer Analytics Dashboard")

st.markdown("""
Analyze customer transactions, sales performance,
and purchasing behavior.
""")

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload Customer Dataset",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.success("Dataset Loaded Successfully")

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # --------------------------------------------------
    # SIDEBAR FILTERS
    # --------------------------------------------------

    st.sidebar.header("Filters")

    numeric_columns = df.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    # --------------------------------------------------
    # KPI SECTION
    # --------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    total_records = len(df)

    total_sales = (
        df["Price"].sum()
        if "Price" in df.columns
        else 0
    )

    total_quantity = (
        df["Quantity"].sum()
        if "Quantity" in df.columns
        else 0
    )

    unique_customers = (
        df["CustomerID"].nunique()
        if "CustomerID" in df.columns
        else 0
    )

    col1.metric(
        "Total Records",
        f"{total_records:,}"
    )

    col2.metric(
        "Total Sales",
        f"₹{total_sales:,.2f}"
    )

    col3.metric(
        "Total Quantity",
        f"{total_quantity:,}"
    )

    col4.metric(
        "Unique Customers",
        f"{unique_customers:,}"
    )

    st.divider()

    # --------------------------------------------------
    # SALES ANALYSIS
    # --------------------------------------------------

    if "Price" in df.columns:

        st.subheader("💰 Sales Distribution")

        fig = px.histogram(
            df,
            x="Price",
            nbins=30,
            title="Sales Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # --------------------------------------------------
    # CUSTOMER ANALYSIS
    # --------------------------------------------------

    if (
        "CustomerID" in df.columns
        and "Price" in df.columns
    ):

        st.subheader("👥 Top Customers")

        customer_sales = (
            df.groupby("CustomerID")["Price"]
            .sum()
            .reset_index()
            .sort_values(
                "Price",
                ascending=False
            )
            .head(10)
        )

        fig = px.bar(
            customer_sales,
            x="CustomerID",
            y="Price",
            title="Top 10 Customers by Sales"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # --------------------------------------------------
    # PRODUCT ANALYSIS
    # --------------------------------------------------

    if "Itemname" in df.columns:

        st.subheader("🛍 Top Products")

        top_products = (
            df["Itemname"]
            .value_counts()
            .head(10)
            .reset_index()
        )

        top_products.columns = [
            "Product",
            "Count"
        ]

        fig = px.bar(
            top_products,
            x="Product",
            y="Count",
            title="Top 10 Products"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # --------------------------------------------------
    # NUMERIC SUMMARY
    # --------------------------------------------------

    st.subheader("📈 Statistical Summary")

    st.dataframe(
        df.describe(),
        use_container_width=True
    )

    # --------------------------------------------------
    # CORRELATION HEATMAP
    # --------------------------------------------------

    if len(numeric_columns) > 1:

        st.subheader("🔥 Correlation Analysis")

        corr = df[numeric_columns].corr()

        fig = px.imshow(
            corr,
            text_auto=True,
            aspect="auto",
            title="Correlation Heatmap"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # --------------------------------------------------
    # DOWNLOAD DATA
    # --------------------------------------------------

    csv = df.to_csv(index=False)

    st.download_button(
        label="⬇ Download Dataset",
        data=csv,
        file_name="customer_data.csv",
        mime="text/csv"
    )

else:

    st.info(
        "Please upload a CSV dataset to start analysis."
    )
