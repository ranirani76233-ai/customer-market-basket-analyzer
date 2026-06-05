import pandas as pd
import streamlit as st


# =====================================================
# LOAD DATA
# =====================================================

@st.cache_data
def load_data(
    filepath="data/Assignment-1_Data.csv"
):
    """
    Load retail dataset
    """

    try:

        df = pd.read_csv(
            filepath,
            encoding="ISO-8859-1"
        )

        return df

    except Exception as e:

        st.error(
            f"Error loading data: {e}"
        )

        return pd.DataFrame()


# =====================================================
# CLEAN DATA
# =====================================================

@st.cache_data
def clean_data(df):
    """
    Clean and preprocess dataset
    """

    df = df.copy()

    # Remove duplicates
    df = df.drop_duplicates()

    # Remove completely empty rows
    df = df.dropna(
        how="all"
    )

    # Convert Date column

    if "Date" in df.columns:

        df["Date"] = pd.to_datetime(
            df["Date"],
            errors="coerce"
        )

    # Convert numeric columns

    numeric_cols = [
        "Qty",
        "Price"
    ]

    for col in numeric_cols:

        if col in df.columns:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

    # Remove null quantity/price

    if "Qty" in df.columns:

        df = df[
            df["Qty"] > 0
        ]

    if "Price" in df.columns:

        df = df[
            df["Price"] > 0
        ]

    # Create revenue

    if (
        "Qty" in df.columns and
        "Price" in df.columns
    ):

        df["Revenue"] = (
            df["Qty"] *
            df["Price"]
        )

    return df


# =====================================================
# DATA SUMMARY
# =====================================================

@st.cache_data
def get_data_summary(df):
    """
    Dataset summary
    """

    summary = {

        "Rows":
        df.shape[0],

        "Columns":
        df.shape[1],

        "Missing Values":
        int(
            df.isnull()
            .sum()
            .sum()
        ),

        "Duplicates":
        int(
            df.duplicated()
            .sum()
        )

    }

    return summary


# =====================================================
# CUSTOMER DATASET
# =====================================================

@st.cache_data
def create_customer_dataset(df):
    """
    Customer-level dataset
    """

    if not all(
        col in df.columns
        for col in [
            "CustomerID",
            "BillNo",
            "Revenue"
        ]
    ):
        return pd.DataFrame()

    customer_df = (

        df.groupby(
            "CustomerID"
        )

        .agg({

            "Revenue":
            "sum",

            "BillNo":
            "nunique",

            "Qty":
            "sum"

        })

        .reset_index()

    )

    customer_df.columns = [

        "CustomerID",

        "TotalSpend",

        "Orders",

        "TotalItems"

    ]

    return customer_df


# =====================================================
# RFM DATASET
# =====================================================

@st.cache_data
def create_rfm(df):
    """
    Create RFM table
    """

    required_cols = [

        "CustomerID",
        "BillNo",
        "Date",
        "Revenue"

    ]

    if not all(
        col in df.columns
        for col in required_cols
    ):
        return pd.DataFrame()

    snapshot_date = (
        df["Date"].max()
    )

    rfm = (

        df.groupby(
            "CustomerID"
        )

        .agg({

            "Date":
            lambda x:
            (
                snapshot_date -
                x.max()
            ).days,

            "BillNo":
            "nunique",

            "Revenue":
            "sum"

        })

        .reset_index()

    )

    rfm.columns = [

        "CustomerID",

        "Recency",

        "Frequency",

        "Monetary"

    ]

    return rfm


# =====================================================
# MACHINE LEARNING DATASET
# =====================================================

@st.cache_data
def create_ml_dataset(df):
    """
    Dataset for ML models
    """

    required_cols = [
        "Qty",
        "Price",
        "Revenue"
    ]

    if not all(
        col in df.columns
        for col in required_cols
    ):
        return pd.DataFrame()

    ml_df = df[[
        "Qty",
        "Price",
        "Revenue"
    ]].copy()

    ml_df = ml_df.dropna()

    return ml_df


# =====================================================
# PRODUCT DATASET
# =====================================================

@st.cache_data
def create_product_dataset(df):
    """
    Product performance dataset
    """

    if not all(
        col in df.columns
        for col in [
            "Itemname",
            "Revenue"
        ]
    ):
        return pd.DataFrame()

    product_df = (

        df.groupby(
            "Itemname"
        )

        .agg({

            "Revenue":
            "sum",

            "Qty":
            "sum"

        })

        .reset_index()

    )

    return product_df


# =====================================================
# COUNTRY DATASET
# =====================================================

@st.cache_data
def create_country_dataset(df):
    """
    Country performance dataset
    """

    if (
        "Country"
        not in df.columns
    ):
        return pd.DataFrame()

    country_df = (

        df.groupby(
            "Country"
        )["Revenue"]

        .sum()

        .reset_index()

        .sort_values(
            "Revenue",
            ascending=False
        )

    )

    return country_df


# =====================================================
# SAVE CSV
# =====================================================

def save_csv(
    df,
    filepath
):
    """
    Save dataframe
    """

    try:

        df.to_csv(
            filepath,
            index=False
        )

        return True

    except Exception:

        return False
