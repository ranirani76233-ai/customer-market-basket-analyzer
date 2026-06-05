"""
data_preprocessing.py

Customer Market Basket Analytics
--------------------------------
Functions for:

1. Load Data
2. Data Cleaning
3. Missing Value Handling
4. Duplicate Removal
5. Data Type Conversion
6. Feature Creation
"""

import pandas as pd
import numpy as np


# =====================================================
# LOAD DATA
# =====================================================

def load_data(file_path):
    """
    Load CSV dataset
    """

    try:
        df = pd.read_csv(
            file_path,
            encoding="ISO-8859-1"
        )

        return df

    except Exception as e:
        raise Exception(
            f"Error loading dataset: {e}"
        )


# =====================================================
# STANDARDIZE COLUMN NAMES
# =====================================================

def standardize_columns(df):

    df.columns = (
        df.columns
        .str.strip()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )

    return df


# =====================================================
# HANDLE MISSING VALUES
# =====================================================

def handle_missing_values(df):

    initial_rows = len(df)

    df = df.dropna(
        subset=["CustomerID"]
    )

    removed_rows = (
        initial_rows - len(df)
    )

    print(
        f"Removed {removed_rows} rows with missing CustomerID"
    )

    return df


# =====================================================
# REMOVE DUPLICATES
# =====================================================

def remove_duplicates(df):

    before = len(df)

    df = df.drop_duplicates()

    after = len(df)

    print(
        f"Removed {before - after} duplicate rows"
    )

    return df


# =====================================================
# CONVERT DATATYPES
# =====================================================

def convert_datatypes(df):

    if "Date" in df.columns:

        df["Date"] = pd.to_datetime(
            df["Date"],
            errors="coerce"
        )

    if "CustomerID" in df.columns:

        df["CustomerID"] = (
            df["CustomerID"]
            .astype(str)
        )

    return df


# =====================================================
# REMOVE CANCELLED ORDERS
# =====================================================

def remove_cancelled_orders(df):

    if "BillNo" not in df.columns:
        return df

    df = df[
        ~df["BillNo"]
        .astype(str)
        .str.startswith("C")
    ]

    return df


# =====================================================
# REMOVE NEGATIVE VALUES
# =====================================================

def remove_invalid_transactions(df):

    if "Quantity" in df.columns:

        df = df[
            df["Quantity"] > 0
        ]

    if "Price" in df.columns:

        df = df[
            df["Price"] > 0
        ]

    return df


# =====================================================
# CREATE SALES FEATURES
# =====================================================

def create_sales_features(df):

    if (
        "Quantity" in df.columns
        and "Price" in df.columns
    ):

        df["Sales"] = (
            df["Quantity"]
            * df["Price"]
        )

    return df


# =====================================================
# DATE FEATURES
# =====================================================

def create_date_features(df):

    if "Date" not in df.columns:
        return df

    df["Year"] = df["Date"].dt.year

    df["Month"] = df["Date"].dt.month

    df["Month_Name"] = (
        df["Date"]
        .dt.month_name()
    )

    df["Quarter"] = (
        df["Date"]
        .dt.quarter
    )

    df["Day"] = df["Date"].dt.day

    df["Weekday"] = (
        df["Date"]
        .dt.day_name()
    )

    return df


# =====================================================
# DATA SUMMARY
# =====================================================

def data_summary(df):

    summary = {

        "Rows": len(df),

        "Columns": len(df.columns),

        "Customers":
        df["CustomerID"].nunique()
        if "CustomerID" in df.columns
        else 0,

        "Products":
        df["Itemname"].nunique()
        if "Itemname" in df.columns
        else 0,

        "Transactions":
        df["BillNo"].nunique()
        if "BillNo" in df.columns
        else 0
    }

    return summary


# =====================================================
# MASTER PREPROCESSING PIPELINE
# =====================================================

def preprocess_data(df):

    df = standardize_columns(df)

    df = handle_missing_values(df)

    df = remove_duplicates(df)

    df = convert_datatypes(df)

    df = remove_cancelled_orders(df)

    df = remove_invalid_transactions(df)

    df = create_sales_features(df)

    df = create_date_features(df)

    return df


# =====================================================
# MAIN TEST
# =====================================================

if __name__ == "__main__":

    FILE_PATH = "data/raw/Assignment-1_Data.csv"

    df = load_data(FILE_PATH)

    df = preprocess_data(df)

    print("\nDataset Shape:")
    print(df.shape)

    print("\nSummary:")
    print(data_summary(df))

    print("\nSample Data:")
    print(df.head())
