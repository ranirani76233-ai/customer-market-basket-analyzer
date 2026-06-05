"""
feature_engineering.py

Creates customer-level features for:

1. Customer Analytics
2. Predictive Modeling
3. Machine Learning
4. Customer Segmentation
"""

import pandas as pd
import numpy as np


# =====================================================
# CUSTOMER AGGREGATION FEATURES
# =====================================================

def create_customer_features(df):

    customer_features = (
        df.groupby("CustomerID")
        .agg(
            Total_Orders=("BillNo", "nunique"),
            Total_Items=("Quantity", "sum"),
            Total_Spend=("Sales", "sum"),
            Unique_Products=("Itemname", "nunique"),
            Avg_Price=("Price", "mean")
        )
        .reset_index()
    )

    customer_features["Avg_Order_Value"] = (
        customer_features["Total_Spend"]
        / customer_features["Total_Orders"]
    )

    return customer_features


# =====================================================
# RFM FEATURES
# =====================================================

def create_rfm_features(df):

    reference_date = (
        df["Date"].max()
        + pd.Timedelta(days=1)
    )

    rfm = (
        df.groupby("CustomerID")
        .agg(
            Recency=(
                "Date",
                lambda x: (
                    reference_date - x.max()
                ).days
            ),

            Frequency=(
                "BillNo",
                "nunique"
            ),

            Monetary=(
                "Sales",
                "sum"
            )
        )
        .reset_index()
    )

    return rfm


# =====================================================
# PRODUCT DIVERSITY FEATURES
# =====================================================

def create_product_features(df):

    product_features = (
        df.groupby("CustomerID")
        .agg(
            Unique_Products=(
                "Itemname",
                "nunique"
            ),

            Total_Items=(
                "Quantity",
                "sum"
            )
        )
        .reset_index()
    )

    product_features["Product_Diversity"] = (
        product_features["Unique_Products"]
        / product_features["Total_Items"]
    )

    return product_features


# =====================================================
# PURCHASE BEHAVIOR FEATURES
# =====================================================

def create_purchase_behavior_features(df):

    behavior = (
        df.groupby("CustomerID")
        .agg(
            First_Purchase=("Date", "min"),
            Last_Purchase=("Date", "max")
        )
        .reset_index()
    )

    behavior["Customer_Lifespan"] = (
        behavior["Last_Purchase"]
        - behavior["First_Purchase"]
    ).dt.days

    return behavior[
        [
            "CustomerID",
            "Customer_Lifespan"
        ]
    ]


# =====================================================
# BASKET FEATURES
# =====================================================

def create_basket_features(df):

    basket = (
        df.groupby(
            ["CustomerID", "BillNo"]
        )
        .agg(
            Basket_Value=("Sales", "sum"),
            Basket_Items=("Quantity", "sum")
        )
        .reset_index()
    )

    basket_features = (
        basket.groupby("CustomerID")
        .agg(
            Avg_Basket_Value=(
                "Basket_Value",
                "mean"
            ),

            Max_Basket_Value=(
                "Basket_Value",
                "max"
            ),

            Avg_Basket_Items=(
                "Basket_Items",
                "mean"
            )
        )
        .reset_index()
    )

    return basket_features


# =====================================================
# CUSTOMER COUNTRY FEATURES
# =====================================================

def create_country_features(df):

    country = (
        df.groupby("CustomerID")
        .agg(
            Country=("Country", "first")
        )
        .reset_index()
    )

    return country


# =====================================================
# BUILD ML DATASET
# =====================================================

def build_ml_dataset(df):

    customer = create_customer_features(df)

    rfm = create_rfm_features(df)

    product = create_product_features(df)

    behavior = create_purchase_behavior_features(df)

    basket = create_basket_features(df)

    country = create_country_features(df)

    ml_df = (
        customer
        .merge(rfm, on="CustomerID")
        .merge(product, on="CustomerID")
        .merge(behavior, on="CustomerID")
        .merge(basket, on="CustomerID")
        .merge(country, on="CustomerID")
    )

    return ml_df


# =====================================================
# CREATE TARGET VARIABLE
# =====================================================

def create_target_variable(df):

    """
    Predict Total Spend
    """

    ml_df = build_ml_dataset(df)

    ml_df["Target"] = (
        ml_df["Total_Spend"]
    )

    return ml_df


# =====================================================
# PREPARE X AND y
# =====================================================

def prepare_ml_data(df):

    ml_df = create_target_variable(df)

    X = ml_df.drop(
        columns=[
            "CustomerID",
            "Country",
            "Target"
        ],
        errors="ignore"
    )

    X = pd.get_dummies(
        X,
        drop_first=True
    )

    y = ml_df["Target"]

    return X, y, ml_df


# =====================================================
# FEATURE IMPORTANCE DATAFRAME
# =====================================================

def feature_summary(ml_df):

    summary = pd.DataFrame({

        "Feature": ml_df.columns,

        "Missing_Values":
        ml_df.isnull().sum().values,

        "Data_Type":
        ml_df.dtypes.values
    })

    return summary


# =====================================================
# MAIN TEST
# =====================================================

if __name__ == "__main__":

    from src.data_preprocessing import (
        load_data,
        preprocess_data
    )

    df = load_data(
        "data/raw/Assignment-1_Data.csv"
    )

    df = preprocess_data(df)

    ml_df = build_ml_dataset(df)

    print("\nML Dataset Shape:")
    print(ml_df.shape)

    print("\nSample Features:")
    print(ml_df.head())

    X, y, final_df = prepare_ml_data(df)

    print("\nFeature Matrix Shape:")
    print(X.shape)

    print("\nTarget Shape:")
    print(y.shape)
