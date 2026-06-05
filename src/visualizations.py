"""
visualizations.py

Reusable charts for:

1. Dashboard Analytics
2. Market Basket Analysis
3. Machine Learning
4. Model Evaluation
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# =====================================================
# KPI METRICS
# =====================================================

def calculate_kpis(df):

    kpis = {

        "Total Revenue":
        round(df["Sales"].sum(), 2),

        "Total Orders":
        df["BillNo"].nunique(),

        "Total Customers":
        df["CustomerID"].nunique(),

        "Total Products":
        df["Itemname"].nunique(),

        "Average Order Value":
        round(
            df.groupby("BillNo")["Sales"]
            .sum()
            .mean(),
            2
        )
    }

    return kpis


# =====================================================
# MONTHLY SALES TREND
# =====================================================

def monthly_sales_chart(df):

    monthly = (
        df.groupby(
            ["Year", "Month"]
        )["Sales"]
        .sum()
        .reset_index()
    )

    fig = px.line(
        monthly,
        x="Month",
        y="Sales",
        markers=True,
        title="Monthly Sales Trend"
    )

    return fig


# =====================================================
# TOP PRODUCTS
# =====================================================

def top_products_chart(
    df,
    top_n=10
):

    top_products = (

        df.groupby("Itemname")
        ["Sales"]
        .sum()
        .nlargest(top_n)
        .reset_index()

    )

    fig = px.bar(

        top_products,

        x="Sales",

        y="Itemname",

        orientation="h",

        title=f"Top {top_n} Products by Revenue"
    )

    return fig


# =====================================================
# TOP CUSTOMERS
# =====================================================

def top_customers_chart(
    df,
    top_n=10
):

    customers = (

        df.groupby("CustomerID")
        ["Sales"]
        .sum()
        .nlargest(top_n)
        .reset_index()

    )

    fig = px.bar(

        customers,

        x="Sales",

        y="CustomerID",

        orientation="h",

        title=f"Top {top_n} Customers"
    )

    return fig


# =====================================================
# COUNTRY SALES
# =====================================================

def country_sales_chart(df):

    country_sales = (

        df.groupby("Country")
        ["Sales"]
        .sum()
        .sort_values(
            ascending=False
        )
        .reset_index()

    )

    fig = px.bar(

        country_sales,

        x="Country",

        y="Sales",

        title="Revenue by Country"
    )

    return fig


# =====================================================
# PIE CHART
# =====================================================

def country_pie_chart(df):

    country_sales = (

        df.groupby("Country")
        ["Sales"]
        .sum()
        .reset_index()

    )

    fig = px.pie(

        country_sales,

        names="Country",

        values="Sales",

        title="Country Contribution"
    )

    return fig


# =====================================================
# PRODUCT FREQUENCY
# =====================================================

def product_frequency_chart(
    df,
    top_n=15
):

    products = (

        df.groupby("Itemname")
        ["Quantity"]
        .sum()
        .nlargest(top_n)
        .reset_index()

    )

    fig = px.bar(

        products,

        x="Quantity",

        y="Itemname",

        orientation="h",

        title="Most Purchased Products"
    )

    return fig


# =====================================================
# ASSOCIATION MATRIX HEATMAP
# =====================================================

def association_heatmap(
    association_matrix
):

    fig = px.imshow(

        association_matrix,

        title="Product Association Matrix",

        aspect="auto"
    )

    return fig


# =====================================================
# SUPPORT VS CONFIDENCE
# =====================================================

def support_confidence_plot(
    rules
):

    fig = px.scatter(

        rules,

        x="support",

        y="confidence",

        size="lift",

        title="Support vs Confidence"
    )

    return fig


# =====================================================
# MODEL COMPARISON
# =====================================================

def model_comparison_chart(
    comparison_df
):

    fig = px.bar(

        comparison_df,

        x="Model",

        y="R2",

        title="Model R² Comparison"
    )

    return fig


# =====================================================
# RMSE COMPARISON
# =====================================================

def rmse_comparison_chart(
    comparison_df
):

    fig = px.bar(

        comparison_df,

        x="Model",

        y="RMSE",

        title="RMSE Comparison"
    )

    return fig


# =====================================================
# MSE COMPARISON
# =====================================================

def mse_comparison_chart(
    comparison_df
):

    fig = px.bar(

        comparison_df,

        x="Model",

        y="MSE",

        title="MSE Comparison"
    )

    return fig


# =====================================================
# ACTUAL VS PREDICTED
# =====================================================

def actual_vs_predicted_plot(
    y_test,
    predictions
):

    plot_df = pd.DataFrame({

        "Actual": y_test,

        "Predicted": predictions

    })

    fig = px.scatter(

        plot_df,

        x="Actual",

        y="Predicted",

        title="Actual vs Predicted"
    )

    fig.add_shape(

        type="line",

        x0=min(y_test),

        y0=min(y_test),

        x1=max(y_test),

        y1=max(y_test)

    )

    return fig


# =====================================================
# RESIDUAL PLOT
# =====================================================

def residual_plot(
    y_test,
    predictions
):

    residuals = y_test - predictions

    fig = px.scatter(

        x=predictions,

        y=residuals,

        title="Residual Plot",

        labels={
            "x": "Predicted",
            "y": "Residual"
        }
    )

    return fig


# =====================================================
# FEATURE IMPORTANCE
# =====================================================

def feature_importance_plot(
    model,
    feature_names
):

    importance = pd.DataFrame({

        "Feature": feature_names,

        "Importance":
        model.feature_importances_

    })

    importance = (

        importance
        .sort_values(
            by="Importance",
            ascending=False
        )
        .head(15)

    )

    fig = px.bar(

        importance,

        x="Importance",

        y="Feature",

        orientation="h",

        title="Feature Importance"
    )

    return fig


# =====================================================
# CORRELATION HEATMAP
# =====================================================

def correlation_heatmap(
    df
):

    corr = df.corr(
        numeric_only=True
    )

    fig = px.imshow(

        corr,

        text_auto=True,

        title="Correlation Matrix"
    )

    return fig


# =====================================================
# MODEL RANKING TABLE
# =====================================================

def rank_models(
    comparison_df
):

    ranking = (

        comparison_df
        .sort_values(
            by="R2",
            ascending=False
        )
        .reset_index(drop=True)

    )

    ranking.index += 1

    ranking.index.name = "Rank"

    return ranking
