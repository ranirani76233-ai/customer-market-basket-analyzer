# utils/analytics.py

import pandas as pd
import numpy as np


class Analytics:

    @staticmethod
    def calculate_kpis(df):
        """
        Calculate business KPIs
        """

        total_revenue = df["Sales"].sum()

        total_orders = df["Order_ID"].nunique()

        total_customers = df["Customer_ID"].nunique()

        avg_order_value = (
            total_revenue / total_orders
            if total_orders > 0 else 0
        )

        return {
            "Total Revenue": round(total_revenue, 2),
            "Total Orders": total_orders,
            "Total Customers": total_customers,
            "Average Order Value": round(avg_order_value, 2)
        }

    @staticmethod
    def monthly_sales(df):

        sales = (
            df.groupby("Month")["Sales"]
            .sum()
            .reset_index()
            .sort_values("Sales", ascending=False)
        )

        return sales

    @staticmethod
    def top_products(df, top_n=10):

        products = (
            df.groupby("Product")["Sales"]
            .sum()
            .reset_index()
            .sort_values("Sales", ascending=False)
            .head(top_n)
        )

        return products

    @staticmethod
    def customer_analysis(df):

        customer = (
            df.groupby("Customer_ID")
            .agg({
                "Sales": "sum",
                "Order_ID": "count"
            })
            .reset_index()
        )

        customer.columns = [
            "Customer_ID",
            "Total_Spend",
            "Total_Orders"
        ]

        return customer

    @staticmethod
    def country_analysis(df):

        if "Country" not in df.columns:
            return pd.DataFrame()

        country = (
            df.groupby("Country")["Sales"]
            .sum()
            .reset_index()
            .sort_values("Sales", ascending=False)
        )

        return country

    @staticmethod
    def category_analysis(df):

        if "Category" not in df.columns:
            return pd.DataFrame()

        category = (
            df.groupby("Category")["Sales"]
            .sum()
            .reset_index()
            .sort_values("Sales", ascending=False)
        )

        return category

    @staticmethod
    def sales_trend(df):

        trend = (
            df.groupby("InvoiceDate")["Sales"]
            .sum()
            .reset_index()
        )

        return trend

    @staticmethod
    def generate_insights(df):

        insights = []

        top_product = (
            df.groupby("Product")["Sales"]
            .sum()
            .idxmax()
        )

        top_product_sales = (
            df.groupby("Product")["Sales"]
            .sum()
            .max()
        )

        insights.append(
            f"Top selling product is {top_product} with revenue of {top_product_sales:,.2f}"
        )

        top_customer = (
            df.groupby("Customer_ID")["Sales"]
            .sum()
            .idxmax()
        )

        insights.append(
            f"Customer {top_customer} generated the highest revenue."
        )

        avg_sales = df["Sales"].mean()

        insights.append(
            f"Average transaction value is {avg_sales:.2f}"
        )

        return insights
