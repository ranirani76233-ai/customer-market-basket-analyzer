import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ==================================================

# PAGE CONFIG

# ==================================================

st.set_page_config(
page_title="Advanced Retail Analytics",
page_icon="📈",
layout="wide"
)

st.title("📈 Advanced Retail Analytics")
st.markdown("Deep business insights, customer intelligence, and revenue analytics")

# ==================================================

# LOAD DATA

# ==================================================

@st.cache_data
def load_data():

```
file_path = "data/Assignment-1_Data.csv.csv"

try:
    df = pd.read_csv(
        file_path,
        sep=None,
        engine="python",
        on_bad_lines="skip"
    )

except Exception as e:

    st.error(f"Dataset Loading Error: {e}")
    st.stop()

return df
```

df = load_data()

# ==================================================

# CLEAN COLUMNS

# ==================================================

df.columns = df.columns.str.strip()

# ==================================================

# AUTO DETECT COLUMNS

# ==================================================

qty_col = None
price_col = None
date_col = None
customer_col = None
product_col = None
country_col = None
bill_col = None

for col in df.columns:

```
c = col.lower()

if c in ["qty", "quantity"]:
    qty_col = col

elif c in ["price", "unitprice"]:
    price_col = col

elif c == "date":
    date_col = col

elif c in ["customerid", "customer_id"]:
    customer_col = col

elif c in ["itemname", "description", "product"]:
    product_col = col

elif c in ["country"]:
    country_col = col

elif c in ["billno", "invoice", "invoiceno"]:
    bill_col = col
```

# ==================================================

# PREPARE DATA

# ==================================================

if qty_col:
df[qty_col] = pd.to_numeric(
df[qty_col],
errors="coerce"
)

if price_col:
df[price_col] = pd.to_numeric(
df[price_col],
errors="coerce"
)

if date_col:
df[date_col] = pd.to_datetime(
df[date_col],
errors="coerce"
)

if qty_col and price_col:

```
df["Revenue"] = (
    df[qty_col].fillna(0)
    *
    df[price_col].fillna(0)
)
```

else:

```
st.error("Revenue cannot be calculated.")
st.write("Columns Found:", list(df.columns))
st.stop()
```

# ==================================================

# KPI SECTION

# ==================================================

st.subheader("📊 Executive Overview")

col1, col2, col3, col4 = st.columns(4)

total_revenue = df["Revenue"].sum()

total_orders = (
df[bill_col].nunique()
if bill_col else 0
)

total_customers = (
df[customer_col].nunique()
if customer_col else 0
)

total_products = (
df[product_col].nunique()
if product_col else 0
)

col1.metric(
"Revenue",
f"${total_revenue:,.0f}"
)

col2.metric(
"Orders",
f"{total_orders:,}"
)

col3.metric(
"Customers",
f"{total_customers:,}"
)

col4.metric(
"Products",
f"{total_products:,}"
)

st.divider()

# ==================================================

# MONTHLY REVENUE TREND

# ==================================================

if date_col:

```
st.subheader("📈 Monthly Revenue Trend")

monthly = (
    df.groupby(
        df[date_col].dt.to_period("M")
    )["Revenue"]
    .sum()
    .reset_index()
)

monthly[date_col] = monthly[
    date_col
].astype(str)

fig = px.line(
    monthly,
    x=date_col,
    y="Revenue",
    markers=True
)

st.plotly_chart(
    fig,
    use_container_width=True
)
```

# ==================================================

# REVENUE DISTRIBUTION

# ==================================================

st.subheader("💰 Revenue Distribution")

fig = px.histogram(
df,
x="Revenue",
nbins=50
)

st.plotly_chart(
fig,
use_container_width=True
)

# ==================================================

# TOP PRODUCTS

# ==================================================

if product_col:

```
st.subheader("🏆 Top Revenue Products")

top_products = (
    df.groupby(product_col)["Revenue"]
    .sum()
    .sort_values(
        ascending=False
    )
    .head(15)
    .reset_index()
)

fig = px.bar(
    top_products,
    x="Revenue",
    y=product_col,
    orientation="h"
)

st.plotly_chart(
    fig,
    use_container_width=True
)
```

# ==================================================

# CUSTOMER SEGMENTATION

# ==================================================

if customer_col:

```
st.subheader("👥 Customer Segmentation")

customer_df = (
    df.groupby(customer_col)
    .agg({
        "Revenue": "sum"
    })
    .reset_index()
)

customer_df["Segment"] = pd.qcut(
    customer_df["Revenue"],
    4,
    labels=[
        "Bronze",
        "Silver",
        "Gold",
        "Platinum"
    ]
)

fig = px.pie(
    customer_df,
    names="Segment"
)

st.plotly_chart(
    fig,
    use_container_width=True
)
```

# ==================================================

# COUNTRY ANALYSIS

# ==================================================

if country_col:

```
st.subheader("🌍 Country Performance")

country_df = (
    df.groupby(country_col)
    .agg({
        "Revenue": "sum"
    })
    .reset_index()
    .sort_values(
        "Revenue",
        ascending=False
    )
    .head(15)
)

fig = px.bar(
    country_df,
    x=country_col,
    y="Revenue"
)

st.plotly_chart(
    fig,
    use_container_width=True
)
```

# ==================================================

# PRODUCT PERFORMANCE TABLE

# ==================================================

if product_col:

```
st.subheader("📦 Product Performance")

product_perf = (
    df.groupby(product_col)
    .agg({
        qty_col: "sum",
        "Revenue": "sum"
    })
    .reset_index()
    .sort_values(
        "Revenue",
        ascending=False
    )
    .head(25)
)

st.dataframe(
    product_perf,
    use_container_width=True
)
```

# ==================================================

# CORRELATION MATRIX

# ==================================================

st.subheader("🔥 Correlation Analysis")

numeric_df = df.select_dtypes(
include=np.number
)

if len(numeric_df.columns) > 1:

```
corr = numeric_df.corr()

fig = px.imshow(
    corr,
    text_auto=True,
    aspect="auto"
)

st.plotly_chart(
    fig,
    use_container_width=True
)
```

# ==================================================

# SALES INSIGHTS

# ==================================================

st.subheader("💡 AI Business Insights")

best_product = ""

if product_col:

```
best_product = (
    df.groupby(product_col)["Revenue"]
    .sum()
    .idxmax()
)
```

st.success(
f"""
• Total Revenue: ${total_revenue:,.0f}

```
• Top Product: {best_product}

• Unique Customers: {total_customers:,}

• Total Orders: {total_orders:,}

• Product Portfolio: {total_products:,}
"""
```

)

# ==================================================

# RAW DATA

# ==================================================

with st.expander("View Dataset"):

```
st.dataframe(
    df.head(100),
    use_container_width=True
)
```

# ==================================================

# DOWNLOAD

# ==================================================

csv = df.to_csv(
index=False
).encode("utf-8")

st.download_button(
"⬇ Download Dataset",
csv,
"advanced_retail_analytics.csv",
"text/csv"
)
