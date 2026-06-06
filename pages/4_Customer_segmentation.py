import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="👥",
    layout="wide"
)

st.title("👥 Customer Segmentation Dashboard")

# ==================================================
# LOAD DATA (SAFE)
# ==================================================

@st.cache_data
def load_data():

    try:
        df = pd.read_csv(
            "data/Assignment-1_Data.csv.csv",
            sep=None,
            engine="python",
            on_bad_lines="skip"
        )
        return df

    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return pd.DataFrame()


df = load_data()

if df.empty:
    st.error("Dataset is empty or failed to load.")
    st.stop()

# ==================================================
# CLEAN COLUMNS
# ==================================================

df.columns = df.columns.str.strip()

st.write("Detected Columns:", df.columns.tolist())

# ==================================================
# AUTO DETECT COLUMNS
# ==================================================

customer_col = None
date_col = None
qty_col = None
price_col = None
invoice_col = None

for col in df.columns:

    c = col.lower()

    if c in ["customerid", "customer_id"]:
        customer_col = col

    elif c in ["date", "invoicedate"]:
        date_col = col

    elif c in ["qty", "quantity"]:
        qty_col = col

    elif c in ["price", "unitprice"]:
        price_col = col

    elif c in ["billno", "invoice", "invoiceno"]:
        invoice_col = col

# ==================================================
# VALIDATION
# ==================================================

if customer_col is None:

    st.error(f"""
    Customer column not found!

    Available Columns:
    {df.columns.tolist()}
    """)
    st.stop()

# ==================================================
# CONVERT DATA TYPES
# ==================================================

if qty_col:
    df[qty_col] = pd.to_numeric(df[qty_col], errors="coerce")

if price_col:
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")

if date_col:
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

# ==================================================
# CREATE REVENUE
# ==================================================

if qty_col and price_col:

    df["Revenue"] = (
        df[qty_col].fillna(0) *
        df[price_col].fillna(0)
    )

else:

    st.error("Revenue cannot be created (Qty/Price missing)")
    st.stop()

# ==================================================
# RFM ANALYSIS
# ==================================================

st.subheader("📊 RFM Analysis")

if date_col:

    snapshot_date = df[date_col].max() + pd.Timedelta(days=1)

    rfm = df.groupby(customer_col).agg({

        date_col: lambda x: (snapshot_date - x.max()).days,
        invoice_col if invoice_col else customer_col: "nunique",
        "Revenue": "sum"

    }).reset_index()

    rfm.columns = [
        "CustomerID",
        "Recency",
        "Frequency",
        "Monetary"
    ]

else:

    rfm = df.groupby(customer_col).agg({

        "Revenue": "sum"

    }).reset_index()

    rfm["Recency"] = 0
    rfm["Frequency"] = 1
    rfm.rename(columns={"Revenue": "Monetary"}, inplace=True)

# ==================================================
# KPIs
# ==================================================

col1, col2, col3 = st.columns(3)

col1.metric("Customers", len(rfm))
col2.metric("Avg Spend", f"${rfm['Monetary'].mean():,.0f}")
col3.metric("Max Spend", f"${rfm['Monetary'].max():,.0f}")

# ==================================================
# CLUSTERING
# ==================================================

st.subheader("🤖 Customer Segmentation (KMeans)")

features = rfm[["Recency", "Frequency", "Monetary"]]

scaler = StandardScaler()
scaled = scaler.fit_transform(features)

n_clusters = st.sidebar.slider(
    "Number of Segments",
    2, 10, 4
)

model = KMeans(
    n_clusters=n_clusters,
    random_state=42,
    n_init=10
)

rfm["Cluster"] = model.fit_predict(scaled)

# ==================================================
# CLUSTER SUMMARY
# ==================================================

cluster_summary = rfm.groupby("Cluster").agg({

    "Recency": "mean",
    "Frequency": "mean",
    "Monetary": "mean"

}).round(2)

st.dataframe(cluster_summary, use_container_width=True)

# ==================================================
# SEGMENT DISTRIBUTION
# ==================================================

st.subheader("📈 Segment Distribution")

fig = px.pie(rfm, names="Cluster")
st.plotly_chart(fig, use_container_width=True)

# ==================================================
# CUSTOMER VALUE MAP
# ==================================================

st.subheader("💰 Customer Value Map")

fig = px.scatter(
    rfm,
    x="Frequency",
    y="Monetary",
    color="Cluster",
    size="Monetary",
    hover_data=["CustomerID"]
)

st.plotly_chart(fig, use_container_width=True)

# ==================================================
# RECENCY ANALYSIS
# ==================================================

st.subheader("⏰ Recency Analysis")

fig = px.box(
    rfm,
    x="Cluster",
    y="Recency",
    color="Cluster"
)

st.plotly_chart(fig, use_container_width=True)

# ==================================================
# TOP CUSTOMERS
# ==================================================

st.subheader("🏆 Top Customers")

top_customers = rfm.sort_values("Monetary", ascending=False).head(20)

st.dataframe(top_customers, use_container_width=True)

# ==================================================
# INSIGHTS
# ==================================================

best_cluster = rfm.groupby("Cluster")["Monetary"].mean().idxmax()

st.success(f"""
Best Customer Segment: Cluster {best_cluster}

Total Customers: {len(rfm)}

Average Revenue: ${rfm['Monetary'].mean():,.2f}
""")
