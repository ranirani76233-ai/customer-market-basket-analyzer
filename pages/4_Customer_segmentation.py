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
# LOAD DATA
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
    st.error("Dataset is empty")
    st.stop()

# ==================================================
# CLEAN COLUMNS
# ==================================================

df.columns = df.columns.str.strip()

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

if customer_col is None:
    st.error("Customer column not found")
    st.stop()

# ==================================================
# CONVERT TYPES
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
    df["Revenue"] = df[qty_col] * df[price_col]
else:
    st.error("Missing Qty/Price columns")
    st.stop()

# ==================================================
# DROP BAD ROWS (IMPORTANT FIX 🔥)
# ==================================================

df = df.dropna(subset=["Revenue", customer_col])

df = df[df["Revenue"] >= 0]

# ==================================================
# RFM ANALYSIS
# ==================================================

st.subheader("📊 RFM Analysis")

if date_col:

    df = df.dropna(subset=[date_col])

    snapshot_date = df[date_col].max() + pd.Timedelta(days=1)

    rfm = df.groupby(customer_col).agg({

        date_col: lambda x: (snapshot_date - x.max()).days,
        invoice_col if invoice_col else customer_col: "nunique",
        "Revenue": "sum"

    }).reset_index()

    rfm.columns = ["CustomerID", "Recency", "Frequency", "Monetary"]

else:

    rfm = df.groupby(customer_col).agg({

        "Revenue": "sum"

    }).reset_index()

    rfm.columns = ["CustomerID", "Monetary"]

    rfm["Recency"] = 0
    rfm["Frequency"] = 1

# ==================================================
# 🚨 FINAL CLEANING (FIX FOR KMEANS ERROR)
# ==================================================

rfm["Recency"] = rfm["Recency"].fillna(rfm["Recency"].median())
rfm["Frequency"] = rfm["Frequency"].fillna(1)
rfm["Monetary"] = rfm["Monetary"].fillna(0)

rfm = rfm.replace([np.inf, -np.inf], np.nan).dropna()

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

st.subheader("🤖 KMeans Segmentation")

features = rfm[["Recency", "Frequency", "Monetary"]]

scaler = StandardScaler()
scaled = scaler.fit_transform(features)

n_clusters = st.sidebar.slider("Clusters", 2, 10, 4)

model = KMeans(
    n_clusters=n_clusters,
    random_state=42,
    n_init=10
)

rfm["Cluster"] = model.fit_predict(scaled)

# ==================================================
# CLUSTER SUMMARY
# ==================================================

st.dataframe(
    rfm.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean(),
    use_container_width=True
)

# ==================================================
# VISUALS
# ==================================================

st.subheader("📈 Segment Distribution")

st.plotly_chart(px.pie(rfm, names="Cluster"), use_container_width=True)

st.subheader("💰 Customer Value Map")

st.plotly_chart(
    px.scatter(
        rfm,
        x="Frequency",
        y="Monetary",
        color="Cluster",
        size="Monetary"
    ),
    use_container_width=True
)

st.subheader("⏰ Recency Analysis")

st.plotly_chart(
    px.box(rfm, x="Cluster", y="Recency", color="Cluster"),
    use_container_width=True
)

# ==================================================
# INSIGHTS
# ==================================================

best_cluster = rfm.groupby("Cluster")["Monetary"].mean().idxmax()

st.success(f"""
Best Customer Segment: Cluster {best_cluster}
Total Customers: {len(rfm)}
""")
