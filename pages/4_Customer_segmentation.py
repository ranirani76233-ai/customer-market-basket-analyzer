import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Customer Segmentation")

# =====================================================
# LOAD DATA
# =====================================================

@st.cache_data
def load_data():

    df = pd.read_csv(
        "data/Assignment-1_Data.csv",
        encoding="ISO-8859-1"
    )

    return df


df = load_data()

# =====================================================
# DATA PREPARATION
# =====================================================

if "Date" in df.columns:
    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

if (
    "Qty" in df.columns and
    "Price" in df.columns
):
    df["Revenue"] = (
        df["Qty"] * df["Price"]
    )

required_cols = [
    "CustomerID",
    "BillNo",
    "Revenue",
    "Date"
]

missing_cols = [
    col for col in required_cols
    if col not in df.columns
]

if missing_cols:

    st.error(
        f"Missing columns: {missing_cols}"
    )

    st.stop()

# =====================================================
# CREATE RFM
# =====================================================

snapshot_date = df["Date"].max()

rfm = (

    df.groupby("CustomerID")

    .agg({

        "Date":
        lambda x:
        (snapshot_date - x.max()).days,

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

# =====================================================
# RFM KPIs
# =====================================================

st.subheader("📊 RFM Overview")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Avg Recency",
    round(rfm["Recency"].mean(), 1)
)

col2.metric(
    "Avg Frequency",
    round(rfm["Frequency"].mean(), 1)
)

col3.metric(
    "Avg Monetary",
    round(rfm["Monetary"].mean(), 2)
)

# =====================================================
# ELBOW METHOD
# =====================================================

st.subheader("📈 Elbow Method")

X = rfm[[
    "Recency",
    "Frequency",
    "Monetary"
]]

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

inertia = []

for k in range(2, 11):

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    model.fit(X_scaled)

    inertia.append(
        model.inertia_
    )

elbow_df = pd.DataFrame({

    "Clusters":
    range(2, 11),

    "Inertia":
    inertia

})

fig = px.line(

    elbow_df,

    x="Clusters",

    y="Inertia",

    markers=True,

    title="Elbow Method"

)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =====================================================
# CLUSTER SELECTION
# =====================================================

st.subheader("⚙ Segmentation Settings")

n_clusters = st.slider(

    "Select Number of Clusters",

    min_value=2,

    max_value=10,

    value=4

)

# =====================================================
# KMEANS
# =====================================================

kmeans = KMeans(

    n_clusters=n_clusters,

    random_state=42,

    n_init=10

)

rfm["Segment"] = kmeans.fit_predict(
    X_scaled
)

# =====================================================
# SEGMENT LABELS
# =====================================================

segment_map = {

    0: "Champions",

    1: "Loyal Customers",

    2: "Potential Customers",

    3: "At Risk",

    4: "Lost Customers",

    5: "Big Spenders",

    6: "Frequent Buyers",

    7: "Occasional Buyers",

    8: "New Customers",

    9: "Dormant Customers"

}

rfm["Segment_Name"] = (

    rfm["Segment"]

    .map(segment_map)

    .fillna("Customer Segment")

)

# =====================================================
# SEGMENT DISTRIBUTION
# =====================================================

st.subheader("👥 Customer Distribution")

segment_counts = (

    rfm["Segment_Name"]

    .value_counts()

    .reset_index()

)

segment_counts.columns = [

    "Segment",

    "Customers"

]

fig = px.pie(

    segment_counts,

    names="Segment",

    values="Customers",

    hole=0.4

)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =====================================================
# CLUSTER VISUALIZATION
# =====================================================

st.subheader("🎯 Customer Segments")

fig = px.scatter(

    rfm,

    x="Frequency",

    y="Monetary",

    color="Segment_Name",

    size="Monetary",

    hover_data=[
        "CustomerID",
        "Recency"
    ],

    title="Customer Segmentation"

)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =====================================================
# SEGMENT PROFILE
# =====================================================

st.subheader("📋 Segment Profiles")

segment_profile = (

    rfm.groupby(
        "Segment_Name"
    )

    .agg({

        "Recency":
        "mean",

        "Frequency":
        "mean",

        "Monetary":
        "mean",

        "CustomerID":
        "count"

    })

    .round(2)

    .reset_index()

)

segment_profile.columns = [

    "Segment",

    "Avg Recency",

    "Avg Frequency",

    "Avg Monetary",

    "Customers"

]

st.dataframe(
    segment_profile,
    use_container_width=True
)

# =====================================================
# SEGMENT FILTER
# =====================================================

st.subheader("🔍 Explore Segment")

selected_segment = st.selectbox(

    "Choose Segment",

    sorted(
        rfm["Segment_Name"].unique()
    )

)

segment_customers = rfm[
    rfm["Segment_Name"]
    == selected_segment
]

st.dataframe(

    segment_customers,

    use_container_width=True

)

# =====================================================
# BUSINESS RECOMMENDATIONS
# =====================================================

st.subheader("💡 Recommendations")

st.success(
    """
    Champions:
    Reward with exclusive offers and loyalty programs.

    Loyal Customers:
    Upsell premium products and memberships.

    Potential Customers:
    Encourage repeat purchases with targeted campaigns.

    At Risk:
    Send reactivation emails and discounts.

    Lost Customers:
    Run win-back campaigns and personalized promotions.
    """
)

# =====================================================
# DOWNLOAD RESULTS
# =====================================================

st.subheader("📥 Export Segments")

csv = rfm.to_csv(
    index=False
).encode("utf-8")

st.download_button(

    label="Download Customer Segments",

    data=csv,

    file_name="customer_segments.csv",

    mime="text/csv"

)

# =====================================================
# SAVE MODEL OPTION
# =====================================================

st.subheader("💾 Save Segmentation Model")

if st.button("Save KMeans Model"):

    import joblib

    joblib.dump(
        kmeans,
        "models/customer_segmentation.pkl"
    )

    st.success(
        "Model saved to models/customer_segmentation.pkl"
    )
