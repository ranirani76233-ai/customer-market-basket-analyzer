# app.py

import streamlit as st

st.set_page_config(
    page_title="Customer Market Basket Analyzer",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Custom CSS
try:
    with open("assets/custom.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except:
    pass

# Sidebar Logo
try:
    st.sidebar.image(
        "assets/logo.png",
        use_container_width=True
    )
except:
    pass

st.sidebar.title("🛒 BasketIQ")
st.sidebar.markdown("---")

# Main Page
st.title("🛒 Customer Market Basket Analyzer")

st.markdown("""
### Transform Retail Data into Actionable Insights

Analyze customer purchasing behavior, discover product associations,
segment customers, and build predictive machine learning models
through an interactive analytics platform.
""")

st.markdown("---")

# KPI Section
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="📦 Products",
        value="500+"
    )

with col2:
    st.metric(
        label="👥 Customers",
        value="1000+"
    )

with col3:
    st.metric(
        label="🧾 Transactions",
        value="10000+"
    )

with col4:
    st.metric(
        label="📈 Analytics",
        value="Real-Time"
    )

st.markdown("---")

# Features Section
st.subheader("🚀 Platform Features")

feature1, feature2, feature3 = st.columns(3)

with feature1:
    st.info("""
    ### 📊 Dashboard
    
    - Revenue KPIs
    - Sales Monitoring
    - Business Insights
    - Interactive Charts
    """)

with feature2:
    st.success("""
    ### 🔗 Market Basket Analysis
    
    - Apriori Algorithm
    - Association Rules
    - Product Affinity
    - Cross-Selling Insights
    """)

with feature3:
    st.warning("""
    ### 🤖 Machine Learning
    
    - Sales Prediction
    - Customer Segmentation
    - Model Comparison
    - Business Forecasting
    """)

st.markdown("---")

# Modules Overview
st.subheader("📂 Application Modules")

modules = {
    "📊 Dashboard": "Business KPIs and Overview",
    "📈 Analytics": "Deep Sales and Customer Analytics",
    "🔗 Market Basket": "Product Association Discovery",
    "🎯 Customer Segmentation": "Customer Group Analysis",
    "🤖 Machine Learning": "Predictive Analytics Models",
    "💡 Recommendations": "Business Recommendations"
}

for module, desc in modules.items():
    st.write(f"**{module}** — {desc}")

st.markdown("---")

# Workflow
st.subheader("🔄 Analytics Workflow")

st.markdown("""
1. Upload Retail Dataset
2. Explore Sales & Customer Analytics
3. Discover Product Associations
4. Segment Customers
5. Train Machine Learning Models
6. Generate Business Recommendations
""")

st.markdown("---")

st.success(
    "✅ Use the sidebar to navigate through different analytics modules."
)

st.caption(
    "Customer Market Basket Analyzer | Retail Analytics & Machine Learning Platform"
)
