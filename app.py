import streamlit as st

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Customer Market Basket Analytics",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------
st.markdown("""
<style>
.main-header {
    font-size: 42px;
    font-weight: bold;
    color: #1f77b4;
}

.sub-header {
    font-size: 22px;
    color: #555;
}

.metric-card {
    background-color: #f7f7f7;
    padding: 15px;
    border-radius: 10px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.title("🛒 Navigation")

st.sidebar.info("""
Use the sidebar to access:

📊 Dashboard

🛒 Market Basket Analysis

🤖 Machine Learning

📈 Model Evaluation
""")

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.markdown(
    '<p class="main-header">🛒 Customer Market Basket Analytics</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="sub-header">Market Basket Analysis • Predictive Analytics • Machine Learning Dashboard</p>',
    unsafe_allow_html=True
)

st.divider()

# --------------------------------------------------
# PROJECT OVERVIEW
# --------------------------------------------------
st.header("📌 Project Overview")

st.write("""
This application analyzes customer purchase behavior and identifies
products frequently purchased together using **Market Basket Analysis**.

The project also includes multiple **Machine Learning Models**
to predict customer spending and basket value.

The application contains:

- Customer Analytics
- Product Analytics
- Root Cause Analysis
- Market Basket Analysis (Apriori)
- Association Rules
- Product Association Matrix
- Machine Learning Models
- Hyperparameter Tuning
- Model Comparison
- Performance Evaluation
""")

# --------------------------------------------------
# FEATURE CARDS
# --------------------------------------------------
st.header("🚀 Key Features")

col1, col2, col3 = st.columns(3)

with col1:
    st.info("""
    ### 📊 Analytics

    - Revenue Analysis
    - Product Performance
    - Customer Insights
    - Country Analysis
    """)

with col2:
    st.info("""
    ### 🛒 Market Basket

    - Apriori Algorithm
    - Association Rules
    - Support
    - Confidence
    - Lift
    """)

with col3:
    st.info("""
    ### 🤖 Machine Learning

    - Linear Regression
    - KNN Regression
    - Random Forest
    - Gradient Boosting
    - Hyperparameter Tuning
    """)

# --------------------------------------------------
# WORKFLOW
# --------------------------------------------------
st.header("🔄 Project Workflow")

st.markdown("""
```text
Raw Retail Dataset
        │
        ▼
Data Cleaning
        │
        ▼
Feature Engineering
        │
        ▼
Analytics Dashboard
        │
        ├──────────────► Market Basket Analysis
        │                    │
        │                    ▼
        │             Association Rules
        │
        ▼
Machine Learning
        │
        ▼
Model Evaluation
        │
        ▼
Best Model Selection
