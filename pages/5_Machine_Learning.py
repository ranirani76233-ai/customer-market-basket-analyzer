import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="Machine Learning",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Machine Learning - Sales Prediction")

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
        st.error(f"Dataset Error: {e}")
        return pd.DataFrame()


df = load_data()

if df.empty:
    st.error("Dataset not loaded")
    st.stop()

# ==================================================
# CLEAN COLUMNS
# ==================================================

df.columns = df.columns.str.strip()

st.write("Columns Found:", df.columns.tolist())

# ==================================================
# AUTO DETECT COLUMNS
# ==================================================

qty_col = None
price_col = None
date_col = None

for col in df.columns:

    c = col.lower()

    if c in ["qty", "quantity"]:
        qty_col = col

    elif c in ["price", "unitprice"]:
        price_col = col

    elif c == "date":
        date_col = col

# ==================================================
# VALIDATION
# ==================================================

if qty_col is None or price_col is None:

    st.error("Required columns (Qty, Price) not found")
    st.stop()

# ==================================================
# CLEAN DATA
# ==================================================

df[qty_col] = pd.to_numeric(df[qty_col], errors="coerce")
df[price_col] = pd.to_numeric(df[price_col], errors="coerce")

df = df.dropna(subset=[qty_col, price_col])

# ==================================================
# CREATE TARGET VARIABLE
# ==================================================

df["Revenue"] = df[qty_col] * df[price_col]

# ==================================================
# FEATURE ENGINEERING
# ==================================================

df["Qty"] = df[qty_col]
df["Price"] = df[price_col]

features = ["Qty", "Price"]

X = df[features]
y = df["Revenue"]

# FINAL CLEAN (IMPORTANT FIX)
X = X.replace([np.inf, -np.inf], np.nan).dropna()
y = y.loc[X.index]

# ==================================================
# TRAIN TEST SPLIT
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==================================================
# MODEL SELECTION
# ==================================================

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# ==================================================
# PREDICTIONS
# ==================================================

y_pred = model.predict(X_test)

# ==================================================
# METRICS
# ==================================================

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# ==================================================
# DISPLAY RESULTS
# ==================================================

st.subheader("📊 Model Performance")

col1, col2 = st.columns(2)

col1.metric("MAE", f"{mae:.2f}")
col2.metric("R² Score", f"{r2:.2f}")

# ==================================================
# FEATURE IMPORTANCE
# ==================================================

st.subheader("🔥 Feature Importance")

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

st.dataframe(importance.sort_values("Importance", ascending=False))

# ==================================================
# PREDICTION SIMULATOR
# ==================================================

st.subheader("🎯 Try Prediction")

qty_input = st.number_input("Quantity", min_value=1, value=1)
price_input = st.number_input("Price", min_value=0.0, value=10.0)

if st.button("Predict Revenue"):

    pred = model.predict([[qty_input, price_input]])

    st.success(f"Predicted Revenue: ${pred[0]:.2f}")

# ==================================================
# RAW DATA
# ==================================================

with st.expander("View Dataset"):

    st.dataframe(df.head(100), use_container_width=True)
