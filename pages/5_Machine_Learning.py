import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso
)

from sklearn.neighbors import KNeighborsRegressor

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor
)

import joblib

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Machine Learning",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Machine Learning & Predictive Analytics")

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
# FEATURE ENGINEERING
# =====================================================

if (
    "Qty" in df.columns and
    "Price" in df.columns
):

    df["Revenue"] = (
        df["Qty"] * df["Price"]
    )

required_cols = [
    "Qty",
    "Price",
    "Revenue"
]

missing = [
    col for col in required_cols
    if col not in df.columns
]

if missing:

    st.error(
        f"Missing columns: {missing}"
    )

    st.stop()

# =====================================================
# FEATURES
# =====================================================

X = df[[
    "Qty",
    "Price"
]]

y = df["Revenue"]

# =====================================================
# TRAIN TEST SPLIT
# =====================================================

test_size = st.sidebar.slider(
    "Test Size",
    0.1,
    0.4,
    0.2
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=test_size,
    random_state=42
)

# =====================================================
# MODELS
# =====================================================

models = {

    "Linear Regression":
    LinearRegression(),

    "Ridge Regression":
    Ridge(),

    "Lasso Regression":
    Lasso(),

    "KNN Regressor":
    KNeighborsRegressor(),

    "Random Forest":
    RandomForestRegressor(
        random_state=42
    ),

    "Gradient Boosting":
    GradientBoostingRegressor(
        random_state=42
    ),

    "Extra Trees":
    ExtraTreesRegressor(
        random_state=42
    )

}

# =====================================================
# TRAIN MODELS
# =====================================================

results = []

predictions_dict = {}

for name, model in models.items():

    model.fit(
        X_train,
        y_train
    )

    preds = model.predict(
        X_test
    )

    mse = mean_squared_error(
        y_test,
        preds
    )

    rmse = np.sqrt(mse)

    mae = mean_absolute_error(
        y_test,
        preds
    )

    r2 = r2_score(
        y_test,
        preds
    )

    results.append({

        "Model": name,

        "MSE": round(mse, 2),

        "RMSE": round(rmse, 2),

        "MAE": round(mae, 2),

        "R2": round(r2, 4)

    })

    predictions_dict[name] = preds

results_df = pd.DataFrame(
    results
)

# =====================================================
# MODEL COMPARISON
# =====================================================

st.subheader("📊 Model Comparison")

st.dataframe(
    results_df.sort_values(
        "R2",
        ascending=False
    ),
    use_container_width=True
)

# =====================================================
# BEST MODEL
# =====================================================

best_model_name = (

    results_df

    .sort_values(
        "R2",
        ascending=False
    )

    .iloc[0]["Model"]

)

st.success(
    f"Best Model: {best_model_name}"
)

# =====================================================
# R2 COMPARISON
# =====================================================

fig = px.bar(

    results_df,

    x="Model",

    y="R2",

    title="R² Score Comparison"

)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =====================================================
# HYPERPARAMETER TUNING
# =====================================================

st.subheader("⚙ Hyperparameter Tuning")

if st.button(
    "Run Hyperparameter Tuning"
):

    rf = RandomForestRegressor(
        random_state=42
    )

    param_grid = {

        "n_estimators":
        [100, 200],

        "max_depth":
        [5, 10, 20],

        "min_samples_split":
        [2, 5]

    }

    grid = GridSearchCV(

        rf,

        param_grid,

        cv=3,

        scoring="r2",

        n_jobs=-1

    )

    grid.fit(
        X_train,
        y_train
    )

    best_rf = grid.best_estimator_

    preds = best_rf.predict(
        X_test
    )

    tuned_r2 = r2_score(
        y_test,
        preds
    )

    tuned_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            preds
        )
    )

    st.success(
        f"Best Parameters: {grid.best_params_}"
    )

    st.metric(
        "Tuned R²",
        round(tuned_r2, 4)
    )

    st.metric(
        "Tuned RMSE",
        round(tuned_rmse, 2)
    )

    joblib.dump(
        best_rf,
        "models/best_model.pkl"
    )

# =====================================================
# ACTUAL VS PREDICTED
# =====================================================

st.subheader("📈 Actual vs Predicted")

selected_model = st.selectbox(

    "Choose Model",

    list(models.keys())

)

preds = predictions_dict[
    selected_model
]

plot_df = pd.DataFrame({

    "Actual":
    y_test.values,

    "Predicted":
    preds

})

fig = px.scatter(

    plot_df,

    x="Actual",

    y="Predicted",

    title=f"{selected_model}"

)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =====================================================
# FEATURE IMPORTANCE
# =====================================================

st.subheader("🎯 Feature Importance")

rf_model = RandomForestRegressor(
    random_state=42
)

rf_model.fit(
    X_train,
    y_train
)

importance_df = pd.DataFrame({

    "Feature":
    X.columns,

    "Importance":
    rf_model.feature_importances_

})

fig = px.bar(

    importance_df,

    x="Feature",

    y="Importance",

    title="Feature Importance"

)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =====================================================
# PREDICTION SECTION
# =====================================================

st.subheader("🔮 Predict Revenue")

qty = st.number_input(
    "Quantity",
    min_value=1,
    value=1
)

price = st.number_input(
    "Price",
    min_value=0.0,
    value=10.0
)

if st.button(
    "Predict Revenue"
):

    best_model = RandomForestRegressor(
        random_state=42
    )

    best_model.fit(
        X_train,
        y_train
    )

    prediction = best_model.predict(
        [[qty, price]]
    )[0]

    st.success(
        f"Predicted Revenue = {prediction:.2f}"
    )

# =====================================================
# EXPORT RESULTS
# =====================================================

st.subheader("📥 Export Results")

csv = results_df.to_csv(
    index=False
).encode("utf-8")

st.download_button(

    "Download Model Results",

    csv,

    "model_comparison.csv",

    "text/csv"

)

# =====================================================
# SAVE BEST MODEL
# =====================================================

if st.button(
    "Save Best Model"
):

    best_model = RandomForestRegressor(
        random_state=42
    )

    best_model.fit(
        X_train,
        y_train
    )

    joblib.dump(
        best_model,
        "models/best_model.pkl"
    )

    st.success(
        "Model saved successfully."
    )
