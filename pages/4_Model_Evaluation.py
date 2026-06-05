import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor
)

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------

st.set_page_config(
    page_title="Model Evaluation",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Machine Learning Model Evaluation")

st.markdown(
    "Compare multiple machine learning models and select the best performer."
)

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload CSV Dataset",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # ---------------------------------------------
    # TARGET COLUMN
    # ---------------------------------------------

    target = st.selectbox(
        "Select Target Column",
        df.columns
    )

    X = df.drop(columns=[target])

    # Encode categorical columns
    X = pd.get_dummies(
        X,
        drop_first=True
    )

    y = df[target]

    # ---------------------------------------------
    # TRAIN TEST SPLIT
    # ---------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # ---------------------------------------------
    # MODELS
    # ---------------------------------------------

    models = {

        "Linear Regression":
        LinearRegression(),

        "KNN":
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

    if st.button("🚀 Evaluate Models"):

        results = []

        progress = st.progress(0)

        for i, (name, model) in enumerate(models.items()):

            model.fit(
                X_train,
                y_train
            )

            predictions = model.predict(
                X_test
            )

            mse = mean_squared_error(
                y_test,
                predictions
            )

            rmse = np.sqrt(mse)

            mae = mean_absolute_error(
                y_test,
                predictions
            )

            r2 = r2_score(
                y_test,
                predictions
            )

            results.append({

                "Model": name,

                "MSE": round(mse, 3),

                "RMSE": round(rmse, 3),

                "MAE": round(mae, 3),

                "R² Score": round(r2, 3)
            })

            progress.progress(
                (i + 1) / len(models)
            )

        results_df = pd.DataFrame(results)

        st.subheader("📊 Model Comparison")

        st.dataframe(
            results_df,
            use_container_width=True
        )

        # -----------------------------------------
        # BEST MODEL
        # -----------------------------------------

        best_model = results_df.sort_values(
            "R² Score",
            ascending=False
        ).iloc[0]

        st.success(
            f"""
            🏆 Best Model:
            {best_model['Model']}

            R² Score:
            {best_model['R² Score']}
            """
        )

        # -----------------------------------------
        # CHARTS
        # -----------------------------------------

        st.subheader("📈 R² Score Comparison")

        st.bar_chart(
            results_df.set_index(
                "Model"
            )["R² Score"]
        )

        st.subheader("📉 RMSE Comparison")

        st.bar_chart(
            results_df.set_index(
                "Model"
            )["RMSE"]
        )

        # -----------------------------------------
        # DOWNLOAD RESULTS
        # -----------------------------------------

        csv = results_df.to_csv(
            index=False
        )

        st.download_button(
            "⬇ Download Results",
            csv,
            "model_evaluation_results.csv",
            "text/csv"
        )

else:

    st.info(
        "Upload a CSV dataset to start model evaluation."
    )
