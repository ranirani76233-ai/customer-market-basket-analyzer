import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Model Evaluation",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Model Evaluation Dashboard")

st.markdown(
    "Evaluate machine learning model performance."
)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload Evaluation Dataset",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    target = st.selectbox(
        "Select Target Column",
        df.columns
    )

    try:

        model = joblib.load(
            "models/best_model.pkl"
        )

        st.success(
            "Model Loaded Successfully"
        )

    except Exception as e:

        st.error(
            "Could not load model.\n"
            "Make sure models/best_model.pkl exists."
        )

        st.stop()

    # --------------------------------------------------
    # PREPARE DATA
    # --------------------------------------------------

    X = df.drop(
        columns=[target]
    )

    X = pd.get_dummies(
        X,
        drop_first=True
    )

    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # --------------------------------------------------
    # PREDICTIONS
    # --------------------------------------------------

    predictions = model.predict(X_test)

    # --------------------------------------------------
    # METRICS
    # --------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    st.subheader("📊 Performance Metrics")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Accuracy",
        f"{accuracy:.3f}"
    )

    col2.metric(
        "Precision",
        f"{precision:.3f}"
    )

    col3.metric(
        "Recall",
        f"{recall:.3f}"
    )

    col4.metric(
        "F1 Score",
        f"{f1:.3f}"
    )

    # --------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------

    st.subheader("🔥 Confusion Matrix")

    cm = confusion_matrix(
        y_test,
        predictions
    )

    fig, ax = plt.subplots()

    im = ax.imshow(cm)

    plt.colorbar(im)

    ax.set_xlabel("Predicted")

    ax.set_ylabel("Actual")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center"
            )

    st.pyplot(fig)

    # --------------------------------------------------
    # CLASSIFICATION REPORT
    # --------------------------------------------------

    st.subheader("📋 Classification Report")

    report = classification_report(
        y_test,
        predictions,
        output_dict=True
    )

    st.dataframe(
        pd.DataFrame(report).transpose()
    )

    # --------------------------------------------------
    # ROC CURVE
    # --------------------------------------------------

    if hasattr(model, "predict_proba"):

        st.subheader("📈 ROC Curve")

        try:

            y_prob = model.predict_proba(
                X_test
            )[:, 1]

            fpr, tpr, _ = roc_curve(
                y_test,
                y_prob
            )

            roc_auc = auc(
                fpr,
                tpr
            )

            fig2, ax2 = plt.subplots()

            ax2.plot(
                fpr,
                tpr,
                label=f"AUC = {roc_auc:.3f}"
            )

            ax2.plot(
                [0, 1],
                [0, 1],
                linestyle="--"
            )

            ax2.set_xlabel(
                "False Positive Rate"
            )

            ax2.set_ylabel(
                "True Positive Rate"
            )

            ax2.legend()

            st.pyplot(fig2)

        except:
            pass

    # --------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------

    if hasattr(model, "feature_importances_"):

        st.subheader("⭐ Feature Importance")

        importance_df = pd.DataFrame({

            "Feature":
            X.columns,

            "Importance":
            model.feature_importances_
        })

        importance_df = (
            importance_df
            .sort_values(
                "Importance",
                ascending=False
            )
            .head(15)
        )

        st.dataframe(
            importance_df
        )

        fig3, ax3 = plt.subplots()

        ax3.barh(
            importance_df["Feature"],
            importance_df["Importance"]
        )

        ax3.invert_yaxis()

        st.pyplot(fig3)

else:

    st.info(
        "Upload a dataset to evaluate the trained model."
    )
