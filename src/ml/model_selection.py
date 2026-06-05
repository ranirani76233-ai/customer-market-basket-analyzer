"""
model_selection.py

Model Selection Module

Responsibilities:
-----------------
1. Rank Models
2. Select Best Model
3. Save Best Model
4. Load Saved Model
5. Generate Summary
"""

import pandas as pd
import joblib


# =====================================================
# RANK MODELS
# =====================================================

def rank_models(results_df):
    """
    Rank models using:

    1. Highest R²
    2. Highest CV_R2
    3. Lowest RMSE
    """

    ranking = (

        results_df

        .sort_values(

            by=[
                "R2",
                "CV_R2",
                "RMSE"
            ],

            ascending=[
                False,
                False,
                True
            ]

        )

        .reset_index(
            drop=True
        )

    )

    ranking.index += 1

    ranking.index.name = "Rank"

    return ranking


# =====================================================
# GET BEST MODEL NAME
# =====================================================

def get_best_model_name(
    results_df
):
    """
    Returns best model name
    """

    ranking = rank_models(
        results_df
    )

    return ranking.iloc[0]["Model"]


# =====================================================
# GET BEST MODEL OBJECT
# =====================================================

def get_best_model(
    trained_models,
    results_df
):
    """
    Returns best trained model
    """

    best_model_name = (
        get_best_model_name(
            results_df
        )
    )

    return trained_models[
        best_model_name
    ]


# =====================================================
# BEST MODEL SUMMARY
# =====================================================

def best_model_summary(
    results_df
):

    ranking = rank_models(
        results_df
    )

    best = ranking.iloc[0]

    summary = {

        "Model":
        best["Model"],

        "R2":
        best["R2"],

        "CV_R2":
        best["CV_R2"],

        "RMSE":
        best["RMSE"],

        "MAE":
        best["MAE"]

    }

    return summary


# =====================================================
# SAVE MODEL
# =====================================================

def save_model(
    model,
    filepath
):
    """
    Save trained model
    """

    joblib.dump(
        model,
        filepath
    )

    print(
        f"Model saved at {filepath}"
    )


# =====================================================
# LOAD MODEL
# =====================================================

def load_model(
    filepath
):
    """
    Load saved model
    """

    return joblib.load(
        filepath
    )


# =====================================================
# SAVE BEST MODEL
# =====================================================

def save_best_model(
    trained_models,
    results_df,
    filepath="models/best_model.pkl"
):
    """
    Automatically save best model
    """

    best_model = get_best_model(
        trained_models,
        results_df
    )

    save_model(
        best_model,
        filepath
    )

    return best_model


# =====================================================
# SAVE MODEL COMPARISON
# =====================================================

def save_model_results(
    results_df,
    filepath="models/model_results.csv"
):
    """
    Save comparison results
    """

    results_df.to_csv(
        filepath,
        index=False
    )

    print(
        f"Results saved at {filepath}"
    )


# =====================================================
# MODEL PERFORMANCE LABEL
# =====================================================

def model_performance_label(
    r2
):
    """
    Human-readable performance
    """

    if r2 >= 0.90:
        return "Excellent"

    elif r2 >= 0.80:
        return "Very Good"

    elif r2 >= 0.70:
        return "Good"

    elif r2 >= 0.60:
        return "Moderate"

    else:
        return "Poor"


# =====================================================
# STREAMLIT SUMMARY CARD
# =====================================================

def generate_summary_card(
    results_df
):

    best = best_model_summary(
        results_df
    )

    return {

        "Best Model":
        best["Model"],

        "R²":
        best["R2"],

        "Cross Validation":
        best["CV_R2"],

        "RMSE":
        best["RMSE"],

        "Performance":
        model_performance_label(
            best["R2"]
        )
    }


# =====================================================
# TOP N MODELS
# =====================================================

def top_models(
    results_df,
    n=3
):
    """
    Returns top N models
    """

    ranking = rank_models(
        results_df
    )

    return ranking.head(n)


# =====================================================
# MAIN TEST
# =====================================================

if __name__ == "__main__":

    sample_results = pd.DataFrame({

        "Model": [

            "Linear Regression",
            "KNN",
            "Random Forest",
            "Gradient Boosting"

        ],

        "MSE": [
            140,
            120,
            80,
            75
        ],

        "RMSE": [
            11.83,
            10.95,
            8.94,
            8.66
        ],

        "MAE": [
            9.2,
            8.4,
            6.1,
            5.8
        ],

        "R2": [
            0.75,
            0.82,
            0.91,
            0.93
        ],

        "CV_R2": [
            0.74,
            0.80,
            0.90,
            0.92
        ]

    })

    print("\nModel Ranking")

    print(
        rank_models(
            sample_results
        )
    )

    print("\nBest Model")

    print(
        best_model_summary(
            sample_results
        )
    )

    print("\nSummary Card")

    print(
        generate_summary_card(
            sample_results
        )
    )
