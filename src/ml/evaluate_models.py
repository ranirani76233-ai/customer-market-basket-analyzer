"""
evaluate_models.py

Model Evaluation Module

Metrics:
---------
1. MSE
2. RMSE
3. MAE
4. R² Score
5. Cross Validation Score

Supports:
---------
- Linear Regression
- KNN Regression
- Random Forest
- Gradient Boosting
- XGBoost
"""

import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

from sklearn.model_selection import (
    cross_val_score
)


# =====================================================
# SINGLE MODEL EVALUATION
# =====================================================

def evaluate_model(
    model,
    X_test,
    y_test
):
    """
    Evaluate one trained model
    """

    predictions = model.predict(X_test)

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

    metrics = {

        "MSE": round(mse, 4),

        "RMSE": round(rmse, 4),

        "MAE": round(mae, 4),

        "R2": round(r2, 4)

    }

    return metrics


# =====================================================
# CROSS VALIDATION
# =====================================================

def calculate_cv_score(
    model,
    X,
    y,
    cv=5
):
    """
    Calculate Cross Validation R²
    """

    scores = cross_val_score(

        model,

        X,

        y,

        cv=cv,

        scoring="r2",

        n_jobs=-1
    )

    return {

        "CV Mean R2":
        round(scores.mean(), 4),

        "CV Std":
        round(scores.std(), 4)
    }


# =====================================================
# EVALUATE MULTIPLE MODELS
# =====================================================

def evaluate_multiple_models(
    models,
    X_train,
    X_test,
    y_train,
    y_test
):
    """
    Evaluate all models
    """

    results = []

    for model_name, model in models.items():

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

        cv_score = cross_val_score(

            model,

            X_train,

            y_train,

            cv=5,

            scoring="r2",

            n_jobs=-1

        ).mean()

        results.append({

            "Model":
            model_name,

            "MSE":
            round(mse, 4),

            "RMSE":
            round(rmse, 4),

            "MAE":
            round(mae, 4),

            "R2":
            round(r2, 4),

            "CV_R2":
            round(cv_score, 4)

        })

    results_df = pd.DataFrame(
        results
    )

    return results_df


# =====================================================
# MODEL RANKING
# =====================================================

def rank_models(
    results_df
):
    """
    Rank models by:
    Highest R²
    Highest CV_R2
    Lowest RMSE
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
# BEST MODEL
# =====================================================

def get_best_model(
    models,
    results_df
):
    """
    Return best model object
    """

    best_model_name = (

        results_df
        .sort_values(
            by="R2",
            ascending=False
        )
        .iloc[0]["Model"]

    )

    return models[
        best_model_name
    ]


# =====================================================
# BEST MODEL SUMMARY
# =====================================================

def best_model_summary(
    results_df
):

    best = (

        results_df

        .sort_values(
            by="R2",
            ascending=False
        )

        .iloc[0]

    )

    return {

        "Model":
        best["Model"],

        "R2":
        best["R2"],

        "RMSE":
        best["RMSE"],

        "MAE":
        best["MAE"]

    }


# =====================================================
# PERFORMANCE INTERPRETATION
# =====================================================

def interpret_r2(
    r2
):
    """
    Interpret R² score
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
        return "Needs Improvement"


# =====================================================
# EXPORT RESULTS
# =====================================================

def export_results(
    results_df,
    filepath
):

    results_df.to_csv(
        filepath,
        index=False
    )

    print(
        f"Results saved to {filepath}"
    )


# =====================================================
# MAIN TEST
# =====================================================

if __name__ == "__main__":

    from sklearn.datasets import (
        make_regression
    )

    from sklearn.model_selection import (
        train_test_split
    )

    from sklearn.linear_model import (
        LinearRegression
    )

    from sklearn.neighbors import (
        KNeighborsRegressor
    )

    from sklearn.ensemble import (
        RandomForestRegressor,
        GradientBoostingRegressor
    )

    X, y = make_regression(

        n_samples=1000,

        n_features=10,

        noise=20,

        random_state=42
    )

    X_train, X_test, y_train, y_test = (

        train_test_split(

            X,

            y,

            test_size=0.2,

            random_state=42

        )

    )

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
        )
    }

    results = evaluate_multiple_models(

        models,

        X_train,

        X_test,

        y_train,

        y_test
    )

    print("\nModel Comparison")
    print(results)

    print("\nRanking")
    print(rank_models(results))

    print("\nBest Model")
    print(best_model_summary(results))
