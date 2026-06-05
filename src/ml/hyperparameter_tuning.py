"""
hyperparameter_tuning.py

Hyperparameter Optimization Module

Models:
---------
1. KNN Regressor
2. Random Forest Regressor
3. Gradient Boosting Regressor
4. Extra Trees Regressor

Methods:
---------
- GridSearchCV
- RandomizedSearchCV
"""

import pandas as pd

from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV
)

from sklearn.neighbors import (
    KNeighborsRegressor
)

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor
)


# =====================================================
# KNN TUNING
# =====================================================

def tune_knn(
    X_train,
    y_train,
    cv=5
):

    params = {

        "n_neighbors": [3, 5, 7, 9, 11],

        "weights": [
            "uniform",
            "distance"
        ],

        "p": [1, 2]

    }

    grid = GridSearchCV(

        estimator=KNeighborsRegressor(),

        param_grid=params,

        cv=cv,

        scoring="r2",

        n_jobs=-1

    )

    grid.fit(
        X_train,
        y_train
    )

    return {

        "model":
        grid.best_estimator_,

        "best_params":
        grid.best_params_,

        "best_score":
        round(
            grid.best_score_,
            4
        )

    }


# =====================================================
# RANDOM FOREST TUNING
# =====================================================

def tune_random_forest(
    X_train,
    y_train,
    cv=5
):

    params = {

        "n_estimators":
        [100, 200, 300],

        "max_depth":
        [10, 20, 30, None],

        "min_samples_split":
        [2, 5, 10],

        "min_samples_leaf":
        [1, 2, 4]

    }

    grid = RandomizedSearchCV(

        estimator=RandomForestRegressor(
            random_state=42
        ),

        param_distributions=params,

        n_iter=20,

        cv=cv,

        scoring="r2",

        random_state=42,

        n_jobs=-1

    )

    grid.fit(
        X_train,
        y_train
    )

    return {

        "model":
        grid.best_estimator_,

        "best_params":
        grid.best_params_,

        "best_score":
        round(
            grid.best_score_,
            4
        )

    }


# =====================================================
# GRADIENT BOOSTING TUNING
# =====================================================

def tune_gradient_boosting(
    X_train,
    y_train,
    cv=5
):

    params = {

        "n_estimators":
        [100, 200, 300],

        "learning_rate":
        [0.01, 0.05, 0.1],

        "max_depth":
        [3, 5, 7],

        "subsample":
        [0.8, 1.0]

    }

    grid = RandomizedSearchCV(

        estimator=GradientBoostingRegressor(
            random_state=42
        ),

        param_distributions=params,

        n_iter=20,

        cv=cv,

        scoring="r2",

        random_state=42,

        n_jobs=-1

    )

    grid.fit(
        X_train,
        y_train
    )

    return {

        "model":
        grid.best_estimator_,

        "best_params":
        grid.best_params_,

        "best_score":
        round(
            grid.best_score_,
            4
        )

    }


# =====================================================
# EXTRA TREES TUNING
# =====================================================

def tune_extra_trees(
    X_train,
    y_train,
    cv=5
):

    params = {

        "n_estimators":
        [100, 200, 300],

        "max_depth":
        [10, 20, 30, None],

        "min_samples_split":
        [2, 5, 10]

    }

    grid = RandomizedSearchCV(

        estimator=ExtraTreesRegressor(
            random_state=42
        ),

        param_distributions=params,

        n_iter=20,

        cv=cv,

        scoring="r2",

        random_state=42,

        n_jobs=-1

    )

    grid.fit(
        X_train,
        y_train
    )

    return {

        "model":
        grid.best_estimator_,

        "best_params":
        grid.best_params_,

        "best_score":
        round(
            grid.best_score_,
            4
        )

    }


# =====================================================
# TUNE ALL MODELS
# =====================================================

def tune_all_models(
    X_train,
    y_train
):

    results = {}

    print("Tuning KNN...")
    results["KNN"] = tune_knn(
        X_train,
        y_train
    )

    print("Tuning Random Forest...")
    results["Random Forest"] = (
        tune_random_forest(
            X_train,
            y_train
        )
    )

    print("Tuning Gradient Boosting...")
    results["Gradient Boosting"] = (
        tune_gradient_boosting(
            X_train,
            y_train
        )
    )

    print("Tuning Extra Trees...")
    results["Extra Trees"] = (
        tune_extra_trees(
            X_train,
            y_train
        )
    )

    return results


# =====================================================
# TUNING SUMMARY
# =====================================================

def tuning_summary(
    tuning_results
):

    summary = []

    for model_name, result in (
        tuning_results.items()
    ):

        summary.append({

            "Model":
            model_name,

            "Best Score":
            result["best_score"],

            "Best Parameters":
            str(
                result["best_params"]
            )

        })

    return pd.DataFrame(summary)


# =====================================================
# BEST TUNED MODEL
# =====================================================

def get_best_tuned_model(
    tuning_results
):

    best_model_name = max(

        tuning_results,

        key=lambda x:
        tuning_results[x]["best_score"]

    )

    return {

        "Model":
        best_model_name,

        "Estimator":
        tuning_results[
            best_model_name
        ]["model"],

        "Score":
        tuning_results[
            best_model_name
        ]["best_score"]

    }


# =====================================================
# SAVE TUNING RESULTS
# =====================================================

def save_tuning_results(
    tuning_results,
    filepath
):

    summary = tuning_summary(
        tuning_results
    )

    summary.to_csv(
        filepath,
        index=False
    )

    print(
        f"Saved to {filepath}"
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

    X, y = make_regression(

        n_samples=1000,

        n_features=10,

        noise=10,

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

    tuning_results = tune_all_models(

        X_train,

        y_train

    )

    summary = tuning_summary(
        tuning_results
    )

    print(summary)

    print("\nBest Tuned Model")

    print(
        get_best_tuned_model(
            tuning_results
        )
    )
