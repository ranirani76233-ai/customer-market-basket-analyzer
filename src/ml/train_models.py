"""
train_models.py

Model Training Module

Models:
--------
1. Linear Regression
2. Ridge Regression
3. Lasso Regression
4. KNN Regressor
5. Random Forest Regressor
6. Gradient Boosting Regressor
7. Extra Trees Regressor

Outputs:
---------
- Trained Models
- Predictions
- Train/Test Sets
"""

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso
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
# TRAIN TEST SPLIT
# =====================================================

def split_data(
    X,
    y,
    test_size=0.20,
    random_state=42
):
    """
    Split data into train/test
    """

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )


# =====================================================
# MODEL DEFINITIONS
# =====================================================

def get_models():
    """
    Returns dictionary of models
    """

    models = {

        # ---------------------
        # Linear Models
        # ---------------------

        "Linear Regression":

        Pipeline([

            ("scaler",
             StandardScaler()),

            ("model",
             LinearRegression())

        ]),

        "Ridge Regression":

        Pipeline([

            ("scaler",
             StandardScaler()),

            ("model",
             Ridge(alpha=1.0))

        ]),

        "Lasso Regression":

        Pipeline([

            ("scaler",
             StandardScaler()),

            ("model",
             Lasso(alpha=0.1))

        ]),

        # ---------------------
        # KNN
        # ---------------------

        "KNN":

        Pipeline([

            ("scaler",
             StandardScaler()),

            ("model",
             KNeighborsRegressor(
                 n_neighbors=5
             ))

        ]),

        # ---------------------
        # Tree Models
        # ---------------------

        "Random Forest":

        RandomForestRegressor(

            n_estimators=300,

            random_state=42,

            n_jobs=-1

        ),

        "Gradient Boosting":

        GradientBoostingRegressor(

            n_estimators=300,

            learning_rate=0.05,

            random_state=42

        ),

        "Extra Trees":

        ExtraTreesRegressor(

            n_estimators=300,

            random_state=42,

            n_jobs=-1

        )
    }

    return models


# =====================================================
# TRAIN SINGLE MODEL
# =====================================================

def train_single_model(
    model,
    X_train,
    y_train
):
    """
    Train one model
    """

    model.fit(
        X_train,
        y_train
    )

    return model


# =====================================================
# TRAIN ALL MODELS
# =====================================================

def train_all_models(
    X_train,
    y_train
):
    """
    Train all models
    """

    models = get_models()

    trained_models = {}

    for model_name, model in models.items():

        print(
            f"Training {model_name}..."
        )

        model.fit(
            X_train,
            y_train
        )

        trained_models[
            model_name
        ] = model

    return trained_models


# =====================================================
# GENERATE PREDICTIONS
# =====================================================

def generate_predictions(
    trained_models,
    X_test
):
    """
    Generate predictions
    for all models
    """

    predictions = {}

    for model_name, model in (

        trained_models.items()

    ):

        predictions[
            model_name
        ] = model.predict(
            X_test
        )

    return predictions


# =====================================================
# TRAINING SUMMARY
# =====================================================

def training_summary(
    trained_models
):
    """
    Model overview
    """

    summary = pd.DataFrame({

        "Model":

        list(
            trained_models.keys()
        ),

        "Status":

        [
            "Trained"
        ] * len(
            trained_models
        )
    })

    return summary


# =====================================================
# COMPLETE TRAINING PIPELINE
# =====================================================

def train_pipeline(
    X,
    y,
    test_size=0.20,
    random_state=42
):
    """
    Full training workflow
    """

    X_train, X_test, y_train, y_test = (

        split_data(

            X,

            y,

            test_size,

            random_state

        )

    )

    trained_models = train_all_models(

        X_train,

        y_train

    )

    predictions = generate_predictions(

        trained_models,

        X_test

    )

    return {

        "X_train":
        X_train,

        "X_test":
        X_test,

        "y_train":
        y_train,

        "y_test":
        y_test,

        "models":
        trained_models,

        "predictions":
        predictions
    }


# =====================================================
# FEATURE LIST
# =====================================================

def get_feature_names(
    X
):

    if isinstance(
        X,
        pd.DataFrame
    ):
        return X.columns.tolist()

    return None


# =====================================================
# MAIN TEST
# =====================================================

if __name__ == "__main__":

    from sklearn.datasets import (
        make_regression
    )

    X, y = make_regression(

        n_samples=1000,

        n_features=15,

        noise=10,

        random_state=42

    )

    X = pd.DataFrame(X)

    results = train_pipeline(
        X,
        y
    )

    print("\nTraining Summary")

    print(

        training_summary(

            results["models"]

        )

    )

    print("\nModels Trained")

    print(

        list(
            results["models"].keys()
        )

    )
