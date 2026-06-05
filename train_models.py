import os
import joblib
import warnings
import pandas as pd
import numpy as np

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    cross_val_score
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

warnings.filterwarnings("ignore")

# =====================================================
# CONFIG
# =====================================================

DATA_PATH = "data/Assignment-1_Data.csv"

MODELS_DIR = "models"

REPORTS_DIR = "reports"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# =====================================================
# LOAD DATA
# =====================================================

print("\nLoading Dataset...")

df = pd.read_csv(
    DATA_PATH,
    encoding="ISO-8859-1"
)

# =====================================================
# DATA CLEANING
# =====================================================

print("Cleaning Data...")

df = df.drop_duplicates()

if "Qty" in df.columns:
    df = df[df["Qty"] > 0]

if "Price" in df.columns:
    df = df[df["Price"] > 0]

df["Revenue"] = (
    df["Qty"] *
    df["Price"]
)

# =====================================================
# FEATURES
# =====================================================

print("Preparing Features...")

X = df[[
    "Qty",
    "Price"
]]

y = df["Revenue"]

# =====================================================
# TRAIN TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

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
# TRAINING
# =====================================================

results = []

best_model = None
best_score = -999
best_model_name = None

print("\nTraining Models...\n")

for model_name, model in models.items():

    print(f"Training {model_name}")

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

        scoring="r2"

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

    if r2 > best_score:

        best_score = r2

        best_model = model

        best_model_name = model_name

# =====================================================
# MODEL COMPARISON
# =====================================================

results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(
    "R2",
    ascending=False
)

print("\nModel Comparison\n")

print(results_df)

results_df.to_csv(

    os.path.join(
        REPORTS_DIR,
        "model_comparison.csv"
    ),

    index=False

)

# =====================================================
# HYPERPARAMETER TUNING
# =====================================================

print("\nRunning Hyperparameter Tuning...")

rf = RandomForestRegressor(
    random_state=42
)

param_grid = {

    "n_estimators":
    [100, 200, 300],

    "max_depth":
    [5, 10, 20, None],

    "min_samples_split":
    [2, 5, 10],

    "min_samples_leaf":
    [1, 2, 4]

}

grid_search = GridSearchCV(

    estimator=rf,

    param_grid=param_grid,

    cv=3,

    scoring="r2",

    n_jobs=-1

)

grid_search.fit(
    X_train,
    y_train
)

best_rf = grid_search.best_estimator_

rf_predictions = best_rf.predict(
    X_test
)

rf_r2 = r2_score(
    y_test,
    rf_predictions
)

rf_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        rf_predictions
    )
)

print("\nBest Parameters")

print(grid_search.best_params_)

print(
    f"Tuned R²: {rf_r2:.4f}"
)

print(
    f"Tuned RMSE: {rf_rmse:.4f}"
)

# =====================================================
# SAVE TUNING REPORT
# =====================================================

tuning_report = pd.DataFrame({

    "Parameter":
    list(
        grid_search.best_params_.keys()
    ),

    "Value":
    list(
        grid_search.best_params_.values()
    )

})

tuning_report.to_csv(

    os.path.join(
        REPORTS_DIR,
        "tuning_results.csv"
    ),

    index=False

)

# =====================================================
# SAVE BEST MODEL
# =====================================================

joblib.dump(

    best_model,

    os.path.join(
        MODELS_DIR,
        "best_model.pkl"
    )

)

joblib.dump(

    best_rf,

    os.path.join(
        MODELS_DIR,
        "tuned_random_forest.pkl"
    )

)

# =====================================================
# FEATURE IMPORTANCE
# =====================================================

importance_df = pd.DataFrame({

    "Feature":
    X.columns,

    "Importance":
    best_rf.feature_importances_

})

importance_df.to_csv(

    os.path.join(
        REPORTS_DIR,
        "feature_importance.csv"
    ),

    index=False

)

# =====================================================
# SUMMARY
# =====================================================

print("\nTraining Completed")

print(
    f"\nBest Base Model: {best_model_name}"
)

print(
    f"Best R² Score: {best_score:.4f}"
)

print(
    "\nSaved Files:"
)

print(
    "models/best_model.pkl"
)

print(
    "models/tuned_random_forest.pkl"
)

print(
    "reports/model_comparison.csv"
)

print(
    "reports/tuning_results.csv"
)

print(
    "reports/feature_importance.csv"
)
