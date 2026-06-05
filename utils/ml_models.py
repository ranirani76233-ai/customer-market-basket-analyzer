# utils/ml_models.py

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


class MLModels:

    def __init__(self):
        self.models = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(
                n_estimators=100,
                random_state=42
            ),
            "Gradient Boosting": GradientBoostingRegressor(
                random_state=42
            )
        }

    def prepare_data(self, df, target_column):

        data = df.copy()

        numeric_columns = data.select_dtypes(
            include=["int64", "float64"]
        ).columns

        data = data[numeric_columns]

        X = data.drop(columns=[target_column])

        y = data[target_column]

        return train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

    def train_models(self, df, target_column):

        X_train, X_test, y_train, y_test = self.prepare_data(
            df,
            target_column
        )

        results = []

        best_model = None
        best_score = -999

        for model_name, model in self.models.items():

            model.fit(X_train, y_train)

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

            results.append({
                "Model": model_name,
                "MSE": round(mse, 4),
                "RMSE": round(rmse, 4),
                "MAE": round(mae, 4),
                "R2 Score": round(r2, 4)
            })

            if r2 > best_score:
                best_score = r2
                best_model = model

        results_df = pd.DataFrame(results)

        return best_model, results_df

    def save_model(
        self,
        model,
        path="models/best_model.pkl"
    ):

        joblib.dump(model, path)

    def load_model(
        self,
        path="models/best_model.pkl"
    ):

        return joblib.load(path)

    def predict(
        self,
        model,
        input_data
    ):

        prediction = model.predict(input_data)

        return prediction

    def feature_importance(
        self,
        model,
        feature_names
    ):

        if hasattr(model, "feature_importances_"):

            importance = pd.DataFrame({
                "Feature": feature_names,
                "Importance": model.feature_importances_
            })

            return importance.sort_values(
                "Importance",
                ascending=False
            )

        return pd.DataFrame()
