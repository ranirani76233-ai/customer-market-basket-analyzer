"""
predict.py

Prediction Module

Responsibilities
----------------
1. Load Saved Model
2. Single Prediction
3. Batch Prediction
4. Add Predictions to DataFrame
5. Save Prediction Results
"""

import pandas as pd
import joblib


# =====================================================
# LOAD MODEL
# =====================================================

def load_model(model_path):
    """
    Load saved model
    """

    model = joblib.load(model_path)

    return model


# =====================================================
# SINGLE CUSTOMER PREDICTION
# =====================================================

def predict_single_customer(
    model,
    customer_features
):
    """
    Predict target value for one customer

    customer_features:
    list or numpy array
    """

    prediction = model.predict(
        [customer_features]
    )[0]

    return round(
        float(prediction),
        2
    )


# =====================================================
# BATCH PREDICTION
# =====================================================

def predict_batch(
    model,
    X
):
    """
    Predict multiple records

    Parameters
    ----------
    X : DataFrame
    """

    predictions = model.predict(X)

    return predictions


# =====================================================
# APPEND PREDICTIONS
# =====================================================

def add_predictions(
    df,
    predictions,
    column_name="Predicted_Spend"
):
    """
    Add predictions back to dataframe
    """

    result_df = df.copy()

    result_df[column_name] = predictions

    return result_df


# =====================================================
# CUSTOMER SEGMENTATION
# =====================================================

def create_prediction_segments(
    df,
    prediction_column="Predicted_Spend"
):
    """
    Segment customers based on prediction
    """

    segmented_df = df.copy()

    segmented_df["Predicted_Segment"] = pd.qcut(

        segmented_df[
            prediction_column
        ],

        q=4,

        labels=[
            "Low Value",
            "Medium Value",
            "High Value",
            "Premium"
        ]

    )

    return segmented_df


# =====================================================
# TOP PREDICTED CUSTOMERS
# =====================================================

def get_top_customers(
    prediction_df,
    n=10,
    prediction_column="Predicted_Spend"
):
    """
    Highest predicted customers
    """

    return (

        prediction_df

        .sort_values(
            by=prediction_column,
            ascending=False
        )

        .head(n)

    )


# =====================================================
# SAVE PREDICTIONS
# =====================================================

def save_predictions(
    prediction_df,
    output_path
):
    """
    Export predictions
    """

    prediction_df.to_csv(
        output_path,
        index=False
    )

    print(
        f"Predictions saved to {output_path}"
    )


# =====================================================
# PREDICTION SUMMARY
# =====================================================

def prediction_summary(
    prediction_df,
    prediction_column="Predicted_Spend"
):
    """
    Prediction statistics
    """

    summary = {

        "Average Prediction":
        round(
            prediction_df[
                prediction_column
            ].mean(),
            2
        ),

        "Maximum Prediction":
        round(
            prediction_df[
                prediction_column
            ].max(),
            2
        ),

        "Minimum Prediction":
        round(
            prediction_df[
                prediction_column
            ].min(),
            2
        ),

        "Total Predicted Revenue":
        round(
            prediction_df[
                prediction_column
            ].sum(),
            2
        )

    }

    return summary


# =====================================================
# MODEL INFERENCE PIPELINE
# =====================================================

def run_prediction_pipeline(
    model_path,
    X,
    original_df
):
    """
    End-to-end prediction workflow
    """

    model = load_model(
        model_path
    )

    predictions = predict_batch(
        model,
        X
    )

    prediction_df = add_predictions(

        original_df,

        predictions,

        "Predicted_Spend"

    )

    prediction_df = (
        create_prediction_segments(
            prediction_df
        )
    )

    return prediction_df


# =====================================================
# STREAMLIT HELPER
# =====================================================

def predict_from_streamlit_input(
    model,
    total_orders,
    total_items,
    total_spend,
    unique_products,
    avg_order_value,
    recency,
    frequency,
    monetary
):
    """
    Predict using Streamlit form values
    """

    features = [[

        total_orders,
        total_items,
        total_spend,
        unique_products,
        avg_order_value,
        recency,
        frequency,
        monetary

    ]]

    prediction = model.predict(
        features
    )[0]

    return round(
        float(prediction),
        2
    )


# =====================================================
# MAIN TEST
# =====================================================

if __name__ == "__main__":

    import numpy as np
    from sklearn.ensemble import RandomForestRegressor

    X = pd.DataFrame({

        "Orders": [1, 2, 3, 4, 5],

        "Spend": [100, 200, 300, 400, 500]

    })

    y = [120, 250, 350, 470, 580]

    model = RandomForestRegressor()

    model.fit(X, y)

    joblib.dump(
        model,
        "temp_model.pkl"
    )

    loaded_model = load_model(
        "temp_model.pkl"
    )

    preds = predict_batch(
        loaded_model,
        X
    )

    result = add_predictions(
        X,
        preds
    )

    print(result.head())

    print(
        prediction_summary(
            result
        )
    )
