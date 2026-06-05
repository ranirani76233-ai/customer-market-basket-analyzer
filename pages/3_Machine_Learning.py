import streamlit as st

st.title(
    "🤖 Machine Learning Models"
)

st.markdown(
    """
    Compare Multiple Algorithms
    """
)

model = st.selectbox(

    "Select Model",

    [

        "Linear Regression",

        "KNN",

        "Random Forest",

        "Gradient Boosting"

    ]
)

if st.button(
    "Train Model"
):

    st.success(
        f"{model} Trained Successfully"
    )
