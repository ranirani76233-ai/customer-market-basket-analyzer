# 🛒 Customer Market Basket Analytics & Machine Learning Dashboard

## 📌 Project Overview

Customer Market Basket Analytics is an end-to-end Data Analytics and Machine Learning application built using Streamlit. The project helps businesses understand customer purchasing behavior, discover product associations, analyze sales performance, and build predictive machine learning models.

The application combines:

* Customer Analytics Dashboard
* Market Basket Analysis (Apriori Algorithm)
* Machine Learning Model Training
* Model Evaluation & Comparison
* Interactive Data Visualization

---

## 🚀 Features

### 📊 Dashboard Analytics

* Total Sales Analysis
* Customer Insights
* Product Performance Analysis
* Statistical Summary
* Correlation Analysis
* Interactive Visualizations

### 🛒 Market Basket Analysis

* Apriori Algorithm
* Frequent Itemset Mining
* Association Rule Generation
* Support Analysis
* Confidence Analysis
* Lift Analysis
* Product Recommendations

### 🤖 Machine Learning

* Data Preprocessing
* Feature Engineering
* Model Training
* Hyperparameter Tuning
* Model Comparison
* Prediction Generation

### 📈 Model Evaluation

* Accuracy Score
* Precision
* Recall
* F1 Score
* Confusion Matrix
* ROC Curve
* Feature Importance Analysis

---

## 🏗️ Project Structure

```text
customer-market-basket-analytics/
│
├── app.py
│
├── pages/
│   ├── 1_Dashboard.py
│   ├── 2_Market_Basket_Analysis.py
│   ├── 3_Machine_Learning.py
│   └── 4_Model_Evaluation.py
│
├── data/
│   └── customer_transactions.csv
│
├── models/
│   └── best_model.pkl
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 📂 Dataset Format

Example dataset structure:

| CustomerID | BillNo | Itemname | Quantity | Price |
| ---------- | ------ | -------- | -------- | ----- |
| 1001       | B001   | Bread    | 2        | 40    |
| 1001       | B001   | Milk     | 1        | 25    |
| 1002       | B002   | Eggs     | 12       | 60    |
| 1003       | B003   | Butter   | 1        | 55    |

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/customer-market-basket-analytics.git

cd customer-market-basket-analytics
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run Application

```bash
streamlit run app.py
```

Application will launch locally at:

```text
http://localhost:8501
```

---

## 📦 Required Libraries

```text
streamlit
pandas
numpy
matplotlib
plotly
scikit-learn
mlxtend
joblib
```

---

## 🧠 Machine Learning Models

The project supports:

* Linear Regression
* K-Nearest Neighbors (KNN)
* Random Forest
* Gradient Boosting
* Extra Trees
* XGBoost (Optional)

---

## 📊 Market Basket Metrics

### Support

Measures how frequently an itemset appears in the dataset.

### Confidence

Measures the probability that a customer buys Product B when Product A is purchased.

### Lift

Measures the strength of the association between products.

* Lift > 1 → Positive Association
* Lift = 1 → No Association
* Lift < 1 → Negative Association

---

## 🎯 Business Benefits

* Increase Cross-Selling Opportunities
* Improve Product Placement Strategy
* Enhance Customer Experience
* Discover Hidden Buying Patterns
* Predict Future Customer Behavior
* Improve Revenue Generation

---

## 📈 Technologies Used

* Python
* Streamlit
* Pandas
* NumPy
* Scikit-Learn
* Mlxtend
* Plotly
* Matplotlib

---

## 🔮 Future Enhancements

* Customer Segmentation
* RFM Analysis
* Customer Lifetime Value Prediction
* Real-Time Recommendation System
* Deep Learning Models
* Cloud Deployment

---

## 👨‍💻 Author

Rakshitha HJ

Data Analytics & Machine Learning Project

---

## ⭐ If you found this project useful

Please consider giving the repository a star.

