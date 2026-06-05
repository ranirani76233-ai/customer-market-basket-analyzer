# 🛒 BasketIQ - Retail Analytics & Market Basket Intelligence

BasketIQ is a comprehensive Retail Analytics platform built using Streamlit, Machine Learning, and Market Basket Analysis techniques. The application helps retailers uncover customer purchasing patterns, segment customers, predict sales, and generate product recommendations.

---

## 🚀 Features

### 📊 Interactive Dashboard

* Sales KPIs
* Revenue Monitoring
* Customer Insights
* Product Performance Analysis
* Interactive Charts and Visualizations

### 📈 Retail Analytics

* Sales Trends Analysis
* Product-wise Revenue Analysis
* Customer Purchase Behavior
* Category Performance Evaluation

### 🔗 Market Basket Analysis

* Apriori Algorithm
* Frequent Itemset Mining
* Association Rule Generation
* Support, Confidence & Lift Metrics
* Product Affinity Analysis

### 🎯 Customer Segmentation

* K-Means Clustering
* Customer Group Identification
* RFM-based Analysis
* Segment Visualization

### 🤖 Machine Learning

* Sales Prediction Models
* Model Training and Evaluation
* Feature Importance Analysis
* Automated Model Selection

### 💡 Recommendation Engine

* Product Recommendation System
* Cross-Selling Opportunities
* Frequently Bought Together Products
* Personalized Recommendations

---

# 📁 Project Structure

```text
BasketIQ-Retail-Analytics/
│
├── data/
│   └── Assignment-1_Data.csv
│
├── assets/
│   ├── logo.png
│   └── custom.css
│
├── models/
│   ├── best_model.pkl
│   ├── customer_segmentation.pkl
│   └── association_rules.pkl
│
├── pages/
│   ├── 1_📊_Dashboard.py
│   ├── 2_📈_Analytics.py
│   ├── 3_🔗_Market_Basket.py
│   ├── 4_🎯_Customer_Segmentation.py
│   ├── 5_🤖_Machine_Learning.py
│   └── 6_💡_Recommendations.py
│
├── utils/
│   ├── data_loader.py
│   ├── analytics.py
│   ├── ml_models.py
│   └── recommender.py
│
├── reports/
│   ├── model_comparison.csv
│   └── association_rules.csv
│
├── app.py
├── train_models.py
├── requirements.txt
├── README.md
└── .streamlit/
    └── config.toml
```

---

# 📊 Dataset

The project uses retail transaction data containing:

* Order ID
* Customer ID
* Product Information
* Quantity
* Sales Amount
* Transaction Date
* Product Categories

Dataset Location:

```text
data/Assignment-1_Data.csv
```

---

# ⚙️ Installation

## Clone Repository

```bash
git clone https://github.com/yourusername/BasketIQ-Retail-Analytics.git

cd BasketIQ-Retail-Analytics
```

## Create Virtual Environment

### Windows

```bash
python -m venv venv

venv\Scripts\activate
```

### Linux / Mac

```bash
python3 -m venv venv

source venv/bin/activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Application

```bash
streamlit run app.py
```

Application will open at:

```text
http://localhost:8501
```

---

# 🧠 Machine Learning Models

The platform supports:

| Model             | Purpose                |
| ----------------- | ---------------------- |
| Linear Regression | Sales Forecasting      |
| Random Forest     | Prediction             |
| Decision Tree     | Classification         |
| K-Means           | Customer Segmentation  |
| Apriori           | Market Basket Analysis |

---

# 📈 Market Basket Metrics

### Support

Measures how frequently an itemset appears.

### Confidence

Measures reliability of an association rule.

### Lift

Measures strength of a rule relative to random occurrence.

---

# 🎯 Customer Segmentation Workflow

1. Data Cleaning
2. Feature Engineering
3. Customer Profiling
4. K-Means Clustering
5. Segment Visualization

---

# 📷 Dashboard Highlights

* Executive Sales Dashboard
* Revenue Trends
* Product Analysis
* Customer Analytics
* Cluster Visualization
* Association Rules Network

---

# 🛠️ Technologies Used

* Python
* Streamlit
* Pandas
* NumPy
* Scikit-Learn
* MLxtend
* Plotly
* Matplotlib
* Seaborn
* Joblib

---

# 📋 Future Enhancements

* Deep Learning Models
* Real-Time Retail Analytics
* Customer Lifetime Value Prediction
* Inventory Forecasting
* AI-Powered Recommendations
* Cloud Deployment

---

# 👩‍💻 Author

Rakshitha H J

Retail Analytics | Machine Learning | Data Science

---

# 📄 License

This project is licensed under the MIT License.

Feel free to use, modify, and distribute this project for educational and research purposes.

