# 🏠 Real Estate Investment Advisor (India)

An end-to-end Machine Learning solution designed to provide data-driven insights into the Indian housing market. This platform helps investors and home-buyers predict property prices and evaluate investment viability using a dataset of over **500,000 records**.

---

## 🚀 Live Deployment

https://real-estate-investment-advisor-aayushi.streamlit.app/

---

## 📊 Business Objective
In the volatile real estate market, identifying the right time and price for investment is challenging. This project aims to:
* **Predict Property Value:** Estimating fair market prices using high-dimensional regression.
* **Evaluate Investment Risk:** Classifying properties as "Good" or "Bad" investments based on cost-to-infrastructure ratios.
* **Market Intelligence:** Providing interactive visualizations of regional price trends and amenity distributions.

---

## 🛠️ Tech Stack
* **Frontend/UI:** Streamlit (Custom Dark Theme)
* **Data Processing:** Python, Pandas, NumPy, Scikit-Learn
* **Visualization:** Plotly, Matplotlib, Seaborn
* **Machine Learning:**
    * **Random Forest Regressor:** For continuous price estimation.
    * **XGBoost Classifier:** For binary investment status categorization.
* **MLOps:** MLflow (Experiment tracking and model versioning)
---

## 📂 Project Structure
```text
├── .streamlit/             # Theme & Server configuration
├── models/                 # Serialized .pkl files (RF, XGBoost, Encoders)
├── app.py                  # Main Streamlit interface
├── data_preprocessing.py   # Feature engineering & Data cleaning logic
├── eda_charts.py           # Plotly-based interactive visualization logic
├── train_models.py         # Model training & MLflow logging script
├── requirements.txt        # Project dependencies
└── india_housing_prices.csv and processed_india_hosing_prices.csv  # Dataset (500,000+ records)
