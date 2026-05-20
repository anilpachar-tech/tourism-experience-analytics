# 🌍 Tourism Experience Analytics

A machine learning project that analyzes tourism data to provide **personalized attraction recommendations**, **predict visit modes**, and **estimate attraction ratings** using collaborative filtering and supervised learning.

---

## 🎯 Project Objectives

| Task | Description | Target |
|---|---|---|
| **Regression** | Predict rating a user will give to an attraction | Rating (1–5) |
| **Classification** | Predict how user will travel | Business / Family / Couples / Friends / Solo |
| **Recommendation** | Suggest personalized attractions | Ranked attraction list |

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| ML Libraries | Scikit-learn, XGBoost |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| App | Streamlit |
| Model Saving | Joblib |

---

## 📁 Project Structure

```
tourism-analytics/
│
├── data/
│   ├── raw/                        ← original CSV files
│   └── processed_data.csv          ← cleaned master dataframe
│
├── models/
│   ├── reg_model.pkl               ← regression model
│   ├── cls_model.pkl               ← classification model
│   ├── label_encoder.pkl           ← visit mode encoder
│   ├── user_item.pkl               ← user-item matrix for recommendations
│   └── features.pkl                ← feature column names
│
├── notebooks/
│   ├── 01_eda.ipynb                ← data cleaning + EDA + visualizations
│   └── 02_models.ipynb             ← model training + evaluation + recommendation
│
├── app.py                          ← Streamlit application
├── requirements.txt
└── README.md
```

---

## 📊 Dataset

9 tables from the Tourism domain:

| File | Description |
|---|---|
| Transaction.csv | User visits, ratings, visit mode |
| User.csv | User demographic IDs |
| Cities.csv | City names and country mapping |
| Countries.csv | Country names |
| Regions.csv | Region names |
| Continents.csv | Continent names |
| Item.csv | Attraction details |
| Types.csv | Attraction type names |
| VisitingMode.csv | Visit mode labels |

---

## 🤖 Models Used

| Task | Models Tried | Best Model (auto-selected) | Metrics |
|---|---|---|---|
| Regression | Linear Regression, Random Forest, Gradient Boosting | Best by R2 score | R2, MSE, MAE |
| Classification | Logistic Regression, Random Forest, XGBoost | Best by Accuracy | Accuracy, F1, Precision, Recall |
| Recommendation | Cosine Similarity (Collaborative Filtering) | — | Predicted Rating |

---

## 📱 App Features

- **📊 Dashboard** — 8 visualizations: visit mode distribution, continent ratings, top countries, attraction types, monthly & yearly trends
- **🧳 Classify Visit Mode** — predict travel type with probability breakdown chart
- **⭐ Predict Rating** — estimate attraction rating with star display
- **🎯 Recommendations** — personalized attraction suggestions based on similar users

---

## 🚀 Run Locally

```bash
# 1. clone repo
git clone https://github.com/anilpachar-tech/tourism-experience-analytics.git
cd tourism-analytics

# 2. install dependencies
pip install -r requirements.txt

# 3. run notebooks in order
# first run notebooks/01_eda.ipynb
# then run notebooks/02_models.ipynb

# 4. run app
streamlit run app.py
```

---

## 📦 Requirements

```
pandas
numpy
scikit-learn
xgboost
streamlit
joblib
matplotlib
seaborn
```

---

## 📈 Project Workflow

```
Raw Data → Cleaning → Merging → EDA → Feature Engineering
    → Model Training → Evaluation → Streamlit App
```

---

## 🔍 Key Insights

- **Family** and **Couples** are the most common visit modes
- **Asia** dominates in total visits
- **Beaches** and **Religious Sites** are the top attraction types
- Visit frequency peaks in **mid-year months** (June–August)

---

## 👤 Author

**Anil** — B.Tech Electrical Engineering, Central University of Karnataka
Self-learning AI/ML independently through projects and internships


---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
