# ✈️ Airline Passenger Satisfaction Predictor

[![Streamlit App](https://img.shields.io/badge/Streamlit-Live_App-red?logo=streamlit)](https://airline-passenger-satisfaction-prediction-project.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.0.5-FF6600)](https://xgboost.readthedocs.io/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.7.2-orange?logo=scikit-learn)](https://scikit-learn.org/)
[![Accuracy](https://img.shields.io/badge/Test_Accuracy-96.40%25-brightgreen)](#-model-overview)

---

## 🚀 Live Demo

👉 **Try it here:** [Airline Passenger Satisfaction App](https://airline-passenger-satisfaction-prediction-project.streamlit.app/)

---

## 📌 Overview

This project predicts whether an **airline passenger will be satisfied or dissatisfied** using machine learning.
It compares **12 classifiers** and ships the best one — a tuned **XGBoost** model — inside an
**interactive Streamlit web app** for real-time predictions.

Users can enter trip details and service ratings to get an instant prediction with **96.40% test accuracy**,
along with a **TreeSHAP explanation** showing exactly which factors drove that specific result.

---

## ⚙️ Features

✅ Real-time satisfaction predictions with 96.40% accuracy
✅ Per-passenger TreeSHAP explanation of every prediction
✅ Batch scoring — upload a CSV and download the results
✅ Interactive EDA answering 7 analytical questions
✅ 12 models trained, tuned and compared side by side
✅ Live confusion matrix computed on 25,976 unseen passengers

---

## 🛠️ Tech Stack

| Layer | Tools |
|-------|--------|
| **Frontend** | [Streamlit](https://streamlit.io/), [Plotly](https://plotly.com/) |
| **Backend / ML** | [Python](https://www.python.org/), [XGBoost](https://xgboost.readthedocs.io/), [LightGBM](https://lightgbm.readthedocs.io/), [CatBoost](https://catboost.ai/), [Scikit-learn](https://scikit-learn.org/) |
| **Explainability** | [SHAP](https://shap.readthedocs.io/) (TreeSHAP) |
| **Data Processing** | [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/) |
| **Model Persistence** | [Joblib](https://joblib.readthedocs.io/) |

---

## 📂 Project Structure

```
Airline-Passenger-Satisfaction/
│
├── Final_notebook.ipynb       # Cleaning, EDA, preprocessing, modelling, evaluation
├── app.py                     # Streamlit application
├── best_model.pkl             # Trained pipeline bundle (2.0 MB)
├── model_comparison.csv       # Scores for all 12 trained models
├── data/
│   ├── train.csv              # 103,904 passengers
│   └── test.csv               # 25,976 passengers
├── requirements.txt           # Pinned dependencies
├── runtime.txt                # Python version for deployment
├── .streamlit/config.toml     # Dark theme and upload limit
└── README.md                  # Documentation
```

---

## 🧠 Model Overview

- **Algorithm:** XGBoost Classifier, tuned with `GridSearchCV`
- **Target Variable:** Satisfied (1) / Neutral or Dissatisfied (0)
- **Training Data:** 103,904 passengers
- **Test Data:** 25,976 passengers
- **Features:** 17 inputs → 18 after encoding
- **Selection Metric:** F1 score, across 12 candidate models

### 📊 Model Comparison

| Model | Test Accuracy | F1 Score |
|-------|--------------|----------|
| **Tuned XGBoost** ⭐ | **0.9640** | **0.9639** |
| Tuned LightGBM | 0.9639 | 0.9638 |
| Tuned CatBoost | 0.9637 | 0.9636 |
| Tuned Random Forest | 0.9622 | 0.9621 |
| SVC | 0.9562 | 0.9562 |
| Tuned Decision Tree | 0.9517 | 0.9516 |
| Gradient Boosting | 0.9406 | 0.9405 |
| KNN | 0.9365 | 0.9362 |
| AdaBoost | 0.9200 | 0.9199 |
| Logistic Regression | 0.8694 | 0.8691 |
| SVC With Linear Kernel | 0.8690 | 0.8686 |
| Naive Bayes | 0.8587 | 0.8582 |

### 🔧 Training Workflow

1. Data cleaning — drop `id`, fill missing arrival delays with that flight's departure delay
2. Exploratory analysis — seven questions about who is satisfied and why
3. Feature selection — drop 5 features with near-zero correlation to the target
4. Preprocessing — `StandardScaler` on numerics, `OneHotEncoder` on class
5. Train and tune 12 classifiers, compared with one shared evaluation function
6. Pick the best by F1 score, then explain it with TreeSHAP
7. Save the model, transformer and encoder together as `best_model.pkl`

---

## 🖥️ How to Run Locally

1. **Clone this repository**
   ```bash
   git clone https://github.com/zeyadmedhat/Airline-Passenger-Satisfaction.git
   cd Airline-Passenger-Satisfaction
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app**
   ```bash
   streamlit run app.py
   ```

4. Visit the local URL shown in the terminal (usually [http://localhost:8501](http://localhost:8501)).

---

## 📦 Requirements

From `requirements.txt`:

```
streamlit==1.50.0
pandas==2.3.2
numpy==2.3.3
scikit-learn==1.7.2
joblib==1.5.2
plotly==6.3.0
shap==0.51.0
xgboost==3.0.5
```

> `best_model.pkl` was written with scikit-learn 1.7.2. That version is pinned deliberately —
> a different one will fail to unpickle the model.

---

## 💻 Application Pages

### 🏠 Home
- Headline metrics: best model, test accuracy, F1 score, overfitting gap
- Key insights drawn from the analysis
- Full comparison of all 12 models with an F1 chart
- Confusion matrix computed live on the unseen test set
- Feature importance and a four-step "How It Works" summary

### 📊 EDA (Exploratory Data Analysis)
Seven questions, each answered with interactive charts and a written takeaway:
1. Is the target balanced?
2. Which passenger groups are most satisfied?
3. Does age affect satisfaction?
4. Which services separate satisfied from dissatisfied passengers?
5. Why does a 0 rating look so strange?
6. Do flight delays make passengers unhappy?
7. Do age and flight distance differ between the two groups?

### 🔮 Prediction
- **One passenger:** enter 17 details, get a prediction with both probabilities
  and a TreeSHAP chart showing what drove it
- **Upload a CSV:** score a whole file at once, see the predicted split and
  confidence distribution, then download every row as CSV

---

## 🔍 Features Analyzed (17 Total)

**Passenger (2):** Age, Customer Type
**Trip (3):** Type of Travel, Class, Flight Distance
**Service Ratings (12):** Inflight wifi service, Ease of Online booking, Food and drink,
Online boarding, Seat comfort, Inflight entertainment, On-board service, Leg room service,
Baggage handling, Checkin service, Inflight service, Cleanliness

**Dropped (5):** Gender, Gate location, Departure/Arrival time convenient,
Departure Delay in Minutes, Arrival Delay in Minutes — all showed near-zero
correlation with satisfaction.

> Ratings run from 1 to 5, where **0 means "not applicable"** rather than a low score —
> a quirk the EDA uncovered and the model accounts for.

---

## 🔑 Key Findings

- **Online boarding is the strongest driver** of satisfaction, with the widest rating
  gap between the two groups and the highest feature importance in the model.
- **Business travellers and loyal customers** are far more satisfied than personal
  travellers and first-time flyers.
- **Delays matter less than expected** — median delay is 0 minutes for both groups,
  making it a weak signal next to the service ratings.
- Improving the **digital experience** (boarding, booking, wifi) is likely to raise
  satisfaction more than reducing delays.

---

## 🚀 Deploy to Streamlit Community Cloud

1. Push this folder to a **public** GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app**, pick the repository and branch, set the main file to `app.py`.
4. Under **Advanced settings**, choose **Python 3.12**.
5. Click **Deploy**.

---

## 🛠️ Troubleshooting

**Port already in use:**
```bash
streamlit run app.py --server.port 8502
```

**Module not found:**
```bash
pip install -r requirements.txt
```

**Model fails to load:** check your scikit-learn version matches `1.7.2` — the pickle
is version-sensitive.

**File not found:** ensure `best_model.pkl`, `model_comparison.csv` and the `data/`
folder sit next to `app.py`.

---

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** — you're free to use, modify, and
distribute it for educational or personal purposes.

---

## 📊 Dataset

Airline Passenger Satisfaction — 103,904 training and 25,976 test passengers, each with
trip details and 14 service ratings.

---

## 👥 The Team

This project was built as the final project of the **NTI Internship** by:

<div align="center">

| Team Member |
|:-----------:|
| **Yassin Abdullah** |
| **Mohamed Emad** |
| **Moaaz Ahmed** |
| **Marwan Sherif** |
| **Zeyad Medhat** |

</div>

---

<div align="center">

**Built by the Project X Team · NTI Internship · © 2026**

⭐ If you found this project useful, consider giving it a star!

</div>
