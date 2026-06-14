# 🌧️ Rain Tomorrow Prediction System
### Smart Weather Station with Machine Learning-Based Forecasting
### for Agricultural Decision Support in RULINDO District

**Author:** IYAMUREMYE Pacifique (25RP20567)
**Supervisor:** Dr. MUGEMANYI Sylvere
**Department:** Electrical and Electronics Engineering
**Programme:** BTech Electronics and Telecommunication Technology
**Institution:** Tumba College of Technology — Rwanda Polytechnic
**Academic Year:** 2025–2026

---

## 🌐 Live App

👉 **[Click here to open the Rain Prediction App](https://your-app-link.streamlit.app)**

> Replace the link above with your actual Streamlit Cloud URL after deployment.

---

## 📌 Project Overview

Agriculture in Rwanda remains highly sensitive to weather variability,
particularly in hilly districts such as Rulindo, where microclimatic
conditions significantly affect crop productivity. Most smallholder
farmers rely on generalized regional weather forecasts that fail to
capture local weather dynamics, leading to poor agricultural decisions,
resource wastage, and reduced crop yields.

This project presents a **Smart Weather Station** integrated with
**Machine Learning-Based Forecasting** to provide localized, real-time
weather monitoring and short-term weather predictions tailored to
Rulindo District. The system translates weather forecasts into
actionable agricultural recommendations covering irrigation scheduling,
fertilizer application timing, pesticide spraying safety, and fungal
disease alerts.

---

## 🧠 Machine Learning Models

Four classification models were trained and evaluated to predict
whether it will rain tomorrow (Yes/No):

| Model | Accuracy | ROC-AUC |
|---|---|---|
| Logistic Regression | ~85% | ~0.87 |
| Decision Tree | ~86% | ~0.88 |
| Random Forest | ~87% | ~0.91 |
| Gradient Boosting | ~87% | ~0.92 |

The best performing model is selected automatically and used
as the default prediction model in the app.

**Evaluation Metrics Used:**
- Accuracy
- ROC-AUC Score
- Precision, Recall, F1-Score
- Confusion Matrix

---

## 📊 Dataset

| Property | Details |
|---|---|
| Name | Rain in Australia |
| Source | Australian Bureau of Meteorology |
| Rows | 145,460 readings |
| Columns | 23 weather features |
| Target Variable | `RainTomorrow` (Yes / No) |

**Key features used:**
Temperature (Min/Max/9am/3pm), Humidity (9am/3pm),
Pressure (9am/3pm), Rainfall, Evaporation, Sunshine,
Wind Gust Speed/Direction, Wind Speed (9am/3pm),
Cloud Cover
