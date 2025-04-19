# 📊 Clickstream-NPS Prediction Dashboard

**Predict Net Promoter Score (NPS) using user clickstream behavior and feedback sentiment via an interactive Streamlit dashboard.**

This project combines behavioral analytics and sentiment processing to estimate customer satisfaction levels for users who may not provide direct feedback, enabling early detection of churn risks and experience optimization.

---

## ⚙️ Features

- 💬 **Text Feedback Sentiment Scoring**  
  Polarity analysis using TextBlob for open-ended NPS comments

- 🧐 **NPS Category Prediction**  
  Random Forest classifier trained on engineered behavior + text features

- 🌐 **Streamlit Dashboard**  
  Upload CSVs, view predictions, visualize feature importance, and test user-level outcomes

---




## 📅 Project Structure

```
.
├── app.py                  # Streamlit app file
├── user_features.csv      # Sample dataset (uploadable)
├── clickstream_nps_analysis.ipynb      # Exploratory + training notebook
├── requirements.txt        # Dependencies
└── README.md
```

---

## 📚 Use Cases

- Predict NPS for users without direct feedback
- Trigger churn prevention actions early
- Feed insights into product UX decisions

---


