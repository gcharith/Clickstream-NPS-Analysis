import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Title
st.title("Clickstream-Based NPS Prediction Dashboard")

# Upload CSV file
uploaded_file = st.file_uploader("Upload preprocessed feature CSV (with NPS category). If trying out get the preprocessed user_features dataset @ https://github.com/gcharith/Clickstream-NPS-Analysis", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Label encode target
    label_encoder = LabelEncoder()
    df['nps_encoded'] = label_encoder.fit_transform(df['category_nps'])

    # Select features and target
    features = [
        'num_of_sessions', 'num_of_events', 'avg_duration_per_event',
        'click_button', 'form_submit', 'hover', 'nav_click', 'scroll',
        'encoded_frequent_page', 'sentiment'
    ]
    target = 'nps_encoded'

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Feature Importance Plot
    importances = model.feature_importances_
    feat_df = pd.DataFrame({'Feature': features, 'Importance': importances})
    feat_df = feat_df.sort_values(by='Importance', ascending=True)

    st.subheader("Feature Importance")
    fig, ax = plt.subplots()
    sns.barplot(x='Importance', y='Feature', data=feat_df, ax=ax)
    st.pyplot(fig)

    # Predict a single user (optional dropdown)
    st.subheader("Predict NPS Category for a Single User")
    user_id = st.selectbox("Choose a user_id", df['user_id'].unique())

    if user_id:
        user_row = df[df['user_id'] == user_id][features]
        pred = model.predict(user_row)[0]
        label = label_encoder.inverse_transform([pred])[0]
        st.markdown(f"**Predicted NPS Category:** {label}")

    # Show dataset preview
    with st.expander("Preview Dataset"):
        st.dataframe(df.head())
