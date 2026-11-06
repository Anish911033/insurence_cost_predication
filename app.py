import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Medical Insurance Cost Predictor", page_icon="🏥")


@st.cache_resource
def load_artifacts():
    model = joblib.load("insurence_model.pkl")
    feature_cols = joblib.load("model_features.pkl")
    return model, feature_cols


model, feature_cols = load_artifacts()

st.title("🏥 Medical Insurance Cost Predictor")
st.caption("Estimates yearly insurance charges from basic personal details.")

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    sex = st.selectbox("Sex", ["female", "male"])
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
with col2:
    children = st.number_input("Children", min_value=0, max_value=10, value=0)
    smoker = st.selectbox("Smoker", ["no", "yes"])
    region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

if st.button("Predict", type="primary"):
    row = pd.DataFrame([{"age": age, "sex": sex, "bmi": bmi,
                         "children": children, "smoker": smoker, "region": region}])
    # Same encoding as training (drop_first=True), then align to training columns
    row = pd.get_dummies(row, columns=["sex", "smoker", "region"])
    row = row.reindex(columns=feature_cols, fill_value=0)

    pred = float(np.expm1(model.predict(row)[0]))  # model predicts log1p(charges)
    st.success(f"Estimated annual charges: **${pred:,.2f}**")
    st.caption("Educational estimate only, not a real insurance quote.")
