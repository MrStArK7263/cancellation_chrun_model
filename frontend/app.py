"""
Streamlit UI for the churn model.
Run locally:  streamlit run frontend/app.py
Set API_URL (env var or Streamlit secret) to your Render URL, e.g. https://churn-api.onrender.com
"""
import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")
try:
    API_URL = st.secrets.get("API_URL", API_URL).rstrip("/")
except Exception:
    pass  # no secrets file locally - use the env var / default

st.set_page_config(page_title="Churn Predictor", page_icon="📉", layout="centered")
st.title("📉 Customer Churn Predictor")
st.caption("Enter the customer details and click Predict. The request goes to the FastAPI backend.")

with st.form("customer_form"):
    col1, col2 = st.columns(2)
    with col1:
        plan = st.selectbox("Plan type", ["Purple Plan", "Green Plan", "Blue Plan", "Red Plan"])
        promotion = st.selectbox("Has promotion?", ["no", "yes"])
        renewed = st.selectbox("Has renewed?", ["no", "yes"])
        device = st.selectbox("Device type", ["-", "iPhone", "Galaxy"],
                              help="'-' means the device is not known")
        base_fee = st.number_input("Base fee", 0.0, 100.0, 30.0, step=0.5)
        penalties = st.number_input("Number of penalties", 0, 50, 1)
    with col2:
        current = st.number_input("Current period usage", 1.0, 1000.0, 200.0, step=1.0)
        prior = st.number_input("Prior period usage", 0.0, 1000.0, 200.0, step=1.0)
        rating = st.slider("Service rating (0-20)", 0.0, 20.0, 10.0, step=0.1)
        tickets = st.number_input("Service tickets", 0, 100, 4)
        tenure = st.number_input("Customer tenure", 0, 500, 100)
    submitted = st.form_submit_button("Predict", type="primary")

if submitted:
    payload = {
        "PlanType": plan, "Promotion": promotion, "HasRenewed": renewed, "DeviceType": device,
        "BaseFee": base_fee, "NumberOfPenalties": int(penalties),
        "CurrentPeriodUsage": current, "PriorPeriodUsage": prior,
        "ServiceRating": rating, "ServiceTickets": int(tickets), "CustomerTenure": int(tenure),
    }
    try:
        resp = requests.post(f"{API_URL}/predict", json=payload, timeout=60)
        if resp.status_code == 200:
            result = resp.json()
            prob = result["churn_probability"]
            if result["prediction"] == "Churn":
                st.error(f"⚠️ Predicted: **{result['prediction']}**  |  Risk: **{result['risk_level']}**")
            else:
                st.success(f"✅ Predicted: **{result['prediction']}**  |  Risk: **{result['risk_level']}**")
            st.metric("Churn probability", f"{prob * 100:.1f}%")
            st.progress(min(max(prob, 0.0), 1.0))
            st.caption(f"Model version {result['model_version']}")
        elif resp.status_code == 422:
            st.warning("The API rejected the input: " + str(resp.json().get("detail")))
        else:
            st.error(f"API error {resp.status_code}: {resp.text}")
    except requests.exceptions.RequestException as exc:
        st.error(f"Could not reach the API at {API_URL}. Is it running? ({exc})")

with st.expander("API status"):
    try:
        st.json(requests.get(f"{API_URL}/health", timeout=60).json())
    except requests.exceptions.RequestException:
        st.write("API not reachable")
