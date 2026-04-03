import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os
import time

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Fraud Detection System", layout="wide")

# -------------------------------
# CUSTOM CSS (PREMIUM LOOK)
# -------------------------------
st.markdown("""
<style>
.main {
    background-color: #0E1117;
}
h1, h2, h3 {
    color: #FFFFFF;
}
.stButton>button {
    background-color: #00ADB5;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
}
.block-container {
    padding-top: 1rem;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# LOAD FILES
# -------------------------------
BASE_DIR = os.path.dirname(__file__)

model = joblib.load(os.path.join(BASE_DIR, "fraud_model.pkl"))
normal_df = pd.read_csv(os.path.join(BASE_DIR, "normal_small.csv"))
fraud_df = pd.read_csv(os.path.join(BASE_DIR, "fraud_small.csv"))

features = normal_df.columns.tolist()

# -------------------------------
# SESSION STATE
# -------------------------------
if "history" not in st.session_state:
    st.session_state["history"] = []

if "total" not in st.session_state:
    st.session_state["total"] = 0

if "fraud_count" not in st.session_state:
    st.session_state["fraud_count"] = 0

# -------------------------------
# SIDEBAR (CONTROL PANEL)
# -------------------------------
st.sidebar.title("⚙ Control Panel")

mode = st.sidebar.radio(
    "Select Transaction Type",
    ["Normal Transaction", "Fraud Scenario"]
)

if st.sidebar.button("Generate Transaction"):
    if mode == "Normal Transaction":
        st.session_state["transaction"] = normal_df.sample(1)
    else:
        st.session_state["transaction"] = fraud_df.sample(1)

# -------------------------------
# HEADER
# -------------------------------
st.markdown("# 💳 AI Fraud Detection Dashboard")
st.markdown("### Real-Time Transaction Risk Monitoring System")

st.divider()

# -------------------------------
# METRICS
# -------------------------------
col1, col2, col3 = st.columns(3)

col1.metric("Total Transactions", st.session_state["total"])
col2.metric("Frauds Detected", st.session_state["fraud_count"])

rate = 0
if st.session_state["total"] > 0:
    rate = (st.session_state["fraud_count"] / st.session_state["total"]) * 100

col3.metric("Detection Rate", f"{rate:.2f}%")

st.divider()

# -------------------------------
# INPUT SECTION
# -------------------------------
st.markdown("## 🧾 Transaction Details")

inputs = {}

for f in features:
    val = 0.0
    if "transaction" in st.session_state:
        val = float(st.session_state["transaction"][f].values[0])
    inputs[f] = st.number_input(f, value=val)

# -------------------------------
# DETECTION BUTTON
# -------------------------------
if st.button("🚀 Analyze Transaction"):

    with st.spinner("Analyzing transaction..."):
        time.sleep(1)

    input_df = pd.DataFrame([inputs])

    prob = model.predict_proba(input_df)[0][1]
    risk_score = round(prob * 100, 2)

    st.session_state["total"] += 1

    status = "LOW"
    if risk_score > 70:
        status = "HIGH"
        st.session_state["fraud_count"] += 1
    elif risk_score > 30:
        status = "MEDIUM"

    # -------------------------------
    # RESULT DISPLAY
    # -------------------------------
    st.markdown("## 🎯 Fraud Detection Result")

    st.markdown(f"""
    <div style="background-color:#1E1E1E;padding:20px;border-radius:10px">
    <h3>Fraud Probability: {prob:.4f}</h3>
    <h3>Risk Score: {risk_score}</h3>
    </div>
    """, unsafe_allow_html=True)

    st.progress(risk_score / 100)

    if status == "HIGH":
        st.error("🚨 HIGH RISK — Transaction Blocked")
    elif status == "MEDIUM":
        st.warning("⚠ MEDIUM RISK — OTP Verification Required")
    else:
        st.success("✅ LOW RISK — Transaction Approved")

    st.info("Model detected deviation from normal transaction behavior")

    # Save history
    st.session_state["history"].append({
        "Risk Score": risk_score,
        "Status": status
    })

st.divider()

# -------------------------------
# ALERTS
# -------------------------------
st.markdown("## 🚨 Recent Fraud Alerts")

alerts = [h for h in st.session_state["history"] if h["Status"] == "HIGH"]

if alerts:
    st.dataframe(pd.DataFrame(alerts[-5:]), use_container_width=True)
else:
    st.write("No high-risk transactions detected")

# -------------------------------
# HISTORY
# -------------------------------
st.markdown("## 📜 Transaction History")

if st.session_state["history"]:
    st.dataframe(pd.DataFrame(st.session_state["history"]), use_container_width=True)
else:
    st.write("No transactions processed yet")

# -------------------------------
# CHART
# -------------------------------
if st.session_state["total"] > 0:

    st.markdown("## 📊 Risk Distribution")

    low = sum(1 for h in st.session_state["history"] if h["Status"] == "LOW")
    med = sum(1 for h in st.session_state["history"] if h["Status"] == "MEDIUM")
    high = sum(1 for h in st.session_state["history"] if h["Status"] == "HIGH")

    labels = ["Low", "Medium", "High"]
    values = [low, med, high]

    fig, ax = plt.subplots()
    ax.bar(labels, values)
    ax.set_title("Transaction Risk Levels")

    st.pyplot(fig)
