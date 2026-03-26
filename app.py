import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
from predictor import predict

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Fraud Detection System",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main { background-color: #0a0d14; }
.block-container { padding-top: 1rem; padding-bottom: 2rem; }

/* Hero Banner */
.hero {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    border: 1px solid #1e3a5f;
}
.hero h1 { color: #ffffff; font-size: 2.2rem; font-weight: 700; margin: 0; }
.hero p  { color: #a0b4c8; font-size: 1rem; margin: 0.3rem 0 0 0; }

/* KPI Cards */
.kpi-row { display: flex; gap: 1rem; margin-bottom: 1.5rem; }
.kpi-card {
    flex: 1;
    background: linear-gradient(135deg, #1a1f2e, #1e2540);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    border-left: 4px solid #4f8bf9;
    box-shadow: 0 4px 15px rgba(0,0,0,0.3);
}
.kpi-card.fraud  { border-left-color: #ff4b4b; }
.kpi-card.legit  { border-left-color: #00c853; }
.kpi-card.rate   { border-left-color: #ffa500; }
.kpi-label { color: #8899aa; font-size: 0.8rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; }
.kpi-value { color: #ffffff; font-size: 2rem; font-weight: 700; margin-top: 0.2rem; }

/* Section Title */
.sec-title {
    font-size: 1.1rem; font-weight: 700;
    color: #4f8bf9; margin: 1rem 0 0.5rem 0;
    border-left: 3px solid #4f8bf9;
    padding-left: 0.6rem;
}

/* Result Cards */
.result-high   { background:#2d0f0f; border:1px solid #ff4b4b; border-radius:10px; padding:1rem; }
.result-medium { background:#2d1f00; border:1px solid #ffa500; border-radius:10px; padding:1rem; }
.result-low    { background:#0d2d1a; border:1px solid #00c853; border-radius:10px; padding:1rem; }

/* Alert box */
.alert-item {
    background: #1e0f0f;
    border-left: 3px solid #ff4b4b;
    border-radius: 8px;
    padding: 0.6rem 1rem;
    margin-bottom: 0.5rem;
    color: #ffaaaa;
    font-size: 0.85rem;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1923, #131d2b);
    border-right: 1px solid #1e3a5f;
}

/* Buttons */
.stButton > button {
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover { transform: translateY(-1px); box-shadow: 0 4px 12px rgba(79,139,249,0.3); }

/* Table */
.stDataFrame { border-radius: 10px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Load Data ─────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(__file__)
normal_df = pd.read_csv(os.path.join(BASE_DIR, "normal_small.csv"))
fraud_df  = pd.read_csv(os.path.join(BASE_DIR, "fraud_small.csv"))
features  = normal_df.columns.tolist()

# ── Session State ─────────────────────────────────────────────────────────────
defaults = {"history": [], "total": 0, "fraud_count": 0, "transaction": None, "last_result": None}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<br>", unsafe_allow_html=True)
    st.image("https://img.icons8.com/fluency/96/bank-card-front-side.png", width=70)
    st.markdown("## 💳 Fraud Shield")
    st.markdown("<hr style='border-color:#1e3a5f'>", unsafe_allow_html=True)

    st.markdown("### 🎯 Quick Test")
    if st.button("🟢 Normal Transaction", use_container_width=True):
        st.session_state["transaction"] = normal_df.sample(1)
    if st.button("🔴 Fraud Transaction", use_container_width=True):
        st.session_state["transaction"] = fraud_df.sample(1)

    st.markdown("<hr style='border-color:#1e3a5f'>", unsafe_allow_html=True)
    st.markdown("### 📊 Session Stats")

    total  = st.session_state["total"]
    frauds = st.session_state["fraud_count"]
    legit  = total - frauds
    rate   = round((frauds / total) * 100, 1) if total > 0 else 0.0

    st.markdown(f"- 🔢 **Total:** {total}")
    st.markdown(f"- ✅ **Legitimate:** {legit}")
    st.markdown(f"- 🚨 **Frauds:** {frauds}")
    st.markdown(f"- 📈 **Fraud Rate:** {rate}%")

    st.markdown("<hr style='border-color:#1e3a5f'>", unsafe_allow_html=True)

    if st.button("🗑️ Reset Dashboard", use_container_width=True):
        for k, v in defaults.items():
            st.session_state[k] = v
        st.rerun()

    st.markdown("<hr style='border-color:#1e3a5f'>", unsafe_allow_html=True)
    st.markdown("### ℹ️ Model Info")
    st.info("**Algorithm:** Random Forest\n\n**Features:** 30 (Time, V1–V28, Amount)\n\n**Thresholds:**\n- 🔴 HIGH > 70%\n- 🟡 MEDIUM 30–70%\n- 🟢 LOW < 30%")

# ── HERO BANNER ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>💳 AI Fraud Detection System</h1>
    <p>Real-Time Digital Payment Fraud Monitoring — Powered by Random Forest Machine Learning</p>
</div>
""", unsafe_allow_html=True)

# ── KPI CARDS ─────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(f"""<div class="kpi-card">
        <div class="kpi-label">📊 Total Transactions</div>
        <div class="kpi-value">{total}</div>
    </div>""", unsafe_allow_html=True)
with k2:
    st.markdown(f"""<div class="kpi-card legit">
        <div class="kpi-label">✅ Legitimate</div>
        <div class="kpi-value">{legit}</div>
    </div>""", unsafe_allow_html=True)
with k3:
    st.markdown(f"""<div class="kpi-card fraud">
        <div class="kpi-label">🚨 Frauds Detected</div>
        <div class="kpi-value">{frauds}</div>
    </div>""", unsafe_allow_html=True)
with k4:
    st.markdown(f"""<div class="kpi-card rate">
        <div class="kpi-label">📈 Fraud Rate</div>
        <div class="kpi-value">{rate}%</div>
    </div>""", unsafe_allow_html=True)

# ── TRANSACTION INPUT ─────────────────────────────────────────────────────────
st.markdown('<div class="sec-title">🔍 Transaction Input</div>', unsafe_allow_html=True)
st.caption("Fill in transaction details manually OR use Quick Test buttons in the sidebar to auto-fill.")

inputs = {}
chunks = [features[i:i+5] for i in range(0, len(features), 5)]
for chunk in chunks:
    cols = st.columns(len(chunk))
    for col, f in zip(cols, chunk):
        val = 0.0
        if st.session_state["transaction"] is not None:
            val = float(st.session_state["transaction"][f].values[0])
        inputs[f] = col.number_input(f, value=val, key=f"inp_{f}", label_visibility="visible", format="%.4f")

st.markdown("<br>", unsafe_allow_html=True)

# ── CHECK BUTTON ──────────────────────────────────────────────────────────────
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    check = st.button("🔎 ANALYZE TRANSACTION", use_container_width=True, type="primary")

# ── RESULT ────────────────────────────────────────────────────────────────────
if check:
    prob, risk_score, status = predict(inputs)
    st.session_state["total"] += 1
    if status == "HIGH":
        st.session_state["fraud_count"] += 1
    st.session_state["history"].append({
        "Txn #":       st.session_state["total"],
        "Amount ($)":  round(inputs.get("Amount", 0), 2),
        "Risk Score":  risk_score,
        "Probability": f"{round(prob*100,2)}%",
        "Status":      status
    })
    st.session_state["last_result"] = (prob, risk_score, status, inputs)

if st.session_state["last_result"]:
    prob, risk_score, status, inp = st.session_state["last_result"]

    st.markdown('<div class="sec-title">🧾 Detection Result</div>', unsafe_allow_html=True)

    css_class = {"HIGH": "result-high", "MEDIUM": "result-medium", "LOW": "result-low"}[status]
    icon      = {"HIGH": "🚨", "MEDIUM": "⚠️", "LOW": "✅"}[status]
    msg       = {
        "HIGH":   "HIGH RISK — Transaction BLOCKED! Possible fraud detected.",
        "MEDIUM": "MEDIUM RISK — OTP Verification Required before processing.",
        "LOW":    "LOW RISK — Transaction Approved successfully."
    }[status]

    st.markdown(f'<div class="{css_class}"><b>{icon} {msg}</b></div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    r1, r2, r3 = st.columns(3)
    r1.metric("🎯 Fraud Probability", f"{round(prob*100,2)}%")
    r2.metric("📊 Risk Score",        f"{risk_score}/100")
    r3.metric("🏷️ Status",            status)

    st.progress(int(risk_score))

    with st.expander("📋 View All Feature Values"):
        st.dataframe(
            pd.DataFrame(inp.items(), columns=["Feature", "Value"]).set_index("Feature"),
            use_container_width=True
        )

st.markdown("<hr style='border-color:#1e3a5f; margin:1.5rem 0'>", unsafe_allow_html=True)

# ── HISTORY + ALERTS ──────────────────────────────────────────────────────────
left, right = st.columns([3, 1])

with left:
    st.markdown('<div class="sec-title">📜 Transaction History</div>', unsafe_allow_html=True)
    if st.session_state["history"]:
        hist_df = pd.DataFrame(st.session_state["history"])

        def color_row(val):
            return {
                "HIGH":   "background-color:#3d1515; color:#ff8080",
                "MEDIUM": "background-color:#3d2e00; color:#ffcc66",
                "LOW":    "background-color:#0d2d1a; color:#66ff99"
            }.get(val, "")

        st.dataframe(
            hist_df.style.applymap(color_row, subset=["Status"]),
            use_container_width=True, height=300
        )

        # Download button
        csv = hist_df.to_csv(index=False)
        st.download_button("⬇️ Download History CSV", csv, "fraud_history.csv", "text/csv")
    else:
        st.info("No transactions yet. Use the sidebar to generate one!")

with right:
    st.markdown('<div class="sec-title">🚨 Live Alerts</div>', unsafe_allow_html=True)
    alerts = [h for h in st.session_state["history"] if h["Status"] == "HIGH"]
    if alerts:
        for a in reversed(alerts[-6:]):
            st.markdown(f"""<div class="alert-item">
                🚨 <b>Txn #{a['Txn #']}</b><br>
                💰 ${a['Amount ($)']} | Score: {a['Risk Score']}
            </div>""", unsafe_allow_html=True)
    else:
        st.success("✅ No fraud alerts")

st.markdown("<hr style='border-color:#1e3a5f; margin:1.5rem 0'>", unsafe_allow_html=True)

# ── ANALYTICS CHARTS ──────────────────────────────────────────────────────────
if st.session_state["history"]:
    st.markdown('<div class="sec-title">📊 Analytics Dashboard</div>', unsafe_allow_html=True)
    hist_df = pd.DataFrame(st.session_state["history"])

    BG = "#1a1f2e"
    COLORS = {"LOW": "#00c853", "MEDIUM": "#ffa500", "HIGH": "#ff4b4b"}

    ch1, ch2 = st.columns(2)

    # Chart 1 — Bar: Risk Distribution
    with ch1:
        st.markdown("**Risk Level Distribution**")
        counts = hist_df["Status"].value_counts().reindex(["LOW", "MEDIUM", "HIGH"], fill_value=0)
        fig, ax = plt.subplots(figsize=(5, 3.2))
        bars = ax.bar(counts.index, counts.values,
                      color=[COLORS[k] for k in counts.index], width=0.5, edgecolor="none")
        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    str(val), ha="center", color="white", fontweight="bold", fontsize=11)
        ax.set_facecolor(BG); fig.patch.set_facecolor(BG)
        ax.tick_params(colors="white"); ax.spines[:].set_visible(False)
        ax.set_title("Transaction Risk Levels", color="white", fontsize=11, pad=10)
        ax.yaxis.label.set_color("white"); ax.set_yticks([])
        st.pyplot(fig)
        plt.close()

    # Chart 2 — Line: Risk Score Trend
    with ch2:
        st.markdown("**Risk Score Trend**")
        fig2, ax2 = plt.subplots(figsize=(5, 3.2))
        ax2.plot(hist_df["Txn #"], hist_df["Risk Score"],
                 marker="o", color="#4f8bf9", linewidth=2, markersize=5)
        ax2.fill_between(hist_df["Txn #"], hist_df["Risk Score"], alpha=0.15, color="#4f8bf9")
        ax2.axhline(70, color="#ff4b4b", linestyle="--", linewidth=1, label="High (70)")
        ax2.axhline(30, color="#ffa500", linestyle="--", linewidth=1, label="Medium (30)")
        ax2.set_facecolor(BG); fig2.patch.set_facecolor(BG)
        ax2.tick_params(colors="white"); ax2.spines[:].set_color("#2a3050")
        ax2.set_title("Risk Score Over Time", color="white", fontsize=11, pad=10)
        ax2.legend(facecolor=BG, labelcolor="white", fontsize=8, framealpha=0.5)
        st.pyplot(fig2)
        plt.close()

    ch3, ch4 = st.columns(2)

    # Chart 3 — Pie: Fraud vs Legit
    with ch3:
        st.markdown("**Fraud vs Legitimate**")
        fig3, ax3 = plt.subplots(figsize=(4, 3.2))
        total_now = st.session_state["total"]
        frauds_now = st.session_state["fraud_count"]
        legit_now  = total_now - frauds_now
        pie_vals = [legit_now, frauds_now] if total_now > 0 else [1, 0]
        wedges, texts, autotexts = ax3.pie(
            pie_vals, labels=["Legitimate", "Fraud"],
            colors=["#00c853", "#ff4b4b"],
            autopct="%1.1f%%", startangle=90,
            wedgeprops={"edgecolor": BG, "linewidth": 2}
        )
        for t in texts + autotexts:
            t.set_color("white")
        fig3.patch.set_facecolor(BG)
        ax3.set_title("Transaction Breakdown", color="white", fontsize=11, pad=10)
        st.pyplot(fig3)
        plt.close()

    # Chart 4 — Scatter: Amount vs Risk
    with ch4:
        st.markdown("**Amount vs Risk Score**")
        fig4, ax4 = plt.subplots(figsize=(5, 3.2))
        for s, c in COLORS.items():
            sub = hist_df[hist_df["Status"] == s]
            if not sub.empty:
                ax4.scatter(sub["Amount ($)"], sub["Risk Score"],
                            label=s, color=c, s=70, alpha=0.85, edgecolors="none")
        ax4.axhline(70, color="#ff4b4b", linestyle="--", linewidth=0.8, alpha=0.5)
        ax4.axhline(30, color="#ffa500", linestyle="--", linewidth=0.8, alpha=0.5)
        ax4.set_facecolor(BG); fig4.patch.set_facecolor(BG)
        ax4.tick_params(colors="white"); ax4.spines[:].set_color("#2a3050")
        ax4.set_xlabel("Amount ($)", color="white", fontsize=9)
        ax4.set_ylabel("Risk Score", color="white", fontsize=9)
        ax4.set_title("Amount vs Risk Score", color="white", fontsize=11, pad=10)
        ax4.legend(facecolor=BG, labelcolor="white", fontsize=8, framealpha=0.5)
        st.pyplot(fig4)
        plt.close()

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align:center; padding:1rem; background:#0f1923;
            border-radius:10px; border:1px solid #1e3a5f;'>
    <span style='color:#4f8bf9; font-weight:700; font-size:1rem;'>💳 AI Fraud Detection System</span><br>
    <span style='color:#556677; font-size:0.8rem;'>
        Powered by Random Forest · Built with Streamlit · Real-Time Payment Fraud Monitoring
    </span>
</div>
""", unsafe_allow_html=True)
