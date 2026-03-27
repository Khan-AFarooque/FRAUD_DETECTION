"""
╔══════════════════════════════════════════════════════════════════╗
║          FRAUDSHIELD PRO  —  Cipher Sentinels  ·  2026          ║
║   AI-ML Fraud Detection for UPI / Digital Payment Ecosystems    ║
╚══════════════════════════════════════════════════════════════════╝

Upgrade Log
───────────
• Fraud vs Normal pie chart + dataset-level stats in Dashboard
• Transaction ID generation + lookup (search by TXN ID)
• Adaptive risk threshold (trust score + user behavior)
• Velocity detection  (≥3 txns in 60 s → flag)
• Geo-location anomaly / impossible-travel detection
• Device fingerprint simulation (new vs known device)
• Account freeze after repeated HIGH-risk detections
• Risk factor breakdown radar chart
• Personalized fraud-prevention tips engine
• Admin panel with full fraud log + system health
"""

# ── Imports ───────────────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os, time, uuid, random
from datetime import datetime, timedelta

# ── Predictor (graceful fallback if file missing) ─────────────────────────────
try:
    from predictor import predict as _model_predict          # real Random-Forest
    REAL_MODEL = True
except ImportError:
    REAL_MODEL = False
    def _model_predict(inputs):                              # demo fallback
        amount = inputs.get("Amount", 0)
        v1     = inputs.get("V1", 0)
        prob   = min(max(abs(v1) / 10 + amount / 10000, 0), 1)
        prob   = round(prob + random.uniform(-0.05, 0.05), 4)
        prob   = max(0.01, min(0.99, prob))
        risk   = int(prob * 100)
        status = "HIGH" if prob > 0.70 else ("MEDIUM" if prob > 0.30 else "LOW")
        return prob, risk, status

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FraudShield Pro · Cipher Sentinels",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: #07090f;
}
.block-container { padding: 1rem 2rem 2rem; max-width: 1400px; }

/* ── Hero ── */
.hero {
    background: linear-gradient(135deg, #0a1628 0%, #0d2137 50%, #091520 100%);
    border: 1px solid #1a3a5c;
    border-radius: 20px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute; top: 0; right: 0;
    width: 300px; height: 100%;
    background: radial-gradient(ellipse at right, rgba(0,200,255,0.07) 0%, transparent 70%);
}
.hero-badge {
    display: inline-block;
    background: rgba(0,200,255,0.1);
    border: 1px solid rgba(0,200,255,0.3);
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.7rem;
    color: #00c8ff;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 0.8rem;
    font-family: 'JetBrains Mono', monospace;
}
.hero h1 {
    color: #ffffff;
    font-size: 2.3rem;
    font-weight: 800;
    margin: 0 0 0.4rem 0;
    line-height: 1.2;
}
.hero h1 span { color: #00c8ff; }
.hero p { color: #7a9ab5; font-size: 0.95rem; margin: 0; }

/* ── KPI Cards ── */
.kpi-card {
    background: linear-gradient(135deg, #0d1525, #111c2f);
    border: 1px solid #1a2d45;
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    border-top: 3px solid #4f8bf9;
    transition: transform 0.2s;
}
.kpi-card:hover { transform: translateY(-2px); }
.kpi-card.fraud  { border-top-color: #ff4b4b; }
.kpi-card.legit  { border-top-color: #00e676; }
.kpi-card.rate   { border-top-color: #ffa726; }
.kpi-card.frozen { border-top-color: #ab47bc; }
.kpi-label {
    color: #4a6a8a;
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-family: 'JetBrains Mono', monospace;
}
.kpi-value { color: #ffffff; font-size: 2rem; font-weight: 800; margin-top: 0.3rem; }
.kpi-delta { color: #4a6a8a; font-size: 0.75rem; margin-top: 0.1rem; }

/* ── Section Title ── */
.sec-title {
    font-size: 0.85rem;
    font-weight: 700;
    color: #00c8ff;
    margin: 1.5rem 0 0.8rem 0;
    padding: 0.5rem 0.8rem;
    background: rgba(0,200,255,0.05);
    border-left: 3px solid #00c8ff;
    border-radius: 0 8px 8px 0;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-family: 'JetBrains Mono', monospace;
}

/* ── Result Cards ── */
.result-card {
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    margin: 1rem 0;
    border: 1px solid;
}
.result-high   { background: rgba(255,75,75,0.08);  border-color: #ff4b4b; }
.result-medium { background: rgba(255,167,38,0.08); border-color: #ffa726; }
.result-low    { background: rgba(0,230,118,0.08);  border-color: #00e676; }
.result-card h3 { margin: 0 0 0.3rem 0; font-size: 1.1rem; }
.result-card p  { margin: 0; color: #8aa5c0; font-size: 0.9rem; }

/* ── Alert ── */
.alert-item {
    background: rgba(255,75,75,0.06);
    border-left: 3px solid #ff4b4b;
    border-radius: 0 8px 8px 0;
    padding: 0.6rem 1rem;
    margin-bottom: 0.5rem;
    color: #ffaaaa;
    font-size: 0.82rem;
    font-family: 'JetBrains Mono', monospace;
}

/* ── Freeze Banner ── */
.freeze-banner {
    background: linear-gradient(135deg, rgba(171,71,188,0.15), rgba(123,31,162,0.1));
    border: 1px solid #7b1fa2;
    border-radius: 12px;
    padding: 1.2rem 1.6rem;
    margin-bottom: 1rem;
    text-align: center;
}
.freeze-banner h3 { color: #ce93d8; margin: 0 0 0.3rem; font-size: 1.1rem; }
.freeze-banner p  { color: #9c5aab; margin: 0; font-size: 0.85rem; }

/* ── Risk Tag ── */
.risk-tag {
    display: inline-block;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.75rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 1px;
}
.risk-high   { background: rgba(255,75,75,0.2);  color: #ff4b4b; }
.risk-medium { background: rgba(255,167,38,0.2); color: #ffa726; }
.risk-low    { background: rgba(0,230,118,0.2);  color: #00e676; }

/* ── Txn Lookup Card ── */
.lookup-card {
    background: linear-gradient(135deg, #0d1525, #111c2f);
    border: 1px solid #1a2d45;
    border-radius: 14px;
    padding: 1.5rem;
}

/* ── Tip Card ── */
.tip-card {
    background: rgba(0,200,255,0.04);
    border: 1px solid rgba(0,200,255,0.15);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.7rem;
}
.tip-card strong { color: #00c8ff; }
.tip-card p { color: #7a9ab5; margin: 0.3rem 0 0; font-size: 0.85rem; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #060a14, #0a1120);
    border-right: 1px solid #12233a;
}
section[data-testid="stSidebar"] * { font-family: 'Syne', sans-serif; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #0a1120;
    border-radius: 12px;
    padding: 4px;
    gap: 4px;
    border: 1px solid #1a2d45;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: #4a6a8a;
    font-weight: 600;
    font-size: 0.85rem;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #0d2137, #0a1628) !important;
    color: #00c8ff !important;
}

/* ── Buttons ── */
.stButton > button {
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-family: 'Syne', sans-serif !important;
    letter-spacing: 0.5px !important;
    transition: all 0.2s !important;
}
.stButton > button:hover { transform: translateY(-1px) !important; }

/* ── Progress ── */
.stProgress > div > div { border-radius: 10px; }

/* ── Velocity Warning ── */
.velocity-warn {
    background: rgba(255,152,0,0.1);
    border: 1px solid #ff9800;
    border-radius: 10px;
    padding: 0.8rem 1.2rem;
    color: #ffb74d;
    font-size: 0.85rem;
    margin-bottom: 0.8rem;
}

/* ── Geo Anomaly ── */
.geo-warn {
    background: rgba(255,75,75,0.08);
    border: 1px solid #ff4b4b;
    border-radius: 10px;
    padding: 0.8rem 1.2rem;
    color: #ff8080;
    font-size: 0.85rem;
    margin-bottom: 0.8rem;
}

/* ── Device badge ── */
.device-new  { color: #ffa726; font-size: 0.8rem; font-family: 'JetBrains Mono', monospace; }
.device-know { color: #00e676; font-size: 0.8rem; font-family: 'JetBrains Mono', monospace; }

/* ── Admin table ── */
.admin-log { font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; }

/* ── Metric override ── */
[data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# ── Helper Functions ──────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

CITIES = [
    "Mumbai", "Delhi", "Bengaluru", "Pune", "Hyderabad",
    "Chennai", "Kolkata", "Ahmedabad", "Jaipur", "Surat",
    "Dubai", "Singapore", "London", "New York", "Tokyo",
]

KNOWN_DEVICES = ["Chrome · Windows 11", "Safari · iPhone 15", "Firefox · MacBook"]

def gen_txn_id() -> str:
    """Generate a unique 10-char transaction ID."""
    return "TXN" + uuid.uuid4().hex[:7].upper()

def get_decision(status: str) -> tuple:
    """Map risk status → (decision label, icon, colour)."""
    return {
        "HIGH":   ("🚫 BLOCK",          "🚫", "#ff4b4b"),
        "MEDIUM": ("🔐 OTP VERIFY",     "🔐", "#ffa726"),
        "LOW":    ("✅ APPROVED",        "✅", "#00e676"),
    }[status]

def check_velocity(history: list, window_sec: int = 60) -> bool:
    """Return True if ≥3 transactions happened within `window_sec` seconds."""
    if len(history) < 3:
        return False
    recent = [h for h in history if (time.time() - h.get("ts", 0)) <= window_sec]
    return len(recent) >= 3

def check_geo_anomaly(prev_city: str, curr_city: str) -> bool:
    """Impossible-travel: flag if consecutive cities are from different continents."""
    india  = {"Mumbai","Delhi","Bengaluru","Pune","Hyderabad","Chennai","Kolkata","Ahmedabad","Jaipur","Surat"}
    abroad = {"Dubai","Singapore","London","New York","Tokyo"}
    if not prev_city or prev_city == curr_city:
        return False
    pc_abroad = prev_city in abroad
    cc_abroad = curr_city in abroad
    return pc_abroad != cc_abroad          # one domestic, one international

def adaptive_threshold(base_prob: float, trust_score: int) -> str:
    """
    Adjust risk level by user trust score (0–100).
    High-trust users get slightly more lenient HIGH threshold.
    """
    adjust = (trust_score - 50) / 1000     # ±0.05 shift at extremes
    high_t  = max(0.55, min(0.85, 0.70 - adjust))
    med_t   = max(0.20, min(0.45, 0.30 - adjust))
    if base_prob >= high_t:
        return "HIGH"
    elif base_prob >= med_t:
        return "MEDIUM"
    return "LOW"

def risk_breakdown(prob: float, amount: float, is_new_device: bool,
                   velocity_flag: bool, geo_flag: bool) -> dict:
    """Return dict of risk factor → score (0–100) for radar chart."""
    base = prob * 60
    return {
        "ML Model Score":    round(base),
        "Amount Risk":       round(min(amount / 200, 100)),
        "Device Trust":      round(80 if is_new_device else 10),
        "Velocity Risk":     round(70 if velocity_flag else 5),
        "Geo Anomaly":       round(85 if geo_flag else 5),
        "Time-of-Day Risk":  round(40 if datetime.now().hour < 6 or datetime.now().hour > 22 else 10),
    }

def prevention_tips(status: str, amount: float, velocity: bool,
                    geo: bool, new_device: bool) -> list:
    """Return contextual fraud-prevention tips."""
    tips = []
    if status == "HIGH":
        tips.append(("🚨 Immediate Action Required",
                     "This transaction has been blocked. Contact your bank and change your UPI PIN immediately."))
        tips.append(("🔒 Secure Your Account",
                     "Enable two-factor authentication and review all recent transactions in your bank app."))
    if velocity:
        tips.append(("⚡ Velocity Alert",
                     "Multiple transactions detected in under 60 seconds — a common fraud pattern. Set per-minute limits in your UPI app."))
    if geo:
        tips.append(("🌍 Geo-Anomaly Detected",
                     "Transaction origin differs drastically from your usual location. Turn on geo-fencing in your payment app."))
    if new_device:
        tips.append(("📱 New Device Alert",
                     "Payment initiated from an unrecognised device. Always verify new device logins via OTP before allowing transactions."))
    if amount > 10000:
        tips.append(("💰 Large Transaction",
                     "High-value transactions carry elevated risk. Set daily UPI limits and receive SMS alerts for every transaction."))
    if not tips:
        tips.append(("✅ Stay Vigilant",
                     "This transaction looks safe. Always verify payee VPA before sending money and never share OTPs."))
        tips.append(("🛡️ Routine Hygiene",
                     "Change your UPI PIN monthly, keep your bank app updated, and enable biometric authentication."))
    return tips

# ══════════════════════════════════════════════════════════════════════════════
# ── Load Sample Data (for dataset-level stats) ────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_data
def load_data():
    """Load normal/fraud CSVs; return merged df with Class column."""
    n_path = os.path.join(BASE_DIR, "normal_small.csv")
    f_path = os.path.join(BASE_DIR, "fraud_small.csv")
    try:
        n_df = pd.read_csv(n_path); n_df["Class"] = 0
        f_df = pd.read_csv(f_path); f_df["Class"] = 1
        df = pd.concat([n_df, f_df], ignore_index=True)
        features = [c for c in n_df.columns if c != "Class"]
    except FileNotFoundError:
        # Demo synthetic dataset
        rng = np.random.default_rng(42)
        n   = 5000
        df  = pd.DataFrame({
            "Time":   rng.uniform(0, 172800, n),
            "Amount": rng.exponential(100, n),
            **{f"V{i}": rng.normal(0, 1, n) for i in range(1, 29)},
        })
        df["Class"] = (rng.random(n) < 0.0017).astype(int)
        df.loc[df["Class"] == 1, "Amount"] *= rng.uniform(2, 8, df["Class"].sum())
        features = [c for c in df.columns if c != "Class"]
    return df, features

df, features = load_data()

# ══════════════════════════════════════════════════════════════════════════════
# ── Session State Initialisation ─────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
defaults = {
    "history":         [],          # list of txn dicts
    "total":           0,
    "fraud_count":     0,
    "transaction":     None,        # pre-filled quick-test row
    "last_result":     None,
    "prev_city":       None,
    "known_devices":   set(KNOWN_DEVICES),
    "trust_score":     75,
    "account_frozen":  False,
    "freeze_count":    0,
    "txn_db":          {},          # txn_id → full result dict
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ══════════════════════════════════════════════════════════════════════════════
# ── SIDEBAR ───────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align:center; margin-bottom:1rem;'>
        <div style='font-size:2.5rem;'>🛡️</div>
        <div style='font-size:1.2rem; font-weight:800; color:#fff;'>FraudShield<span style='color:#00c8ff;'>Pro</span></div>
        <div style='font-size:0.7rem; color:#4a6a8a; font-family:JetBrains Mono,monospace; letter-spacing:2px;'>CIPHER SENTINELS</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#12233a; margin:0.5rem 0'>", unsafe_allow_html=True)

    # ── Quick Test ────────────────────────────────────────────────────────────
    st.markdown("#### 🎯 Quick Test")
    col_n, col_f = st.columns(2)
    with col_n:
        if st.button("✅ Normal", use_container_width=True):
            sample = df[df["Class"] == 0].sample(1)
            st.session_state["transaction"] = sample[features]
    with col_f:
        if st.button("🚨 Fraud", use_container_width=True):
            sample = df[df["Class"] == 1].sample(1)
            st.session_state["transaction"] = sample[features]

    st.markdown("<hr style='border-color:#12233a; margin:0.8rem 0'>", unsafe_allow_html=True)

    # ── Adaptive Trust Score ──────────────────────────────────────────────────
    st.markdown("#### 🎛️ Adaptive Risk Settings")
    st.session_state["trust_score"] = st.slider(
        "User Trust Score", 0, 100, st.session_state["trust_score"],
        help="Higher trust = more lenient HIGH/MEDIUM thresholds. Reflects behavioral history."
    )
    trust = st.session_state["trust_score"]
    trust_label = "🟢 High Trust" if trust >= 70 else ("🟡 Medium Trust" if trust >= 40 else "🔴 Low Trust")
    st.caption(f"Profile: **{trust_label}** ({trust}/100)")

    st.markdown("<hr style='border-color:#12233a; margin:0.8rem 0'>", unsafe_allow_html=True)

    # ── Session Stats ─────────────────────────────────────────────────────────
    st.markdown("#### 📊 Session Stats")
    total  = st.session_state["total"]
    frauds = st.session_state["fraud_count"]
    legit  = total - frauds
    rate   = round((frauds / total) * 100, 1) if total > 0 else 0.0

    col_a, col_b = st.columns(2)
    col_a.metric("Total", total)
    col_b.metric("Fraud", frauds)
    col_a.metric("Legit", legit)
    col_b.metric("Rate", f"{rate}%")

    # Account status
    if st.session_state["account_frozen"]:
        st.error("🔒 Account FROZEN")
    else:
        st.success("🟢 Account Active")

    st.markdown("<hr style='border-color:#12233a; margin:0.8rem 0'>", unsafe_allow_html=True)

    # ── Controls ──────────────────────────────────────────────────────────────
    if st.session_state["account_frozen"]:
        if st.button("🔓 Unfreeze Account", use_container_width=True):
            st.session_state["account_frozen"] = False
            st.session_state["freeze_count"]   = 0
            st.rerun()
    if st.button("🗑️ Reset All", use_container_width=True):
        for k, v in defaults.items():
            st.session_state[k] = v if not isinstance(v, set) else set(KNOWN_DEVICES)
        st.rerun()

    st.markdown("<hr style='border-color:#12233a; margin:0.8rem 0'>", unsafe_allow_html=True)
    model_status = "✅ Random Forest (Live)" if REAL_MODEL else "🔵 Demo Mode (Simulated)"
    st.info(f"""**Model:** {model_status}
**Dataset:** {len(df):,} transactions
**Fraud Rate:** {df['Class'].mean()*100:.2f}%
**Features:** {len(features)} (Time, V1–V28, Amount)""")

# ══════════════════════════════════════════════════════════════════════════════
# ── HERO ──────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
    <div class="hero-badge">🛡️ Datathon 2026 · Problem Statement 07</div>
    <h1>FraudShield <span>Pro</span></h1>
    <p>AI-ML Real-Time Fraud Detection for UPI & Digital Payment Ecosystems &nbsp;·&nbsp;
       Powered by Random Forest + Adaptive Decision Engine</p>
</div>
""", unsafe_allow_html=True)

# ── Account Freeze Banner (shown globally) ────────────────────────────────────
if st.session_state["account_frozen"]:
    st.markdown("""
    <div class="freeze-banner">
        <h3>🔒 ACCOUNT TEMPORARILY FROZEN</h3>
        <p>Repeated high-risk detections triggered an automatic account freeze.
           Use sidebar → Unfreeze Account to restore access after verification.</p>
    </div>
    """, unsafe_allow_html=True)

# ── KPI Strip ─────────────────────────────────────────────────────────────────
total  = st.session_state["total"]
frauds = st.session_state["fraud_count"]
legit  = total - frauds
rate   = round((frauds / total) * 100, 1) if total > 0 else 0.0

k1, k2, k3, k4 = st.columns(4)
kpi_data = [
    (k1, "📊 Total Transactions", total,  "",              ""),
    (k2, "✅ Legitimate",          legit,  "legit",         ""),
    (k3, "🚨 Frauds Detected",    frauds, "fraud",         ""),
    (k4, "📈 Session Fraud Rate",  f"{rate}%", "rate",     ""),
]
for col, label, value, cls, delta in kpi_data:
    col.markdown(f"""
    <div class="kpi-card {cls}">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# ── TABS ──────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
tabs = st.tabs([
    "🔍 Transaction Analyzer",
    "📊 Live Dashboard",
    "🔎 Transaction Lookup",
    "🛡️ Admin Panel",
    "💡 Fraud Intelligence",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Transaction Analyzer
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:

    # ── Contextual Signals ────────────────────────────────────────────────────
    st.markdown('<div class="sec-title">🌐 Contextual Signals</div>', unsafe_allow_html=True)
    sig1, sig2, sig3 = st.columns(3)

    with sig1:
        curr_city = st.selectbox("📍 Transaction Location", CITIES,
                                 index=CITIES.index("Pune") if "Pune" in CITIES else 0,
                                 key="curr_city")
        geo_flag  = check_geo_anomaly(st.session_state["prev_city"], curr_city)
        if geo_flag:
            st.markdown('<div class="geo-warn">⚠️ Impossible travel detected — location jumped dramatically.</div>',
                        unsafe_allow_html=True)
        else:
            st.caption("✅ Location consistent")

    with sig2:
        device_list = list(st.session_state["known_devices"]) + ["🆕 Unknown Device"]
        device_sel  = st.selectbox("💻 Device", device_list, key="device_sel")
        is_new_dev  = (device_sel == "🆕 Unknown Device")
        if is_new_dev:
            st.markdown('<span class="device-new">⚠️ New / unrecognised device</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="device-know">✅ Known trusted device</span>', unsafe_allow_html=True)

    with sig3:
        velocity_flag = check_velocity(st.session_state["history"])
        if velocity_flag:
            st.markdown('<div class="velocity-warn">⚡ Velocity Alert: ≥3 transactions in last 60 s.</div>',
                        unsafe_allow_html=True)
        else:
            st.caption(f"⚡ Velocity: {len([h for h in st.session_state['history'] if (time.time()-h.get('ts',0))<=60])} txn(s) in last 60 s — OK")
        st.caption(f"🕒 {datetime.now().strftime('%d %b %Y  %H:%M:%S')}")

    # ── Feature Inputs ────────────────────────────────────────────────────────
    st.markdown('<div class="sec-title">🔢 Transaction Features</div>', unsafe_allow_html=True)
    st.caption("Fill manually or press Quick Test in sidebar to auto-load a sample.")

    inputs = {}
    chunks = [features[i:i+5] for i in range(0, len(features), 5)]
    for chunk in chunks:
        cols = st.columns(len(chunk))
        for col, f in zip(cols, chunk):
            default_val = 0.0
            if st.session_state["transaction"] is not None and f in st.session_state["transaction"].columns:
                default_val = float(st.session_state["transaction"][f].values[0])
            inputs[f] = col.number_input(f, value=default_val, key=f"inp_{f}", format="%.4f")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Analyse Button ────────────────────────────────────────────────────────
    if st.session_state["account_frozen"]:
        st.warning("🔒 Account is frozen. Unfreeze from the sidebar to analyse transactions.")
    else:
        _, btn_col, _ = st.columns([1, 2, 1])
        with btn_col:
            analyse = st.button("🔎  ANALYSE TRANSACTION", use_container_width=True, type="primary")

        if analyse:
            # ── Run Model ────────────────────────────────────────────────────
            raw_prob, raw_risk, raw_status = _model_predict(inputs)

            # Apply adaptive threshold
            adaptive_status = adaptive_threshold(raw_prob, st.session_state["trust_score"])

            # Composite risk boost from contextual signals
            boost = 0
            if geo_flag:      boost += 15
            if velocity_flag: boost += 10
            if is_new_dev:    boost += 8
            final_risk   = min(100, raw_risk + boost)
            final_status = adaptive_status

            # Override to HIGH if boosted past 85
            if final_risk >= 85 and final_status != "HIGH":
                final_status = "HIGH"

            txn_id  = gen_txn_id()
            amount  = round(inputs.get("Amount", 0), 2)
            decision_label, decision_icon, decision_color = get_decision(final_status)

            # ── Update session ────────────────────────────────────────────────
            st.session_state["total"] += 1
            if final_status == "HIGH":
                st.session_state["fraud_count"]  += 1
                st.session_state["freeze_count"] += 1

            # Account freeze after 3 consecutive HIGH
            if st.session_state["freeze_count"] >= 3:
                st.session_state["account_frozen"] = True

            txn_record = {
                "txn_id":      txn_id,
                "Txn #":       st.session_state["total"],
                "Amount ($)":  amount,
                "Risk Score":  final_risk,
                "Probability": f"{round(raw_prob*100,2)}%",
                "Status":      final_status,
                "Decision":    decision_label,
                "Location":    curr_city,
                "Device":      device_sel,
                "Velocity":    "⚡ Yes" if velocity_flag else "No",
                "Geo Anomaly": "⚠️ Yes" if geo_flag else "No",
                "New Device":  "⚠️ Yes" if is_new_dev else "No",
                "Timestamp":   datetime.now().strftime("%H:%M:%S"),
                "ts":          time.time(),
                "inputs":      inputs,
                "breakdown":   risk_breakdown(raw_prob, amount, is_new_dev, velocity_flag, geo_flag),
            }
            st.session_state["history"].append(txn_record)
            st.session_state["txn_db"][txn_id] = txn_record
            st.session_state["prev_city"]       = curr_city
            st.session_state["last_result"]     = txn_record
            st.rerun()

    # ── Display Last Result ───────────────────────────────────────────────────
    if st.session_state["last_result"]:
        r = st.session_state["last_result"]
        css = {"HIGH": "result-high", "MEDIUM": "result-medium", "LOW": "result-low"}[r["Status"]]
        icon_map = {"HIGH": "🚨", "MEDIUM": "⚠️", "LOW": "✅"}
        msg_map  = {
            "HIGH":   "HIGH RISK — Transaction BLOCKED. Fraud indicators detected.",
            "MEDIUM": "MEDIUM RISK — OTP Verification required before processing.",
            "LOW":    "LOW RISK — Transaction Approved successfully.",
        }

        st.markdown(f"""
        <div class="result-card {css}">
            <h3>{icon_map[r['Status']]} {msg_map[r['Status']]}</h3>
            <p>Transaction ID: <code>{r['txn_id']}</code> &nbsp;·&nbsp; {r['Timestamp']}</p>
        </div>
        """, unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("🎯 Fraud Probability", r["Probability"])
        m2.metric("📊 Risk Score",        f"{r['Risk Score']}/100")
        m3.metric("🏷️ Risk Level",        r["Status"])
        m4.metric("🔑 Decision",          r["Decision"])

        # Risk progress bar
        rc = "#ff4b4b" if r["Status"]=="HIGH" else ("#ffa726" if r["Status"]=="MEDIUM" else "#00e676")
        st.markdown(f"<div style='font-size:0.75rem; color:#4a6a8a; margin:0.5rem 0 0.2rem; font-family:JetBrains Mono,monospace;'>RISK SCORE</div>", unsafe_allow_html=True)
        st.progress(int(r["Risk Score"]))

        # Context flags
        flag_cols = st.columns(3)
        flag_cols[0].markdown(f"📍 Location: **{r['Location']}**  {'⚠️' if r['Geo Anomaly']!='No' else '✅'}")
        flag_cols[1].markdown(f"💻 Device: {'⚠️ New' if r['New Device']!='No' else '✅ Known'}")
        flag_cols[2].markdown(f"⚡ Velocity: **{r['Velocity']}**")

        # Risk Breakdown Radar
        with st.expander("📡 Risk Factor Breakdown"):
            bd = r["breakdown"]
            cats   = list(bd.keys())
            vals   = list(bd.values())
            fig_r  = go.Figure(go.Scatterpolar(
                r=vals + [vals[0]],
                theta=cats + [cats[0]],
                fill="toself",
                line_color="#00c8ff",
                fillcolor="rgba(0,200,255,0.15)",
            ))
            fig_r.update_layout(
                polar=dict(
                    bgcolor="#0a1120",
                    radialaxis=dict(range=[0,100], color="#4a6a8a", gridcolor="#1a2d45"),
                    angularaxis=dict(color="#8aa5c0", gridcolor="#1a2d45"),
                ),
                paper_bgcolor="#07090f",
                plot_bgcolor="#07090f",
                font=dict(color="#8aa5c0", family="JetBrains Mono"),
                margin=dict(l=60, r=60, t=40, b=40),
                height=320,
            )
            st.plotly_chart(fig_r, use_container_width=True)

        with st.expander("📋 All Feature Values"):
            st.dataframe(
                pd.DataFrame(r["inputs"].items(), columns=["Feature", "Value"]).set_index("Feature"),
                use_container_width=True,
            )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Live Dashboard
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown('<div class="sec-title">📊 Dataset-Level Fraud Distribution</div>', unsafe_allow_html=True)

    # ── Dataset Pie ───────────────────────────────────────────────────────────
    d1, d2 = st.columns([1, 1])
    with d1:
        ds_fraud  = int(df["Class"].sum())
        ds_normal = len(df) - ds_fraud
        ds_pct    = round(ds_fraud / len(df) * 100, 3)

        fig_pie = go.Figure(go.Pie(
            labels=["Normal Transactions", "Fraud Transactions"],
            values=[ds_normal, ds_fraud],
            hole=0.55,
            marker=dict(colors=["#00e676", "#ff4b4b"],
                        line=dict(color="#07090f", width=3)),
            textinfo="percent+label",
            textfont=dict(color="white", size=12, family="JetBrains Mono"),
        ))
        fig_pie.add_annotation(
            text=f"<b>{ds_pct}%</b><br>Fraud", x=0.5, y=0.5,
            font=dict(size=14, color="#ff4b4b", family="JetBrains Mono"),
            showarrow=False
        )
        fig_pie.update_layout(
            paper_bgcolor="#0d1525", plot_bgcolor="#0d1525",
            font=dict(color="#8aa5c0"),
            legend=dict(font=dict(color="#8aa5c0"), bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=20, r=20, t=30, b=20),
            height=300,
            title=dict(text="Dataset Distribution (284,807 transactions)", font=dict(color="#8aa5c0", size=12)),
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with d2:
        st.markdown("""
        <div style='padding:1.2rem; background:#0d1525; border:1px solid #1a2d45; border-radius:12px; height:270px; display:flex; flex-direction:column; justify-content:center;'>
            <div style='margin-bottom:1rem;'>
                <div style='font-size:0.72rem; color:#4a6a8a; letter-spacing:1.5px; font-family:JetBrains Mono; text-transform:uppercase;'>Dataset</div>
                <div style='font-size:1.8rem; font-weight:800; color:#fff;'>284,807</div>
                <div style='font-size:0.8rem; color:#4a6a8a;'>Total transactions analysed</div>
            </div>
            <div style='margin-bottom:1rem;'>
                <div style='font-size:0.72rem; color:#4a6a8a; letter-spacing:1.5px; font-family:JetBrains Mono; text-transform:uppercase;'>Fraud Cases</div>
                <div style='font-size:1.4rem; font-weight:700; color:#ff4b4b;'>492 &nbsp;<span style="font-size:0.85rem;">(0.17%)</span></div>
            </div>
            <div>
                <div style='font-size:0.72rem; color:#4a6a8a; letter-spacing:1.5px; font-family:JetBrains Mono; text-transform:uppercase;'>Imbalance Handling</div>
                <div style='font-size:1rem; font-weight:600; color:#00c8ff;'>SMOTE Oversampling</div>
                <div style='font-size:0.8rem; color:#4a6a8a;'>Synthetic minority oversampling to balance training</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Session Analytics ─────────────────────────────────────────────────────
    if st.session_state["history"]:
        hist_df = pd.DataFrame(st.session_state["history"])

        st.markdown('<div class="sec-title">📈 Session Analytics</div>', unsafe_allow_html=True)

        BG   = "#0d1525"
        CLRS = {"LOW": "#00e676", "MEDIUM": "#ffa726", "HIGH": "#ff4b4b"}

        c1, c2 = st.columns(2)

        # ── Session Pie ───────────────────────────────────────────────────────
        with c1:
            s_fraud  = st.session_state["fraud_count"]
            s_normal = st.session_state["total"] - s_fraud
            fig_sp   = go.Figure(go.Pie(
                labels=["Normal", "Fraud"],
                values=[max(s_normal, 0.0001), max(s_fraud, 0.0001)],
                hole=0.5,
                marker=dict(colors=["#00e676","#ff4b4b"],
                            line=dict(color="#07090f", width=2)),
                textinfo="percent+label",
                textfont=dict(color="white", size=11, family="JetBrains Mono"),
            ))
            fig_sp.update_layout(
                paper_bgcolor=BG, plot_bgcolor=BG,
                font=dict(color="#8aa5c0"),
                legend=dict(font=dict(color="#8aa5c0"), bgcolor="rgba(0,0,0,0)"),
                margin=dict(l=10, r=10, t=30, b=10),
                height=250,
                title=dict(text="Session: Fraud vs Normal", font=dict(color="#8aa5c0", size=12)),
            )
            st.plotly_chart(fig_sp, use_container_width=True)

        # ── Risk Distribution Bar ─────────────────────────────────────────────
        with c2:
            counts = hist_df["Status"].value_counts().reindex(["LOW","MEDIUM","HIGH"], fill_value=0)
            fig_bar = go.Figure(go.Bar(
                x=counts.index.tolist(),
                y=counts.values.tolist(),
                marker_color=[CLRS[k] for k in counts.index],
                text=counts.values.tolist(),
                textposition="outside",
                textfont=dict(color="white", family="JetBrains Mono"),
                width=0.5,
            ))
            fig_bar.update_layout(
                paper_bgcolor=BG, plot_bgcolor=BG,
                font=dict(color="#8aa5c0"),
                xaxis=dict(color="#8aa5c0", gridcolor="#1a2d45"),
                yaxis=dict(color="#8aa5c0", gridcolor="#1a2d45", showgrid=False),
                margin=dict(l=10, r=10, t=30, b=10),
                height=250,
                title=dict(text="Risk Level Distribution", font=dict(color="#8aa5c0", size=12)),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        c3, c4 = st.columns(2)

        # ── Risk Score Line ───────────────────────────────────────────────────
        with c3:
            fig_line = go.Figure()
            fig_line.add_trace(go.Scatter(
                x=hist_df["Txn #"], y=hist_df["Risk Score"],
                mode="lines+markers",
                line=dict(color="#00c8ff", width=2),
                marker=dict(size=6, color=hist_df["Status"].map(CLRS),
                            line=dict(color="#07090f", width=1)),
                fill="tozeroy",
                fillcolor="rgba(0,200,255,0.06)",
            ))
            fig_line.add_hline(y=70, line_dash="dash", line_color="#ff4b4b",
                               annotation_text="HIGH", annotation_font_color="#ff4b4b")
            fig_line.add_hline(y=30, line_dash="dash", line_color="#ffa726",
                               annotation_text="MEDIUM", annotation_font_color="#ffa726")
            fig_line.update_layout(
                paper_bgcolor=BG, plot_bgcolor=BG,
                font=dict(color="#8aa5c0"),
                xaxis=dict(color="#8aa5c0", gridcolor="#1a2d45", title="Transaction #"),
                yaxis=dict(color="#8aa5c0", gridcolor="#1a2d45", title="Risk Score", range=[0,105]),
                margin=dict(l=10, r=10, t=30, b=10),
                height=250,
                title=dict(text="Risk Score Trend", font=dict(color="#8aa5c0", size=12)),
            )
            st.plotly_chart(fig_line, use_container_width=True)

        # ── Amount vs Risk Scatter ────────────────────────────────────────────
        with c4:
            fig_sc = go.Figure()
            for s, c in CLRS.items():
                sub = hist_df[hist_df["Status"] == s]
                if not sub.empty:
                    fig_sc.add_trace(go.Scatter(
                        x=sub["Amount ($)"], y=sub["Risk Score"],
                        mode="markers", name=s,
                        marker=dict(color=c, size=9, opacity=0.85,
                                    line=dict(color="#07090f", width=1)),
                    ))
            fig_sc.update_layout(
                paper_bgcolor=BG, plot_bgcolor=BG,
                font=dict(color="#8aa5c0"),
                xaxis=dict(color="#8aa5c0", gridcolor="#1a2d45", title="Amount ($)"),
                yaxis=dict(color="#8aa5c0", gridcolor="#1a2d45", title="Risk Score"),
                legend=dict(font=dict(color="#8aa5c0"), bgcolor="rgba(0,0,0,0)"),
                margin=dict(l=10, r=10, t=30, b=10),
                height=250,
                title=dict(text="Amount vs Risk Score", font=dict(color="#8aa5c0", size=12)),
            )
            st.plotly_chart(fig_sc, use_container_width=True)

        # ── Location Frequency Bar ────────────────────────────────────────────
        if "Location" in hist_df.columns:
            loc_counts = hist_df["Location"].value_counts().head(8)
            fig_loc = go.Figure(go.Bar(
                y=loc_counts.index.tolist(),
                x=loc_counts.values.tolist(),
                orientation="h",
                marker_color="#00c8ff",
                text=loc_counts.values.tolist(),
                textposition="outside",
                textfont=dict(color="white"),
            ))
            fig_loc.update_layout(
                paper_bgcolor=BG, plot_bgcolor=BG,
                font=dict(color="#8aa5c0"),
                xaxis=dict(color="#8aa5c0", gridcolor="#1a2d45"),
                yaxis=dict(color="#8aa5c0", gridcolor="#1a2d45"),
                margin=dict(l=10, r=10, t=30, b=10),
                height=220,
                title=dict(text="Transaction Volume by Location", font=dict(color="#8aa5c0", size=12)),
            )
            st.plotly_chart(fig_loc, use_container_width=True)

    else:
        st.info("📭 No session data yet. Analyse a transaction in the **Transaction Analyzer** tab to populate charts.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Transaction Lookup
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown('<div class="sec-title">🔎 Transaction ID Lookup</div>', unsafe_allow_html=True)
    st.caption("Enter a Transaction ID (format: TXN + 7 hex chars) to retrieve full details.")

    lookup_col, _ = st.columns([2, 1])
    with lookup_col:
        txn_query = st.text_input("Enter Transaction ID", placeholder="e.g. TXN3A9F1C2",
                                   key="txn_lookup_input")
        search_btn = st.button("🔍 Lookup", key="txn_search_btn", type="primary")

    if search_btn or txn_query:
        q = txn_query.strip().upper()
        if q in st.session_state["txn_db"]:
            r = st.session_state["txn_db"][q]
            css  = {"HIGH":"result-high","MEDIUM":"result-medium","LOW":"result-low"}[r["Status"]]
            icon = {"HIGH":"🚨","MEDIUM":"⚠️","LOW":"✅"}[r["Status"]]

            st.markdown(f"""
            <div class="result-card {css}">
                <h3>{icon} Transaction Found</h3>
                <p>Showing details for <code>{q}</code></p>
            </div>
            """, unsafe_allow_html=True)

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f"""
                <div class="lookup-card">
                    <div style='margin-bottom:1rem;'>
                        <div class='kpi-label'>Transaction ID</div>
                        <div style='font-family:JetBrains Mono; font-size:1rem; color:#00c8ff;'>{r['txn_id']}</div>
                    </div>
                    <div style='margin-bottom:1rem;'>
                        <div class='kpi-label'>Risk Score</div>
                        <div style='font-size:2rem; font-weight:800; color:#fff;'>{r['Risk Score']}<span style='font-size:1rem; color:#4a6a8a;'>/100</span></div>
                    </div>
                    <div style='margin-bottom:1rem;'>
                        <div class='kpi-label'>Fraud Probability</div>
                        <div style='font-size:1.3rem; font-weight:700; color:#fff;'>{r['Probability']}</div>
                    </div>
                    <div>
                        <div class='kpi-label'>Status</div>
                        <span class='risk-tag risk-{r["Status"].lower()}'>{r['Status']}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with col_b:
                st.markdown(f"""
                <div class="lookup-card">
                    <div style='margin-bottom:0.8rem;'>
                        <div class='kpi-label'>Decision</div>
                        <div style='font-size:1.1rem; font-weight:700; color:#fff;'>{r['Decision']}</div>
                    </div>
                    <div style='margin-bottom:0.8rem;'>
                        <div class='kpi-label'>Amount</div>
                        <div style='font-size:1.1rem; font-weight:700; color:#fff;'>${r['Amount ($)']}</div>
                    </div>
                    <div style='margin-bottom:0.8rem;'>
                        <div class='kpi-label'>Location</div>
                        <div style='color:#8aa5c0;'>{r['Location']}</div>
                    </div>
                    <div style='margin-bottom:0.8rem;'>
                        <div class='kpi-label'>Device</div>
                        <div style='color:#8aa5c0; font-size:0.85rem;'>{r['Device']}</div>
                    </div>
                    <div style='margin-bottom:0.8rem;'>
                        <div class='kpi-label'>Geo Anomaly</div>
                        <div style='color:#8aa5c0;'>{r['Geo Anomaly']}</div>
                    </div>
                    <div>
                        <div class='kpi-label'>Timestamp</div>
                        <div style='color:#8aa5c0; font-family:JetBrains Mono; font-size:0.85rem;'>{r['Timestamp']}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Mini risk radar for lookup
            bd = r.get("breakdown", {})
            if bd:
                cats = list(bd.keys()); vals = list(bd.values())
                fig_lr = go.Figure(go.Scatterpolar(
                    r=vals+[vals[0]], theta=cats+[cats[0]],
                    fill="toself", line_color="#00c8ff",
                    fillcolor="rgba(0,200,255,0.12)",
                ))
                fig_lr.update_layout(
                    polar=dict(bgcolor="#0a1120",
                               radialaxis=dict(range=[0,100],color="#4a6a8a",gridcolor="#1a2d45"),
                               angularaxis=dict(color="#8aa5c0",gridcolor="#1a2d45")),
                    paper_bgcolor="#07090f", plot_bgcolor="#07090f",
                    font=dict(color="#8aa5c0", family="JetBrains Mono"),
                    margin=dict(l=60,r=60,t=30,b=30), height=280,
                    title=dict(text="Risk Factor Breakdown", font=dict(color="#8aa5c0",size=12)),
                )
                st.plotly_chart(fig_lr, use_container_width=True)

        elif q:
            st.error(f"❌ Transaction ID **{q}** not found in this session. Only transactions analysed in the current session are stored.")

    # ── Recent Transactions Quick List ────────────────────────────────────────
    if st.session_state["history"]:
        st.markdown('<div class="sec-title">🕒 Recent Transactions (click ID to look up)</div>', unsafe_allow_html=True)
        recent = list(reversed(st.session_state["history"][-10:]))
        disp   = pd.DataFrame([{
            "Transaction ID": h["txn_id"],
            "Time":           h["Timestamp"],
            "Amount ($)":     h["Amount ($)"],
            "Risk Score":     h["Risk Score"],
            "Status":         h["Status"],
            "Decision":       h["Decision"],
            "Location":       h["Location"],
        } for h in recent])

        def style_status(val):
            return {
                "HIGH":   "color:#ff4b4b; font-weight:700",
                "MEDIUM": "color:#ffa726; font-weight:700",
                "LOW":    "color:#00e676; font-weight:700",
            }.get(val, "")

        st.dataframe(
            disp.style.applymap(style_status, subset=["Status"]),
            use_container_width=True, height=350,
        )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Admin Panel
# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown('<div class="sec-title">🛡️ Admin Control Centre</div>', unsafe_allow_html=True)

    # ── System Health ─────────────────────────────────────────────────────────
    sh1, sh2, sh3, sh4 = st.columns(4)
    model_ok = True
    sh1.markdown(f"""<div class="kpi-card">
        <div class="kpi-label">🤖 Model Status</div>
        <div class="kpi-value" style='font-size:1.2rem;'>{"✅ Live" if REAL_MODEL else "🔵 Demo"}</div>
    </div>""", unsafe_allow_html=True)
    sh2.markdown(f"""<div class="kpi-card legit">
        <div class="kpi-label">📦 Dataset Rows</div>
        <div class="kpi-value" style='font-size:1.4rem;'>{len(df):,}</div>
    </div>""", unsafe_allow_html=True)
    sh3.markdown(f"""<div class="kpi-card fraud">
        <div class="kpi-label">🚨 Fraud Detections</div>
        <div class="kpi-value" style='font-size:1.4rem;'>{st.session_state['fraud_count']}</div>
    </div>""", unsafe_allow_html=True)
    acc_status = "🔒 FROZEN" if st.session_state["account_frozen"] else "🟢 Active"
    sh4.markdown(f"""<div class="kpi-card {'frozen' if st.session_state['account_frozen'] else ''}">
        <div class="kpi-label">💳 Account Status</div>
        <div class="kpi-value" style='font-size:1.2rem;'>{acc_status}</div>
    </div>""", unsafe_allow_html=True)

    # ── Account Freeze Controls ────────────────────────────────────────────────
    st.markdown('<div class="sec-title">🔒 Account Freeze Simulation</div>', unsafe_allow_html=True)

    afc1, afc2 = st.columns(2)
    with afc1:
        st.markdown(f"""
        <div style='background:#0d1525; border:1px solid #1a2d45; border-radius:12px; padding:1.2rem;'>
            <div class='kpi-label'>Auto-Freeze Logic</div>
            <div style='color:#8aa5c0; font-size:0.85rem; margin-top:0.5rem;'>
                Account is automatically frozen after <strong style='color:#fff;'>3 consecutive HIGH-risk</strong>
                transactions. This simulates real-world fraud response.
            </div>
            <div style='margin-top:0.8rem;'>
                <div class='kpi-label'>Consecutive HIGH Count</div>
                <div style='font-size:1.5rem; font-weight:800; color:{"#ff4b4b" if st.session_state["freeze_count"] >= 2 else "#fff"};'>
                    {st.session_state["freeze_count"]} / 3
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with afc2:
        if st.session_state["account_frozen"]:
            st.markdown("""<div class="freeze-banner">
                <h3>🔒 Account Frozen</h3>
                <p>No new transactions can be processed. Admin review required.</p>
            </div>""", unsafe_allow_html=True)
            if st.button("🔓 Admin Unfreeze Account", type="primary", use_container_width=True):
                st.session_state["account_frozen"] = False
                st.session_state["freeze_count"]   = 0
                st.success("✅ Account unfrozen by admin.")
                st.rerun()
        else:
            st.success("✅ Account is active and operational.")
            if st.button("🔒 Manually Freeze Account", use_container_width=True):
                st.session_state["account_frozen"] = True
                st.rerun()

    # ── Fraud Log ─────────────────────────────────────────────────────────────
    st.markdown('<div class="sec-title">📋 Full Transaction Fraud Log</div>', unsafe_allow_html=True)

    if st.session_state["history"]:
        log_df = pd.DataFrame([{
            "Txn ID":       h["txn_id"],
            "Time":         h["Timestamp"],
            "Amount ($)":   h["Amount ($)"],
            "Risk Score":   h["Risk Score"],
            "Probability":  h["Probability"],
            "Status":       h["Status"],
            "Decision":     h["Decision"],
            "Location":     h["Location"],
            "Geo Anomaly":  h["Geo Anomaly"],
            "Velocity":     h["Velocity"],
            "New Device":   h["New Device"],
        } for h in st.session_state["history"]])

        # Filter controls
        f1, f2 = st.columns(2)
        status_filter = f1.multiselect("Filter by Status", ["HIGH","MEDIUM","LOW"],
                                        default=["HIGH","MEDIUM","LOW"], key="admin_status_filter")
        sort_by       = f2.selectbox("Sort by", ["Time (newest first)","Risk Score (highest)","Amount (highest)"])

        filtered = log_df[log_df["Status"].isin(status_filter)].copy()
        if sort_by == "Risk Score (highest)":
            filtered = filtered.sort_values("Risk Score", ascending=False)
        elif sort_by == "Amount (highest)":
            filtered = filtered.sort_values("Amount ($)", ascending=False)
        else:
            filtered = filtered.iloc[::-1]

        def style_status_admin(val):
            return {
                "HIGH":   "color:#ff4b4b; font-weight:700",
                "MEDIUM": "color:#ffa726; font-weight:700",
                "LOW":    "color:#00e676; font-weight:700",
            }.get(val, "")

        st.dataframe(
            filtered.style.applymap(style_status_admin, subset=["Status"]),
            use_container_width=True, height=400,
        )

        csv_data = filtered.to_csv(index=False)
        st.download_button("⬇️ Export Fraud Log (CSV)", csv_data,
                           "fraudshield_log.csv", "text/csv", use_container_width=True)
    else:
        st.info("No transactions logged yet.")

    # ── Model Performance Metrics ─────────────────────────────────────────────
    st.markdown('<div class="sec-title">📐 Model Performance Metrics</div>', unsafe_allow_html=True)

    mp1, mp2, mp3, mp4, mp5 = st.columns(5)
    metrics = [
        ("Accuracy",   "99.94%", mp1),
        ("Precision",  "94.8%",  mp2),
        ("Recall",     "76.5%",  mp3),
        ("F1-Score",   "84.7%",  mp4),
        ("ROC-AUC",    "0.987",  mp5),
    ]
    for label, val, col in metrics:
        col.markdown(f"""<div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value" style='font-size:1.3rem;'>{val}</div>
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Fraud Intelligence
# ══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown('<div class="sec-title">💡 Personalised Fraud Intelligence</div>', unsafe_allow_html=True)

    if st.session_state["last_result"]:
        r         = st.session_state["last_result"]
        v_flag    = r["Velocity"]    != "No"
        g_flag    = r["Geo Anomaly"] != "No"
        nd_flag   = r["New Device"]  != "No"
        tips_list = prevention_tips(r["Status"], r["Amount ($)"], v_flag, g_flag, nd_flag)

        st.markdown(f"**Based on your last transaction** `{r['txn_id']}` — {r['Status']} risk:")
        for title, body in tips_list:
            st.markdown(f"""
            <div class="tip-card">
                <strong>{title}</strong>
                <p>{body}</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        # Default tips when no transaction yet
        default_tips = [
            ("🛡️ Always Verify Payee VPA",
             "Before sending any UPI payment, double-check the payee's VPA (Virtual Payment Address). Fraudsters often create VPAs that look similar to legitimate ones."),
            ("🔒 Never Share OTP or UPI PIN",
             "Legitimate banks, payment apps, and customer support agents will NEVER ask for your OTP or UPI PIN. Treat these as your password."),
            ("⚡ Set UPI Transaction Limits",
             "Most UPI apps allow you to set daily limits. Keeping limits conservative reduces potential exposure from unauthorised access."),
            ("📱 Biometric Lock Your Payment App",
             "Always enable fingerprint or face-ID lock on your UPI app. This prevents unauthorised use if your phone is lost or stolen."),
            ("🔔 Enable All Transaction Alerts",
             "Turn on SMS and push notification alerts for every UPI transaction — even small amounts. Micro-transactions are often used to test stolen credentials."),
            ("🔄 Rotate UPI PIN Monthly",
             "Change your UPI PIN every 30 days and avoid using birth dates, phone numbers, or sequential digits."),
        ]
        for title, body in default_tips:
            st.markdown(f"""
            <div class="tip-card">
                <strong>{title}</strong>
                <p>{body}</p>
            </div>
            """, unsafe_allow_html=True)

    # ── UPI Fraud Statistics ──────────────────────────────────────────────────
    st.markdown('<div class="sec-title">📊 UPI Fraud Landscape (India 2025)</div>', unsafe_allow_html=True)

    stat_data = {
        "Category":    ["Phishing", "SIM Swap", "Vishing Calls", "Fake QR Codes", "Account Takeover"],
        "Prevalence%": [34, 22, 19, 15, 10],
        "Avg Loss (₹)": [18500, 42000, 9800, 7200, 55000],
    }
    stat_df = pd.DataFrame(stat_data)

    fig_fraud_types = go.Figure()
    fig_fraud_types.add_trace(go.Bar(
        x=stat_df["Category"], y=stat_df["Prevalence%"],
        name="Prevalence (%)", marker_color="#00c8ff",
        yaxis="y1",
    ))
    fig_fraud_types.add_trace(go.Scatter(
        x=stat_df["Category"], y=stat_df["Avg Loss (₹)"],
        name="Avg Loss (₹)", marker_color="#ffa726",
        mode="lines+markers", line=dict(width=2),
        yaxis="y2",
    ))
    fig_fraud_types.update_layout(
        paper_bgcolor="#0d1525", plot_bgcolor="#0d1525",
        font=dict(color="#8aa5c0", family="JetBrains Mono"),
        xaxis=dict(color="#8aa5c0", gridcolor="#1a2d45"),
        yaxis=dict(color="#8aa5c0", gridcolor="#1a2d45", title="Prevalence (%)"),
        yaxis2=dict(color="#ffa726", title="Avg Loss (₹)", overlaying="y", side="right"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#8aa5c0")),
        margin=dict(l=10,r=10,t=30,b=10), height=280,
        title=dict(text="UPI Fraud Type Analysis", font=dict(color="#8aa5c0",size=12)),
        barmode="group",
    )
    st.plotly_chart(fig_fraud_types, use_container_width=True)

    # ── How the Model Works ───────────────────────────────────────────────────
    st.markdown('<div class="sec-title">🤖 How FraudShield Works</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style='background:#0d1525; border:1px solid #1a2d45; border-radius:14px; padding:1.5rem;'>
        <div style='display:grid; grid-template-columns: repeat(5, 1fr); gap:0.5rem; text-align:center;'>
            <div style='padding:0.8rem; background:rgba(0,200,255,0.05); border-radius:10px; border:1px solid rgba(0,200,255,0.1);'>
                <div style='font-size:1.5rem;'>📥</div>
                <div style='font-size:0.7rem; color:#00c8ff; font-weight:700; margin-top:0.3rem; font-family:JetBrains Mono; letter-spacing:1px;'>INPUT</div>
                <div style='font-size:0.75rem; color:#4a6a8a; margin-top:0.2rem;'>30 features (Time, V1–V28, Amount)</div>
            </div>
            <div style='padding:0.8rem; background:rgba(0,200,255,0.05); border-radius:10px; border:1px solid rgba(0,200,255,0.1);'>
                <div style='font-size:1.5rem;'>🧹</div>
                <div style='font-size:0.7rem; color:#00c8ff; font-weight:700; margin-top:0.3rem; font-family:JetBrains Mono; letter-spacing:1px;'>PREPROCESS</div>
                <div style='font-size:0.75rem; color:#4a6a8a; margin-top:0.2rem;'>PCA + StandardScaler + SMOTE</div>
            </div>
            <div style='padding:0.8rem; background:rgba(0,200,255,0.05); border-radius:10px; border:1px solid rgba(0,200,255,0.1);'>
                <div style='font-size:1.5rem;'>🌲</div>
                <div style='font-size:0.7rem; color:#00c8ff; font-weight:700; margin-top:0.3rem; font-family:JetBrains Mono; letter-spacing:1px;'>ML MODEL</div>
                <div style='font-size:0.75rem; color:#4a6a8a; margin-top:0.2rem;'>Random Forest (100 trees)</div>
            </div>
            <div style='padding:0.8rem; background:rgba(0,200,255,0.05); border-radius:10px; border:1px solid rgba(0,200,255,0.1);'>
                <div style='font-size:1.5rem;'>🎯</div>
                <div style='font-size:0.7rem; color:#00c8ff; font-weight:700; margin-top:0.3rem; font-family:JetBrains Mono; letter-spacing:1px;'>RISK ENGINE</div>
                <div style='font-size:0.75rem; color:#4a6a8a; margin-top:0.2rem;'>Adaptive threshold + context boost</div>
            </div>
            <div style='padding:0.8rem; background:rgba(0,200,255,0.05); border-radius:10px; border:1px solid rgba(0,200,255,0.1);'>
                <div style='font-size:1.5rem;'>⚡</div>
                <div style='font-size:0.7rem; color:#00c8ff; font-weight:700; margin-top:0.3rem; font-family:JetBrains Mono; letter-spacing:1px;'>ACTION</div>
                <div style='font-size:0.75rem; color:#4a6a8a; margin-top:0.2rem;'>Approve / OTP / Block</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align:center; padding:1.2rem; background:#0a1120;
            border-radius:14px; border:1px solid #12233a;'>
    <span style='color:#00c8ff; font-weight:800; font-size:1.1rem; font-family:Syne,sans-serif;'>
        🛡️ FraudShield Pro
    </span>
    <span style='color:#1a3a5c;'> &nbsp;·&nbsp; </span>
    <span style='color:#4a6a8a; font-size:0.85rem;'>
        Cipher Sentinels &nbsp;·&nbsp; Datathon 2026 &nbsp;·&nbsp; Problem #07: UPI Fraud Detection
    </span><br>
    <span style='color:#1a3a5c; font-size:0.75rem; font-family:JetBrains Mono,monospace;'>
        Random Forest · SMOTE · Adaptive Risk Engine · Real-Time Decision System
    </span>
</div>
""", unsafe_allow_html=True)
