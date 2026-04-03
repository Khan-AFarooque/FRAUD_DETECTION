from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
import joblib
import os
import random

app = Flask(__name__, static_folder='static')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load model and data
model = joblib.load(os.path.join(BASE_DIR, "fraud_model.pkl"))
normal_df = pd.read_csv(os.path.join(BASE_DIR, "normal_small.csv"))
fraud_df = pd.read_csv(os.path.join(BASE_DIR, "fraud_small.csv"))

FEATURES = normal_df.columns.tolist()
NORMAL_MEANS = normal_df[FEATURES].mean()
NORMAL_STDS = normal_df[FEATURES].std().replace(0, 1)


@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


@app.route('/api/features')
def get_features():
    return jsonify({"features": FEATURES})


@app.route('/api/sample')
def sample_transaction():
    mode = request.args.get('mode', 'normal')
    df = fraud_df if mode == 'fraud' else normal_df
    row = df.sample(1).iloc[0]
    return jsonify({f: round(float(row[f]), 6) for f in FEATURES})


@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json()
    values = [float(data.get(f, 0)) for f in FEATURES]
    input_df = pd.DataFrame([values], columns=FEATURES)

    prob = float(model.predict_proba(input_df)[0][1])
    risk_score = round(prob * 100, 2)

    # Compute Explainable AI Risk Factors
    z_scores = []
    for f, val in zip(FEATURES, values):
        mean_val = NORMAL_MEANS[f]
        std_val = NORMAL_STDS[f]
        z = abs(val - mean_val) / std_val
        z_scores.append((f, val, mean_val, z))

    z_scores.sort(key=lambda x: x[3], reverse=True)
    top_3 = z_scores[:3]

    risk_factors = []
    if risk_score > 30:
        for i, (f, val, mean_val, z) in enumerate(top_3):
            if i == 0 and risk_score > 70:
                risk_factors.append(f"• {f} pattern matches fraud behavior")
            elif val > mean_val:
                risk_factors.append(f"• {f} shows extreme deviation from normal range")
            else:
                risk_factors.append(f"• {f} negative spike detected")
    else:
        risk_factors = [
            "• All features within normal ranges",
            "• No significant deviations",
            "• Behavioral patterns normal"
        ]

    if risk_score > 70:
        status = "HIGH"
        decision = "BLOCKED"
        explanation = "Transaction exhibits multiple anomalous patterns and is classified as high-risk with strong confidence."
    elif risk_score > 30:
        status = "MEDIUM"
        decision = "OTP_REQUIRED"
        explanation = "Transaction shows suspicious behavior. Moderate anomalies detected requiring further verification."
    else:
        status = "LOW"
        decision = "APPROVED"
        explanation = "Transaction exhibits normal behavior. No significant anomalies detected."

    return jsonify({
        "probability": round(prob, 6),
        "riskScore": risk_score,
        "status": status,
        "decision": decision,
        "explanation": explanation,
        "riskFactors": risk_factors
    })


if __name__ == '__main__':
    app.run(debug=True, port=5000)
