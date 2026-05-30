import streamlit as st
import pandas as pd
import numpy as np
import pickle
from ultralytics import YOLO
from PIL import Image
import shap
from transformers import pipeline
import torch
from streamlit.components.v1 import html

st.set_page_config(page_title="AI Vehicle System", layout="wide")

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: white;
}

h1, h2, h3 {
    color: #f8fafc;
}

div.stButton > button {
    background-color: #2563eb;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 16px;
    font-weight: bold;
}

div.stButton > button:hover {
    background-color: #1d4ed8;
}

.card {
    background-color: #1e293b;
    padding: 20px;
    border-radius: 15px;
    margin-top: 10px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.3);
}
</style>
""", unsafe_allow_html=True)

# =========================
# LOAD MODELS
# =========================

# Loading our trained model

car_model = pickle.load(open("models/car_model.pkl", "rb"))
car_cols = pickle.load(open("models/car_columns.pkl", "rb"))

bike_model = pickle.load(open("models/bike_model.pkl", "rb"))
bike_cols = pickle.load(open("models/bike_columns.pkl", "rb"))

@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)
def load_sentiment_model():
    device = 0 if torch.cuda.is_available() else -1
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        framework="pt",
        device=device
    )

sentiment_model = load_sentiment_model()

rules_df = pd.read_csv("dataset/Inspection rule dataset.csv")

# =========================
# HELPER FUNCTIONS
# =========================

# Preprocess input for car price prediction

def owner_penalty_category(owner):
    mapping = {
        'First': 1,
        'Second': 2,
        'Third': 3,
        'Fourth': 4
    }
    return mapping.get(owner, 0)  # default = 0 if unknown

def preprocess_car(name, kms, year, owner, fuel, transmission):
    df = pd.DataFrame([{
        "name": name,
        "kms_driven": kms,
        "registration_year": year,
        "owner_penalty": owner,
        "fuel_type": fuel,
        "transmission": transmission
    }])

    df['brand'] = df['name'].apply(lambda x: x.split()[0])
    df['vehicle_age'] = 2024 - df['registration_year']
    df['usage_intensity'] = df['kms_driven'] / (df['vehicle_age'] + 1)
    df['owner_penalty'] = df['owner_penalty'].apply(owner_penalty_category)
    df = pd.get_dummies(df, columns=['brand','fuel_type','transmission'])
    df = df.drop(['name', 'registration_year' ], axis=1)

    return df

# Preprocess input for bike price prediction

def brand_category(brand):
    premium = ['BMW', 'Kawasaki', 'Harley-Davidson']
    mid = ['Royal Enfield', 'KTM', 'Yamaha']        
    budget = ['Hero', 'Bajaj', 'TVS', 'Honda']

    if brand in premium:
        return 3
    elif brand in mid:
        return 2
    elif brand in budget:
        return 1
    else:
        return 1

def preprocess_bike(name, kms, year, owner):
    df = pd.DataFrame([{
        "name": name,
        "kms_driven": kms,
        "registration_year": year,
        "owner": owner
    }])

    df['brand'] = df['name'].apply(lambda x: x.split()[0])
    df['vehicle_age'] = 2022 - df['registration_year']
    df['brand_category'] = df['brand'].apply(brand_category)
    df['usage_intensity'] = df['kms_driven'] / (df['vehicle_age'] + 1)
    df['owner_penalty'] = df['owner'].apply(owner_penalty_category)
    df = pd.get_dummies(df, columns=['brand'])
    df = df.drop(['name', 'registration_year', 'owner'], axis=1)

    return df

# Unified prediction function with SHAP value extraction

def get_shap_values(vehicle_type, df):

    if vehicle_type == "Car":
        model_fn = lambda x: car_model.predict(x)
    else:
        model_fn = lambda x: bike_model.predict(x)

    explainer = shap.Explainer(model_fn, df)

    shap_values = explainer(df)

    return shap_values.values

def shap_to_dataframe(df, shap_values):
    shap_df = pd.DataFrame({
        "feature": df.columns,
        "value": df.iloc[0].values,
        "impact": shap_values[0]
    })

    shap_df["abs_impact"] = np.abs(shap_df["impact"])
    shap_df = shap_df.sort_values(by="abs_impact", ascending=False)

    return shap_df

def predict(vehicle_type, input_data):
    
    if vehicle_type == "Car":
        df = preprocess_car(**input_data)
        df = df.reindex(columns=car_cols, fill_value=0)
        pred = car_model.predict(df)
        pred = np.expm1(pred)[0]

    else:
        df = preprocess_bike(**input_data)
        df = df.reindex(columns=bike_cols, fill_value=0)
        pred = bike_model.predict(df)
        pred = pred[0]

    # SHAP values
    df = df.astype(float)
    shap_values = get_shap_values(vehicle_type, df)

    # Convert to table
    shap_df = shap_to_dataframe(df, shap_values)

    shap_df = shap_df[~(
        (shap_df["feature"].str.startswith("brand_")) &
        (shap_df["value"] == 0)
    )]

    # Range
    lower = pred * 0.9
    upper = pred * 1.1

    return shap_df, int(pred), int(lower), int(upper)

# =========================
# SENTIMENT ANALYSIS (SIMULATED)
# ========================= 

def get_sentiment_score(text):
    if not text.strip():
        return 0

    result = sentiment_model(text[:512])[0]  # limit length

    label = result['label']
    score = result['score']

    if label == "POSITIVE":
        return score
    else:
        return -score
 
# =========================
# DAMAGE MODEL (SIMULATED)
# =========================

def run_detection(image, vehicle_type):
    
    if vehicle_type == "Bike":
        model = load_model("models/two_wheeler.pt")
    else:
        model = load_model("models/four_wheeler.pt")

    img_array = np.array(image)
    results = model(img_array)

    # Initialize counts (IMPORTANT)
    dent_count = 0
    scratch_count = 0
    rust_count = 0

    detections = []

    # Loop through detections
    for box in results[0].boxes:
        cls = int(box.cls[0])
        label = model.names[cls].lower()  # make lowercase for safety

        detections.append(label)

        if "dent" in label:
            dent_count += 1
        elif "scratch" in label:
            scratch_count += 1
        elif "rust" in label:
            rust_count += 1

    return detections, dent_count, scratch_count, rust_count

# =========================
# INSPECTION FUNCTIONS
# =========================

def map_to_condition_flags(vehicle_info, yolo_output):
    flags = []

    if vehicle_info["kms_driven"] > 60000:
        flags.append("high_km")

    if vehicle_info["age"] > 7:
        flags.append("old_vehicle")

    if any(x in vehicle_info["owner"] for x in ["Second", "Third", "Fourth"]):
        flags.append("more_owners")

    if yolo_output["damage_count"].get("dent", 0) > 0:
        flags.append("dent_detected")

    if yolo_output["damage_count"].get("scratch", 0) > 0:
        flags.append("scratch_detected")

    if yolo_output["damage_count"].get("rust", 0) > 0:
        flags.append("rust_detected")

    return flags

def generate_inspection_checklist(vehicle_type, condition_flags, rules_df):
    checklist = []

    for flag in condition_flags:
        matched = rules_df[
            ((rules_df['vehicle_type'] == vehicle_type) |
             (rules_df['vehicle_type'] == 'any')) &
            (rules_df['condition'] == flag)
        ]

        for _, row in matched.iterrows():
            checklist.append({
                "component": row["component"],
                "recommendation": row["recommendation"],
                "severity": row["severity"],
                "triggered_by": flag
            })

    severity_order = {"high": 0, "medium": 1, "low": 2}
    checklist.sort(key=lambda x: severity_order.get(x["severity"], 3))

    return checklist

# =========================
# RISK ASSESSMENT 
# =========================

def compute_risk_level(damage_output, vehicle_info):
    score = damage_output["damage_score"] # base score from damage detection

    if vehicle_info["kms_driven"] > 80000:  score += 2
    if vehicle_info["age"] > 8:             score += 2
    if "Third" in vehicle_info["owner"]:    score += 2
    if sentiment_score < -0.5:              score += 2
    if contradictions:                      score += 3

    if score <= 4:    risk_level = "Low"
    elif score <= 8:  risk_level = "Medium"
    else:             risk_level = "High"

    print(f"Risk Score: {score} → Risk Level: {risk_level}")
    return {"risk_score": score, "risk_level": risk_level}

# =========================
# EXPLANATION GENERATION
# =========================

def generate_explanation(vehicle_info, yolo_output, risk_result, contradictions):
    reasons = []
    dmg = yolo_output["damage_count"]
    level = risk_result["risk_level"]

    # -------------------------
    # 🔥 NEW: Contradiction Logic
    # -------------------------
    if contradictions:
        reasons.append(
            "Detected inconsistencies between seller description and visual inspection."
        )

        reasons.append(
            "This mismatch increases uncertainty and potential hidden risk in the vehicle."
        )

        for c in contradictions:
            reasons.append(f"⚠️ {c}")

    # Core Reason (Level Based)
    if level == "Low":
        reasons.append(
            "Overall risk is low due to minimal damage and stable vehicle condition."
        )
    elif level == "Medium":
        reasons.append(
            "Risk is moderate due to a combination of usage factors and visible wear."
        )
    else:
        reasons.append(
            "Risk is high due to significant damage and/or heavy vehicle usage."
        )

    # Tabular reasons
    if vehicle_info["kms_driven"] > 60000:
        reasons.append(
            f"High odometer reading ({vehicle_info['kms_driven']} km) "
            f"significantly reduces resale value."
        )
    if vehicle_info["age"] > 7:
        reasons.append(
            f"Vehicle is {vehicle_info['age']} years old, "
            f"increasing mechanical wear risk."
        )
    if any(x in vehicle_info["owner"] for x in ["Second", "Third", "Fourth"]):
        reasons.append(
            f"Multiple ownership ({vehicle_info['owner']}) "
            f"lowers market desirability."
        )

    # Damage reasons
    if dmg.get("rust", 0) > 0:
        reasons.append(
            f"Rust detected — indicates potential structural "
            f"or long-term corrosion issues."
        )
    if dmg.get("dent", 0) > 0:
        reasons.append(
            f"{dmg['dent']} dent(s) detected — "
            f"body repair costs should be factored in."
        )
    if dmg.get("scratch", 0) > 0:
        reasons.append(
            f"{dmg['scratch']} scratch(es) detected — "
            f"cosmetic damage affects perceived value."
        )

    return reasons

# =========================
# CLAIM ANALYSIS (SIMULATED)
# =========================

def extract_claims(text):
    text = text.lower()

    claims = {
        "no_accident": "no accident" in text,
        "well_maintained": "well maintained" in text,
        "no_damage": "no damage" in text or "scratchless" in text
    }

    return claims

def detect_contradictions(claims, damage_output):
    contradictions = []

    if claims["no_damage"]:
        if damage_output["damage_count"]["dent"] > 0:
            contradictions.append("Dent detected but seller claims no damage")

        if damage_output["damage_count"]["scratch"] > 0:
            contradictions.append("Scratch detected but seller claims no damage")

    if claims["no_accident"] and damage_output["damage_count"]["dent"] > 2:
        contradictions.append("Possible accident damage despite 'no accident' claim")

    return contradictions

# =========================
# STREAMLIT UI
# =========================

st.markdown("""
<h1 style='text-align:center;'> 🚗 Smarter Used Vehicle Pricing System </h1>
""", unsafe_allow_html=True)

# Input form in sidebar 
with st.sidebar:
    st.title("⚙️ Input Details")

    vehicle_type = st.selectbox("Vehicle Type", ["Car", "Bike"])

    if vehicle_type == "Car":
        name = st.text_input("Car Name", "Hyundai i20")
        fuel = st.selectbox("Fuel", ["Petrol", "Diesel", "CNG"])
        transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
    else:
        name = st.text_input("Bike Name", "Yamaha FZ-S")

    kms = st.number_input("KMs Driven", 0, 200000, 20000)
    year = st.number_input("Registration Year", 2005, 2024, 2015)
    owner = st.selectbox("Owner", ["First", "Second", "Third", "Fourth"])

    description = st.text_area("Seller Description")

    images = st.file_uploader("Upload Images", accept_multiple_files=True)

    predict_btn = st.button("🚀 Predict")

if predict_btn:

    # Preprocess
    input_data = {
            "name": name,
            "kms": kms,
            "year": year,
            "owner": owner
        }

    if vehicle_type == "Car":
            input_data["fuel"] = fuel
            input_data["transmission"] = transmission

    with st.spinner("Analyzing vehicle..."):

        # Price prediction
        shap_df, price, lower, upper = predict(vehicle_type, input_data)

    # Damage detection
    if not images:
        st.warning("Please upload at least one image for damage analysis.")

    if images:

        total_dent = 0
        total_scratch = 0
        total_rust = 0

        for img_file in images:
            image = Image.open(img_file)

            _, dent, scratch, rust = run_detection(image, vehicle_type)

            total_dent += dent
            total_scratch += scratch
            total_rust += rust

    damage_score = (
        total_dent * 3 +
        total_scratch * 1 +
        total_rust * 4
    )

    adjusted_price = price

    if damage_score == 0:
        reduction_percent = 0

    elif damage_score <= 3:
        reduction_percent = 0.05   # 5%

    elif damage_score <= 7:
        reduction_percent = 0.10   # 10%

    elif damage_score <= 12:
        reduction_percent = 0.20   # 20%

    else:
        reduction_percent = 0.30   # 30%

    adjusted_price = round(price * (1 - reduction_percent), -2)
    lower = round(adjusted_price * 0.9, -2)
    upper = round(adjusted_price * 1.1, -2)

    damage_output = {
        "damage_count": {
            "dent": total_dent,
            "scratch": total_scratch,
            "rust": total_rust
        },
        "damage_score": damage_score
    }

    # Fuse data

    if vehicle_type == "Bike":
        age = 2020 - year
    else:  # Car
        age = 2024 - year

    vehicle_info = {
        "kms_driven": kms,
        "age": age,
        "owner": owner,
    }

    yolo_output = {
        "damage_count": {
            "dent": total_dent,
            "scratch": total_scratch,
            "rust": total_rust
        }
    }

    # Generate condition flags
    flags = map_to_condition_flags(vehicle_info, yolo_output)

    # Generate checklist
    checklist = generate_inspection_checklist(vehicle_type.lower(), flags, rules_df)

    sentiment_score = get_sentiment_score(description)
    claims = extract_claims(description)
    contradictions = detect_contradictions(claims, damage_output)

    # Compute risk level
    risk_result = compute_risk_level(damage_output, vehicle_info)

    # Generate explanations
    explanations = generate_explanation(vehicle_info, yolo_output, risk_result, contradictions)

    # =========================
    # OUTPUT
    # =========================

    st.markdown(f"""
    <div class="card">
        <h3>💰 Estimated Price</h3>
        <h2>₹{adjusted_price}</h2>
        <p>Range: ₹{lower} - ₹{upper}</p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("📝 Seller Description Analysis")
    if contradictions:

        items_html = "".join(
            [f"<p style='margin:5px 0;'>• {c}</p>" for c in contradictions]
        )

        st.markdown(f"""
        <div style="background:#7f1d1d;
                    padding:15px;
                    margin-top:5px;
                    border-radius:10px;
                    border-left:5px solid #ef4444;
                    color:white;">

        <b>⚠️ Seller Claims Mismatch Detected</b><br><br>

        {''.join([f"• {c}<br>" for c in contradictions])}

        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="background:#14532d;
                    padding:15px;
                    margin-top:5px;
                    border-radius:10px;
                    border-left:6px solid #22c55e;
                    color:white;">

        <b>✔ Seller claims are consistent with vehicle condition</b>

        </div>
        """, unsafe_allow_html=True)

    def risk_color(level):
        if level == "High":
            return "#ef4444", "🔴 HIGH RISK"
        elif level == "Medium":
            return "#facc15", "🟡 MEDIUM RISK"
        else:
            return "#22c55e", "🟢 LOW RISK"
            
    color, label = risk_color(risk_result["risk_level"])

    st.subheader("⚠️ Risk Assessment")
    st.markdown(f"""
    <div style="background:#1e293b;padding:20px;border-radius:15px;border-left:6px solid {color};">
    <h2 style="color:{color};">{label}</h2>
    <p>Risk Score: <b>{risk_result['risk_score']}</b></p>
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("🔍 Inspection Recommendations")
    if checklist:
        unique_recommendations = list(dict.fromkeys(
            item['recommendation'] for item in checklist
        ))

        items_html = "".join(
            [f"• {rec}<br>" for rec in unique_recommendations]
        )

        st.markdown(f"""
<div style="background:#1e293b;
            padding:18px;
            border-radius:12px;
            margin-top:20px;
            border-left:6px solid #3b82f6;
            color:white;">

{items_html}

</div>
""", unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="background:#14532d;
                    padding:15px;
                    margin-top:5px;
                    border-radius:10px;
                    border-left:6px solid #22c55e;
                    color:white;">

        <b>✔ No recommendations needed</b>
                    
        </div>
        """, unsafe_allow_html=True)

    st.subheader("📊 Price Explanation (SHAP)")

    top_features = shap_df.head(3)

    items_html = "".join([
        f"• {row['feature'].replace('_',' ').title()} → "
        f"{'⬆️ increases price' if row['impact'] > 0 else '⬇️ decreases price'}<br>"
        for _, row in top_features.iterrows()
    ])

    st.markdown(f"""
<div style="background:#1e293b;
            padding:18px;
            border-radius:12px;
            margin-top:5px;
            border-left:6px solid #8b5cf6;
            color:white;">

<b>🧠 Key Factors Affecting Price</b><br><br>

{items_html}

</div>
""", unsafe_allow_html=True)

    st.subheader("🔍 Why this result?")

    items_html = "".join([
    f"• {r}<br>" for r in explanations
    ])
    st.markdown(f"""
<div style="background:#1e293b;
            padding:18px;
            border-radius:12px;
            margin-top:5px;
            border-left:6px solid #f97316;
            color:white;">

<b>📌 Explanation Summary</b><br><br>

{items_html}

</div>
""", unsafe_allow_html=True)

