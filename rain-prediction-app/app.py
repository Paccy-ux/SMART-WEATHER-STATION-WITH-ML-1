# ================================================================
#  STREAMLIT APP — Rain Tomorrow Prediction
#  File: app.py
#  Author: IYAMUREMYE Pacifique — Capstone Project 2026
#
#  HOW TO RUN:
#    1. Copy this file (app.py) into the same folder that contains
#       your  saved_models/  folder from the Jupyter notebook.
#    2. Open terminal / Spyder terminal and run:
#           streamlit run app.py
#    3. The app opens automatically in your browser.
#
#  INSTALL Streamlit first (run once in terminal):
#       pip install streamlit
# ================================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os

# ── Page configuration ────────────────────────────────────────────
st.set_page_config(
    page_title="Rain Prediction System",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #0A2240, #0D7A6F);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
    }
    .main-header h1 { color: white; margin: 0; font-size: 2rem; }
    .main-header p  { color: #A8D4CF; margin: 0.3rem 0 0; font-size: 1rem; }

    .result-rain {
        background: linear-gradient(135deg, #1565A7, #0A2240);
        padding: 1.5rem; border-radius: 12px; color: white;
        text-align: center; margin: 1rem 0;
    }
    .result-norain {
        background: linear-gradient(135deg, #15803D, #0A2240);
        padding: 1.5rem; border-radius: 12px; color: white;
        text-align: center; margin: 1rem 0;
    }
    .metric-card {
        background: #F4F8F7; padding: 1rem; border-radius: 8px;
        border-left: 4px solid #0D7A6F; margin: 0.4rem 0;
    }
    .advice-box {
        background: #FEF3C7; border: 1px solid #D97706;
        padding: 1rem; border-radius: 8px; margin: 0.5rem 0;
        color: #7A4B00;
    }
    .good-box {
        background: #D1FAE5; border: 1px solid #15803D;
        padding: 1rem; border-radius: 8px; margin: 0.5rem 0;
        color: #15803D;
    }
    div[data-testid="stSidebar"] { background: #F4F8F7; }
    .stButton>button {
        background: #0D7A6F; color: white; border: none;
        border-radius: 8px; padding: 0.6rem 1.5rem; font-weight: bold;
        font-size: 1rem; width: 100%;
    }
    .stButton>button:hover { background: #0A5C54; }
</style>
""", unsafe_allow_html=True)


# ── Load saved model artefacts ────────────────────────────────────
@st.cache_resource
def load_artefacts():
    base = 'saved_models'
    with open(f'{base}/meta.json') as f:
        meta = json.load(f)

    models = {}
    for name in meta['model_names']:
        fname = name.lower().replace(' ', '_') + '.pkl'
        path  = f'{base}/{fname}'
        if os.path.exists(path):
            models[name] = joblib.load(path)

    artefacts = {
        'models':         models,
        'scaler':         joblib.load(f'{base}/scaler.pkl'),
        'num_imputer':    joblib.load(f'{base}/num_imputer.pkl'),
        'cat_imputer':    joblib.load(f'{base}/cat_imputer.pkl'),
        'label_encoders': joblib.load(f'{base}/label_encoders.pkl'),
        'meta':           meta,
    }
    return artefacts

try:
    artefacts = load_artefacts()
    models_ok = True
except Exception as e:
    models_ok = False
    load_error = str(e)


# ── Header ────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🌧️ Rain Tomorrow Prediction System</h1>
    <p>Smart Weather Station | Capstone Project 2026 | IYAMUREMYE Pacifique</p>
</div>
""", unsafe_allow_html=True)

if not models_ok:
    st.error(f"❌ Could not load models from saved_models/ folder.\n\nError: {load_error}")
    st.info("Make sure the saved_models/ folder is in the same directory as app.py")
    st.stop()


# ── Wind direction options ─────────────────────────────────────────
WIND_DIRS = ['N','NNE','NE','ENE','E','ESE','SE','SSE',
             'S','SSW','SW','WSW','W','WNW','NW','NNW']


# ── Sidebar — model selector ──────────────────────────────────────
st.sidebar.markdown("## ⚙️ Settings")
selected_model_name = st.sidebar.selectbox(
    "Select ML Model",
    options=list(artefacts['models'].keys()),
    index=list(artefacts['models'].keys()).index(artefacts['meta']['best_model'])
          if artefacts['meta']['best_model'] in artefacts['models'] else 0,
    help="Best model is pre-selected based on highest ROC-AUC score from training."
)
selected_model = artefacts['models'][selected_model_name]

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Model Performance")
try:
    results_df = pd.read_csv('saved_models/results_summary.csv', index_col=0)
    if selected_model_name in results_df.index:
        row = results_df.loc[selected_model_name]
        st.sidebar.markdown(f"""
        <div class="metric-card">
        <b>Accuracy</b>: {row.get('Accuracy', 'N/A')}<br>
        <b>ROC-AUC</b>:  {row.get('ROC-AUC', 'N/A')}<br>
        <b>F1-Score</b>: {row.get('F1-Score', 'N/A')}
        </div>
        """, unsafe_allow_html=True)
except:
    pass

st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ About")
st.sidebar.info(
    "This app predicts whether it will rain tomorrow based on "
    "today's weather measurements. Enter the values in the form "
    "and click **Predict**."
)


# ── Main form ─────────────────────────────────────────────────────
st.markdown("### 📋 Enter Today's Weather Measurements")
st.markdown("Fill in all available readings. Leave unknown values at their default.")

with st.form("prediction_form"):

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**🌡️ Temperature**")
        MinTemp      = st.number_input("Min Temperature (°C)",     value=12.0,  step=0.1)
        MaxTemp      = st.number_input("Max Temperature (°C)",     value=25.0,  step=0.1)
        Temp9am      = st.number_input("Temperature at 9am (°C)",  value=17.0,  step=0.1)
        Temp3pm      = st.number_input("Temperature at 3pm (°C)",  value=23.0,  step=0.1)

        st.markdown("**🌧️ Rainfall**")
        Rainfall     = st.number_input("Rainfall (mm)",            value=0.0,   step=0.1, min_value=0.0)
        Evaporation  = st.number_input("Evaporation (mm)",         value=5.0,   step=0.1, min_value=0.0)
        Sunshine     = st.number_input("Sunshine (hours)",         value=7.0,   step=0.1, min_value=0.0, max_value=24.0)
        RainToday    = st.selectbox("Rain Today?",                 ['No', 'Yes'])

    with col2:
        st.markdown("**💧 Humidity**")
        Humidity9am  = st.slider("Humidity at 9am (%)",  0, 100, 65)
        Humidity3pm  = st.slider("Humidity at 3pm (%)",  0, 100, 40)

        st.markdown("**🌬️ Wind**")
        WindGustDir  = st.selectbox("Wind Gust Direction",  WIND_DIRS, index=WIND_DIRS.index('W'))
        WindGustSpeed= st.number_input("Wind Gust Speed (km/h)", value=35.0, step=1.0, min_value=0.0)
        WindDir9am   = st.selectbox("Wind Direction at 9am", WIND_DIRS, index=WIND_DIRS.index('N'))
        WindDir3pm   = st.selectbox("Wind Direction at 3pm", WIND_DIRS, index=WIND_DIRS.index('W'))
        WindSpeed9am = st.number_input("Wind Speed at 9am (km/h)", value=15.0, step=1.0, min_value=0.0)
        WindSpeed3pm = st.number_input("Wind Speed at 3pm (km/h)", value=20.0, step=1.0, min_value=0.0)

    with col3:
        st.markdown("**🔵 Pressure**")
        Pressure9am  = st.number_input("Pressure at 9am (hPa)", value=1015.0, step=0.1, min_value=900.0, max_value=1100.0)
        Pressure3pm  = st.number_input("Pressure at 3pm (hPa)", value=1010.0, step=0.1, min_value=900.0, max_value=1100.0)

        st.markdown("**☁️ Cloud Cover**")
        Cloud9am     = st.slider("Cloud Cover at 9am (oktas)", 0, 9, 3)
        Cloud3pm     = st.slider("Cloud Cover at 3pm (oktas)", 0, 9, 4)

    submitted = st.form_submit_button("🔮  Predict Rain Tomorrow", use_container_width=True)


# ── Prediction ─────────────────────────────────────────────────────
if submitted:
    st.markdown("---")

    # Build raw input DataFrame — must match training columns exactly
    input_dict = {
        'MinTemp':       MinTemp,
        'MaxTemp':       MaxTemp,
        'Rainfall':      Rainfall,
        'Evaporation':   Evaporation,
        'Sunshine':      Sunshine,
        'WindGustDir':   WindGustDir,
        'WindGustSpeed': WindGustSpeed,
        'WindDir9am':    WindDir9am,
        'WindDir3pm':    WindDir3pm,
        'WindSpeed9am':  WindSpeed9am,
        'WindSpeed3pm':  WindSpeed3pm,
        'Humidity9am':   Humidity9am,
        'Humidity3pm':   Humidity3pm,
        'Pressure9am':   Pressure9am,
        'Pressure3pm':   Pressure3pm,
        'Cloud9am':      Cloud9am,
        'Cloud3pm':      Cloud3pm,
        'Temp9am':       Temp9am,
        'Temp3pm':       Temp3pm,
        'RainToday':     RainToday,
    }
    input_df = pd.DataFrame([input_dict])

    meta        = artefacts['meta']
    num_cols    = meta['num_cols']
    cat_cols    = meta['cat_cols']
    all_features= meta['all_features']

    # Add any missing columns with NaN (so imputer handles them)
    for col in all_features:
        if col not in input_df.columns:
            input_df[col] = np.nan

    # Reorder to match training
    input_df = input_df[all_features]

    # Impute
    input_df[num_cols] = artefacts['num_imputer'].transform(input_df[num_cols])
    input_df[cat_cols] = artefacts['cat_imputer'].transform(input_df[cat_cols])

    # Encode categoricals
    le_dict = artefacts['label_encoders']
    for col in cat_cols:
        le = le_dict[col]
        val = input_df[col].iloc[0]
        if val in le.classes_:
            input_df[col] = le.transform(input_df[col])
        else:
            input_df[col] = le.transform([le.classes_[0]])  # fallback to first class

    # Scale
    input_sc = artefacts['scaler'].transform(input_df)

    # Predict
    prediction = selected_model.predict(input_sc)[0]
    probability = selected_model.predict_proba(input_sc)[0]
    prob_rain   = probability[1]
    prob_norain = probability[0]

    # ── Display result ─────────────────────────────────────────────
    res_col1, res_col2 = st.columns([1, 1])

    with res_col1:
        if prediction == 1:
            st.markdown(f"""
            <div class="result-rain">
                <h2>🌧️ YES — Rain Predicted Tomorrow</h2>
                <h3>Probability: {prob_rain:.1%}</h3>
                <p>Model: {selected_model_name}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-norain">
                <h2>☀️ NO — No Rain Predicted Tomorrow</h2>
                <h3>Probability: {prob_norain:.1%}</h3>
                <p>Model: {selected_model_name}</p>
            </div>
            """, unsafe_allow_html=True)

    with res_col2:
        st.markdown("**Prediction Confidence**")
        st.metric("🌧️ Rain Probability",    f"{prob_rain:.1%}")
        st.metric("☀️ No Rain Probability", f"{prob_norain:.1%}")
        st.progress(float(prob_rain))

    # ── Agricultural Advice ────────────────────────────────────────
    st.markdown("### 🌱 Agricultural Decision-Support Advice")
    adv_col1, adv_col2 = st.columns(2)

    advice_items = []

    if prediction == 1 and prob_rain > 0.65:
        advice_items.append(("⚠️", "Heavy rain expected — avoid fertilizer application to prevent nutrient leaching.", "warn"))
        advice_items.append(("⚠️", "Delay pesticide/herbicide spraying — rain will wash chemicals away.", "warn"))
        advice_items.append(("✅", "No need for irrigation — natural rainfall will hydrate crops.", "good"))
    elif prediction == 1 and prob_rain <= 0.65:
        advice_items.append(("ℹ️", "Moderate rain likely — monitor conditions before fertilizer application.", "warn"))
        advice_items.append(("✅", "Light irrigation may not be needed — wait for rainfall first.", "good"))
    else:
        advice_items.append(("✅", "Dry conditions expected — schedule irrigation today or tomorrow.", "good"))
        advice_items.append(("✅", "Good conditions for fertilizer application — low runoff risk.", "good"))
        advice_items.append(("✅", "Suitable for pesticide spraying — check wind speed before application.", "good"))

    if Humidity3pm > 80:
        advice_items.append(("⚠️", f"High humidity ({Humidity3pm}%) — monitor crops for fungal disease outbreaks.", "warn"))
    if MaxTemp > 32:
        advice_items.append(("⚠️", f"High temperature ({MaxTemp}°C) — increase irrigation frequency to reduce heat stress.", "warn"))
    if WindGustSpeed > 40:
        advice_items.append(("⚠️", f"Strong wind gusts ({WindGustSpeed} km/h) — suspend spraying operations.", "warn"))

    half = len(advice_items) // 2 + len(advice_items) % 2

    for i, (icon, text, style) in enumerate(advice_items):
        col = adv_col1 if i < half else adv_col2
        css_class = "advice-box" if style == "warn" else "good-box"
        col.markdown(f'<div class="{css_class}">{icon} {text}</div>', unsafe_allow_html=True)

    # ── Input summary ──────────────────────────────────────────────
    with st.expander("📋 View Input Summary"):
        display_df = pd.DataFrame([input_dict]).T
        display_df.columns = ['Value']
        st.dataframe(display_df, use_container_width=True)


# ── Footer ────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#64748B; font-size:0.85rem;'>"
    "Smart Weather Station &nbsp;|&nbsp; Capstone Project 2026 &nbsp;|&nbsp; "
    "IYAMUREMYE Pacifique (25RP20567) &nbsp;|&nbsp; Supervisor: Dr. MUGEMANYI Sylvere"
    "</div>",
    unsafe_allow_html=True
)
