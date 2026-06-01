import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Page Configuration (Professional Branding)
st.set_page_config(
    page_title="Pharmacogenomic Sensitivity Predictor",
    page_icon="🧬",
    layout="wide"
)

# 2. Load the trained assets securely
@st.cache_resource
def load_assets():
    model = joblib.load('gdsc_model.pkl')
    scaler = joblib.load('gdsc_scaler.pkl')
    pathway_encoder = joblib.load('pathway_encoder.pkl')
    return model, scaler, pathway_encoder

try:
    model, scaler, pathway_encoder = load_assets()
except Exception as e:
    st.error("Error loading model assets. Ensure .pkl files are in the same directory as app.py.")
    st.stop()

# 3. Main UI Header
st.title("🧬 Pharmacogenomic Sensitivity Predictor")
st.markdown("""
This clinical decision-support tool predicts cancer cell line sensitivity to specific pharmacological compounds based on the GDSC (Genomics of Drug Sensitivity in Cancer) dataset.
""")
st.divider()

# 4. Sidebar for Clinical Inputs
st.sidebar.header("🔬 Clinical Input Parameters")
st.sidebar.markdown("Enter the pharmacokinetic and biological data below:")

# Extract classes for the dropdowns
pathway_options = pathway_encoder.classes_

# User Inputs
selected_pathway = st.sidebar.selectbox("Biological Target Pathway", pathway_options)
# For a streamlined UI, we will simulate a drug ID input since there are hundreds of drugs
drug_id = st.sidebar.number_input("Drug Compound ID (Encoded)", min_value=0, max_value=500, value=10)

min_conc = st.sidebar.number_input("Minimum Concentration (µM)", value=0.001, format="%.4f")
max_conc = st.sidebar.number_input("Maximum Concentration (µM)", value=10.0, format="%.2f")
auc = st.sidebar.number_input("Area Under Curve (AUC)", value=0.85, format="%.3f")
rmse = st.sidebar.number_input("Root Mean Square Error (RMSE)", value=0.05, format="%.3f")
z_score = st.sidebar.number_input("Z-Score", value=-1.5, format="%.2f")

# 5. Prediction Logic
if st.sidebar.button("Predict Sensitivity", type="primary"):
    
    # Feature Engineering (Replicating Step 2 from your pipeline)
    efficacy_index = auc / (rmse + 0.0001)
    
    # Encode categorical pathway
    pathway_encoded = pathway_encoder.transform([selected_pathway])[0]
    
    # Prepare the feature array in the exact order the model expects
    # ['MIN_CONC', 'MAX_CONC', 'AUC', 'RMSE', 'Z_SCORE', 'efficacy_index', 'pathway_encoded', 'drug_encoded']
    input_data = np.array([[min_conc, max_conc, auc, rmse, z_score, efficacy_index, pathway_encoded, drug_id]])
    
    # Scale the inputs
    input_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_scaled)
    prediction_proba = model.predict_proba(input_scaled)[0]
    
    # 6. Display Results in Main Panel
    st.subheader("📊 Diagnostic Output")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(label="Calculated Efficacy Index", value=f"{efficacy_index:.2f}")
    
    with col2:
        if prediction[0] == 1:
            st.success("🟢 **Prediction: SENSITIVE**")
            st.write(f"Confidence: **{prediction_proba[1] * 100:.1f}%**")
            st.write("The genomic profile suggests this cell line will exhibit a positive therapeutic response to the compound.")
        else:
            st.error("🔴 **Prediction: RESISTANT**")
            st.write(f"Confidence: **{prediction_proba[0] * 100:.1f}%**")
            st.write("The genomic profile suggests this cell line will resist the compound's mechanism of action.")
            
    st.divider()
    st.caption("Data Source: GDSC2 Dataset. For academic demonstration purposes only.")