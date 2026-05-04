import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CreditWise – Loan Approval System",
    page_icon="🏦",
    layout="centered",
)

# ── Load artifacts ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model           = joblib.load("model.pkl")
    scaler          = joblib.load("scaler.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    label_encoder   = joblib.load("label_encoder.pkl")
    return model, scaler, feature_columns, label_encoder

try:
    model, scaler, feature_columns, label_encoder = load_artifacts()
except FileNotFoundError as e:
    st.error(
        f"❌ Model file not found: {e}\n\n"
        "Make sure `model.pkl`, `scaler.pkl`, `feature_columns.pkl`, and "
        "`label_encoder.pkl` are in the same folder as `app.py`."
    )
    st.stop()

# ── UI ─────────────────────────────────────────────────────────────────────────
st.title("🏦 CreditWise – Loan Approval Predictor")
st.markdown("Fill in the applicant details below and click **Predict** to check loan eligibility.")
st.divider()

# ── Input form ─────────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    applicant_income = st.number_input("Applicant Income (₹)", min_value=0, value=50000, step=1000)
    credit_score     = st.number_input("Credit Score", min_value=300, max_value=900, value=700)
    loan_amount      = st.number_input("Loan Amount (₹)", min_value=0, value=200000, step=5000)
    loan_term        = st.number_input("Loan Term (months)", min_value=1, max_value=360, value=60)
    dti_ratio        = st.number_input("DTI Ratio (%)", min_value=0.0, max_value=100.0, value=30.0, step=0.5)

with col2:
    savings          = st.number_input("Savings (₹)", min_value=0, value=100000, step=5000)
    gender           = st.selectbox("Gender", ["Male", "Female"])
    education_level  = st.selectbox("Education Level", ["Graduate", "Not Graduate"])
    employment_status= st.selectbox("Employment Status", ["Employed", "Self-Employed", "Unemployed"])
    marital_status   = st.selectbox("Marital Status", ["Married", "Single", "Divorced"])

col3, col4 = st.columns(2)
with col3:
    loan_purpose     = st.selectbox("Loan Purpose", ["Home", "Education", "Business", "Personal", "Other"])
    property_area    = st.selectbox("Property Area", ["Urban", "Rural", "Semiurban"])
with col4:
    employer_category= st.selectbox("Employer Category", ["Government", "Private", "NGO", "Self"])
    num_dependents   = st.number_input("Number of Dependents", min_value=0, max_value=10, value=0)

st.divider()

# ── Prediction ─────────────────────────────────────────────────────────────────
def build_input_df(
    applicant_income, credit_score, loan_amount, loan_term,
    dti_ratio, savings, gender, education_level,
    employment_status, marital_status, loan_purpose,
    property_area, employer_category, num_dependents
):
    """
    Replicates the exact preprocessing pipeline from the notebook:
      1. Label-encode Education_Level (Graduate=0, Not Graduate=1 or vice-versa)
      2. OneHotEncode with drop='first' the 6 categorical columns
      3. Feature-engineer DTI_Ratio_sq and Credit_score_sq
      4. Drop original Credit_Score and DTI_Ratio
      5. Align to saved feature_columns order
    """
    # Raw row
    raw = {
        "Applicant_Income":   applicant_income,
        "Loan_Amount":        loan_amount,
        "Loan_Term":          loan_term,
        "Credit_Score":       credit_score,
        "DTI_Ratio":          dti_ratio,
        "Savings":            savings,
        "Num_Dependents":     num_dependents,
        "Education_Level":    education_level,
        "Employment_Status":  employment_status,
        "Marital_Status":     marital_status,
        "Loan_Purpose":       loan_purpose,
        "Property_Area":      property_area,
        "Gender":             gender,
        "Employer_Category":  employer_category,
    }
    df = pd.DataFrame([raw])

    # --- Label-encode Education_Level (same as notebook) ---
    edu_map = {"Graduate": 0, "Not Graduate": 1}
    df["Education_Level"] = df["Education_Level"].map(edu_map)

    # --- One-hot encode (drop first) to match notebook ---
    ohe_cols = ["Employment_Status", "Marital_Status", "Loan_Purpose",
                "Property_Area", "Gender", "Employer_Category"]

    df_ohe = pd.get_dummies(df, columns=ohe_cols, drop_first=True)

    # --- Feature engineering ---
    df_ohe["DTI_Ratio_sq"]     = df_ohe["DTI_Ratio"] ** 2
    df_ohe["Credit_score_sq"]  = df_ohe["Credit_Score"] ** 2

    # Drop originals (as done in notebook after feature engineering)
    df_ohe = df_ohe.drop(columns=["Credit_Score", "DTI_Ratio"], errors="ignore")

    # --- Align to training feature columns (fill missing OHE cols with 0) ---
    df_aligned = df_ohe.reindex(columns=feature_columns, fill_value=0)

    return df_aligned

if st.button("🔍 Predict Loan Approval", use_container_width=True, type="primary"):
    input_df = build_input_df(
        applicant_income, credit_score, loan_amount, loan_term,
        dti_ratio, savings, gender, education_level,
        employment_status, marital_status, loan_purpose,
        property_area, employer_category, num_dependents
    )

    input_scaled = scaler.transform(input_df)
    prediction   = model.predict(input_scaled)[0]
    probability  = model.predict_proba(input_scaled)[0]

    # Map numeric prediction back to label
    # label_encoder was fit on the numeric Loan_Approved column (0/1)
    # Classes: 0 → No, 1 → Yes
    approved = bool(prediction)
    prob_approved = probability[1] * 100

    st.divider()
    if approved:
        st.success(f"✅ **Loan Approved!** (Confidence: {prob_approved:.1f}%)")
        st.balloons()
    else:
        st.error(f"❌ **Loan Not Approved** (Approval probability: {prob_approved:.1f}%)")

    # Probability bar
    st.markdown("#### Prediction Confidence")
    st.progress(int(prob_approved))
    st.caption(f"Approval probability: **{prob_approved:.1f}%** | Rejection probability: **{100 - prob_approved:.1f}%**")

    # Input summary
    with st.expander("📋 View submitted details"):
        summary = {
            "Applicant Income": f"₹{applicant_income:,}",
            "Credit Score": credit_score,
            "Loan Amount": f"₹{loan_amount:,}",
            "Loan Term": f"{loan_term} months",
            "DTI Ratio": f"{dti_ratio}%",
            "Savings": f"₹{savings:,}",
            "Gender": gender,
            "Education": education_level,
            "Employment": employment_status,
            "Marital Status": marital_status,
            "Loan Purpose": loan_purpose,
            "Property Area": property_area,
            "Employer Category": employer_category,
            "Dependents": num_dependents,
        }
        st.table(pd.DataFrame(summary.items(), columns=["Field", "Value"]))

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.caption("CreditWise Loan Approval System · Powered by Logistic Regression · Built with Streamlit")
