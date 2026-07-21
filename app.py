import streamlit as st
import pandas as pd
import joblib
import numpy as np
model = joblib.load("models/best_logistic_model.pkl")
scaler = joblib.load("models/scaler.pkl")
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📱",
    layout="wide"
)

st.title("📊 Customer Churn Prediction System")

st.caption(
    "Machine Learning powered dashboard for predicting telecom customer churn."
)
st.info(
    "Model: Logistic Regression | Test Accuracy: 80.45%"
)
st.sidebar.header("Customer Information")
tenure = st.sidebar.slider(
    "Tenure (Months)",
    0,
    72,
    12
)

monthly = st.sidebar.slider(
    "Monthly Charges",
    18.0,
    120.0,
    70.0
)

total = st.sidebar.number_input(
    "Total Charges",
    value=1000.0
)
gender = st.sidebar.selectbox(
    "Gender",
    ["Female","Male"]
)

partner = st.sidebar.selectbox(
    "Partner",
    ["No","Yes"]
)

dependents = st.sidebar.selectbox(
    "Dependents",
    ["No","Yes"]
)
phone_service = st.sidebar.selectbox(
    "Phone Service",
    ["No", "Yes"]
)

multiple_lines = st.sidebar.selectbox(
    "Multiple Lines",
    ["No", "Yes", "No phone service"]
)

internet_service = st.sidebar.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

online_security = st.sidebar.selectbox(
    "Online Security",
    ["No", "Yes", "No internet service"]
)

online_backup = st.sidebar.selectbox(
    "Online Backup",
    ["No", "Yes", "No internet service"]
)

device_protection = st.sidebar.selectbox(
    "Device Protection",
    ["No", "Yes", "No internet service"]
)

tech_support = st.sidebar.selectbox(
    "Tech Support",
    ["No", "Yes", "No internet service"]
)

streaming_tv = st.sidebar.selectbox(
    "Streaming TV",
    ["No", "Yes", "No internet service"]
)

streaming_movies = st.sidebar.selectbox(
    "Streaming Movies",
    ["No", "Yes", "No internet service"]
)

contract = st.sidebar.selectbox(
    "Contract",
    ["Month-to-month", "One year", "Two year"]
)

paperless_billing = st.sidebar.selectbox(
    "Paperless Billing",
    ["No", "Yes"]
)

payment_method = st.sidebar.selectbox(
    "Payment Method",
    [
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check"
    ]
)

senior_citizen = st.sidebar.selectbox(
    "Senior Citizen",
    [0, 1]
)
# Create customer input dictionary
customer = {
    "SeniorCitizen": senior_citizen,
    "tenure": tenure,
    "MonthlyCharges": monthly,
    "TotalCharges": total,

    "gender_Male": 1 if gender == "Male" else 0,

    "Partner_Yes": 1 if partner == "Yes" else 0,

    "Dependents_Yes": 1 if dependents == "Yes" else 0,

    "PhoneService_Yes": 1 if phone_service == "Yes" else 0,

    "MultipleLines_No phone service": 1 if multiple_lines == "No phone service" else 0,
    "MultipleLines_Yes": 1 if multiple_lines == "Yes" else 0,

    "InternetService_Fiber optic": 1 if internet_service == "Fiber optic" else 0,
    "InternetService_No": 1 if internet_service == "No" else 0,

    "OnlineSecurity_No internet service": 1 if online_security == "No internet service" else 0,
    "OnlineSecurity_Yes": 1 if online_security == "Yes" else 0,

    "OnlineBackup_No internet service": 1 if online_backup == "No internet service" else 0,
    "OnlineBackup_Yes": 1 if online_backup == "Yes" else 0,

    "DeviceProtection_No internet service": 1 if device_protection == "No internet service" else 0,
    "DeviceProtection_Yes": 1 if device_protection == "Yes" else 0,

    "TechSupport_No internet service": 1 if tech_support == "No internet service" else 0,
    "TechSupport_Yes": 1 if tech_support == "Yes" else 0,

    "StreamingTV_No internet service": 1 if streaming_tv == "No internet service" else 0,
    "StreamingTV_Yes": 1 if streaming_tv == "Yes" else 0,

    "StreamingMovies_No internet service": 1 if streaming_movies == "No internet service" else 0,
    "StreamingMovies_Yes": 1 if streaming_movies == "Yes" else 0,

    "Contract_One year": 1 if contract == "One year" else 0,
    "Contract_Two year": 1 if contract == "Two year" else 0,

    "PaperlessBilling_Yes": 1 if paperless_billing == "Yes" else 0,

    "PaymentMethod_Credit card (automatic)": 1 if payment_method == "Credit card (automatic)" else 0,
    "PaymentMethod_Electronic check": 1 if payment_method == "Electronic check" else 0,
    "PaymentMethod_Mailed check": 1 if payment_method == "Mailed check" else 0,
}
feature_names = model.feature_names_in_

input_df = pd.DataFrame([customer])

# Ensure column order matches training
input_df = input_df.reindex(columns=feature_names, fill_value=0)

# Scale numerical columns
numerical_columns = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

input_df[numerical_columns] = scaler.transform(
    input_df[numerical_columns]
)
if st.button("Predict Churn"):

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    # Everything below should be INSIDE this block

    st.header("📊 Prediction Dashboard")

    left, right = st.columns([2, 1])

    with left:

        if probability < 0.30:
            st.success("🟢 LOW RISK")
        elif probability < 0.70:
            st.warning("🟡 MEDIUM RISK")
        else:
            st.error("🔴 HIGH RISK")

        if prediction == 1:
            st.error("⚠️ Customer is likely to churn.")
        else:
            st.success("✅ Customer is likely to stay.")

        st.progress(float(probability))

    with right:
        st.metric(
            "Churn Probability",
            f"{probability:.2%}"
        )

    st.divider()

    st.subheader("📋 Customer Summary")

    col1, col2 = st.columns(2)

    with col1:
        st.write(f"**Gender:** {gender}")
        st.write(f"**Tenure:** {tenure} Months")
        st.write(f"**Contract:** {contract}")
        st.write(f"**Internet Service:** {internet_service}")
        st.write(f"**Monthly Charges:** ₹{monthly}")

    with col2:
        st.write(f"**Partner:** {partner}")
        st.write(f"**Dependents:** {dependents}")
        st.write(f"**Tech Support:** {tech_support}")
        st.write(f"**Payment Method:** {payment_method}")
        st.write(f"**Paperless Billing:** {paperless_billing}")

    st.divider()

    st.subheader("💡 Recommendation")

    if probability < 0.30:
        st.success("Low churn risk. Regular engagement is sufficient.")
    elif probability < 0.70:
        st.warning("Medium churn risk. Consider retention offers.")
    else:
        st.error("High churn risk. Immediate retention action is recommended.")
    st.warning(
    "This prediction is based on historical telecom customer data and should support—not replace—business decisions."
)
    st.divider()

    st.caption(
        "Developed using Python • Scikit-learn • Streamlit • Logistic Regression"
    )
    st.divider()

    st.subheader("📈 Business Insight")
    if probability < 0.30:

        st.success("""
        This customer shows strong signs of loyalty.

        Reasons:
        • Long tenure
        • Stable contract
        • Good support services

        Business Action:
        Continue regular engagement.
        """)

    elif probability < 0.70:

        st.warning("""
        Customer has a moderate risk of churn.

        Business Action:
        • Offer loyalty rewards
        • Send personalized offers
        • Monitor customer engagement
        """)

    else:

        st.error("""
        Customer has a high probability of churn.

        Business Action:
        • Contact customer immediately
        • Offer discounts
        • Upgrade contract plans
        • Provide dedicated customer support
        """)