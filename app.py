import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load the trained model
model_path = "random_forest_model.pkl"  # Update the path as needed
try:
    model = joblib.load(model_path)
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    model = None

# App configuration
st.set_page_config(page_title="Heart Health Predictor", page_icon="❤️")
st.title("Heart Disease Prediction App")

# Sidebar navigation
#st.sidebar.markdown("### Navigate")
#navigation = st.sidebar.radio("Go to:", ["Home", "About", "Contact Us"])
# Sidebar Header
st.sidebar.markdown(
    "<h2 style='text-align: center; color: #ff4b4b;'>🔍 Navigate</h2>", 
    unsafe_allow_html=True
)

# Sidebar Navigation with Icons
navigation = st.sidebar.radio(
    "",
    options=["Home", "About", "Contact Us"],
    index=0
)
# Decorative Separator
st.sidebar.markdown("<hr style='border:1px solid #ff4b4b;'>", unsafe_allow_html=True)

# Sidebar Footer or Branding
st.sidebar.markdown(
    """
    <div style='text-align: center; color: gray; font-size: small;'>
        ❤️ <b>Heart Health Predictor</b><br>
        Made with <a href="https://streamlit.io/" target="_blank" style="text-decoration: none; color: #ff4b4b;">Streamlit</a>
    </div>
    """,
    unsafe_allow_html=True
)

# Home Page
if navigation == "Home":
    # Welcome message
    st.markdown("#### Your Heart Health Companion Welcomes You! Let's Get Started.")
    patient_name = st.text_input("What's your name?", value="Guest")

    if patient_name:
        st.subheader(f"👋 Hello, {patient_name}! Let’s take a closer look at your heart health.")

    # Form: Patient Data Input
    st.markdown("### Please fill in your health details below:")

    with st.form("patient_form"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", min_value=0, max_value=120, value=30)
            sex = st.selectbox("Sex", options=["Male", "Female"])
            chest_pain_type = st.selectbox(
                "Chest Pain Type",
                options=["ATA (Typical Angina)", "NAP (Non-Anginal Pain)", "TA (Asymptomatic)"],
            )
            resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=0, value=120)
            cholesterol = st.number_input("Cholesterol (mg/dL)", min_value=0, value=200)
            fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL?", options=["Yes", "No"])
        with col2:
            resting_ecg = st.selectbox("Resting ECG", options=["Normal", "ST-T Wave", "Left Ventricular Hypertrophy"])
            max_hr = st.number_input("Maximum Heart Rate Achieved", min_value=0, value=150)
            exercise_angina = st.selectbox("Exercise-Induced Angina", options=["Yes", "No"])
            oldpeak = st.number_input("ST Depression (Oldpeak)", min_value=0.0, step=0.1, value=1.0)
            st_slope = st.selectbox("ST Slope", options=["Up", "Flat", "Down"])

        submitted = st.form_submit_button("Submit Details 💾")

    # Process Input and Predict
    if submitted:
        # Map inputs to the relevant columns
        input_data = {
            "Oldpeak": oldpeak,
            "Sex_M": 1 if sex == "Male" else 0,
            "MaxHR": max_hr,
            "ChestPainType_ATA": 1 if chest_pain_type == "ATA (Typical Angina)" else 0,
            "ExerciseAngina_Y": 1 if exercise_angina == "Yes" else 0,
            "ST_Slope_Flat": 1 if st_slope == "Flat" else 0,
            "ST_Slope_Up": 1 if st_slope == "Up" else 0,
        }

        # Convert to a DataFrame
        input_df = pd.DataFrame([input_data])

        # Prediction
        if model:
            prediction = model.predict(input_df)[0]
            risk = "High" if prediction == 1 else "Low"
            st.success(
                f"Prediction Complete: 💡 {patient_name}'s heart disease risk is **{risk}**."
            )

            # Display additional insights
            if risk == "High":
                st.warning(
                    "⚠️ Your risk of heart disease is high. Please consult a healthcare professional immediately!"
                )
            else:
                st.info(
                    "✅ Your risk of heart disease is low. Maintain a healthy lifestyle to keep your heart strong."
                )
        else:
            st.error("Model is not loaded. Unable to make predictions.")

        # Display the submitted data for user confirmation
        st.markdown("### Here's what we received:")
        st.dataframe(input_df)

# About Page
elif navigation == "About":
    st.title("About Us")
    st.markdown(
        """
        This application predicts the likelihood of heart disease based on patient details.
        It leverages machine learning models trained on clinical data to provide risk insights.
        Our mission is to help individuals detect heart health risks early.
        """
    )

# Contact Us Page
elif navigation == "Contact Us":
    st.title("Contact Us")
    st.markdown(
        """
        We'd love to hear from you! Reach out to us for any inquiries or feedback:
        - 📧 Email: [info@hearthealth.com](mailto:info@hearthealth.com)
        - 📞 Phone: +123-456-7890
        """
    )

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center;">
        <h4>Developed with ❤️ by Your Team</h4>
        <p>Contact us at <a href="mailto:info@hearthealth.com">info@hearthealth.com</a></p>
    </div>
    """,
    unsafe_allow_html=True,
)
