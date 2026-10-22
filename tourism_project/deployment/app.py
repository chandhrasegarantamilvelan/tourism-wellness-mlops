"""
Streamlit web app for the Wellness Tourism Package prediction.

It loads the model that the pipeline trained and committed into this folder,
collects customer details from the user, puts them into a one-row dataframe
with the same columns the model was trained on, and shows the prediction.
"""

import os
import joblib
import pandas as pd
import streamlit as st

# load the model from the same folder as this script (works regardless of cwd)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_tourism_model_v1.joblib")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()

st.set_page_config(page_title="Wellness Tourism Package Predictor", page_icon="🧳")
st.title("🧳 Wellness Tourism Package Predictor")
st.write(
    "Predict whether a customer is likely to purchase the new Wellness Tourism "
    "Package. Fill in the customer details below and click **Predict**."
)

# --- collect the inputs ---
st.subheader("Customer details")
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    type_of_contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
    city_tier = st.selectbox("City Tier", [1, 2, 3])
    occupation = st.selectbox(
        "Occupation", ["Salaried", "Free Lancer", "Small Business", "Large Business"]
    )
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    designation = st.selectbox(
        "Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"]
    )
    monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=100000, value=20000)
    passport = st.selectbox("Holds a Passport?", [0, 1], format_func=lambda x: "Yes" if x else "No")

with col2:
    product_pitched = st.selectbox(
        "Product Pitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"]
    )
    preferred_property_star = st.selectbox("Preferred Property Star", [3.0, 4.0, 5.0])
    duration_of_pitch = st.number_input("Duration of Pitch (minutes)", min_value=0, max_value=200, value=15)
    number_of_followups = st.number_input("Number of Follow-ups", min_value=0, max_value=10, value=3)
    pitch_satisfaction = st.slider("Pitch Satisfaction Score", 1, 5, 3)
    number_of_persons = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=3)
    number_of_trips = st.number_input("Number of Trips per year", min_value=0, max_value=30, value=2)
    number_of_children = st.number_input("Number of Children Visiting (under 5)", min_value=0, max_value=5, value=0)
    own_car = st.selectbox("Owns a Car?", [0, 1], format_func=lambda x: "Yes" if x else "No")

# --- assemble a single-row dataframe with the exact training columns ---
input_df = pd.DataFrame([{
    "Age": age,
    "TypeofContact": type_of_contact,
    "CityTier": city_tier,
    "DurationOfPitch": duration_of_pitch,
    "Occupation": occupation,
    "Gender": gender,
    "NumberOfPersonVisiting": number_of_persons,
    "NumberOfFollowups": number_of_followups,
    "ProductPitched": product_pitched,
    "PreferredPropertyStar": preferred_property_star,
    "MaritalStatus": marital_status,
    "NumberOfTrips": number_of_trips,
    "Passport": passport,
    "PitchSatisfactionScore": pitch_satisfaction,
    "OwnCar": own_car,
    "NumberOfChildrenVisiting": number_of_children,
    "Designation": designation,
    "MonthlyIncome": monthly_income,
}])

st.subheader("Model input")
st.dataframe(input_df)

if st.button("Predict"):
    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]
    if pred == 1:
        st.success(f"✅ Likely to PURCHASE the Wellness Package (probability {proba:.2%})")
    else:
        st.info(f"❌ Unlikely to purchase the Wellness Package (probability {proba:.2%})")
