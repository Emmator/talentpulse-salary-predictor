
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import requests

# ------------------------------------------------------------
# Page configuration
# ------------------------------------------------------------

st.set_page_config(
    page_title="TalentPulse Salary Predictor",
    page_icon="💼",
    layout="centered"
)

# ------------------------------------------------------------
# Load the trained pipeline
# ------------------------------------------------------------

@st.cache_resource
def load_model():
    model_path = (
        Path(__file__).resolve().parent
        / "deployment"
        / "salary_prediction_pipeline.joblib"
    )
    return joblib.load(model_path)


model = load_model()

# Currency used by each country
LOCAL_CURRENCIES = {
    "UK": ("GBP", "£"),
    "Canada": ("CAD", "C$"),
    "Australia": ("AUD", "A$"),
    "India": ("INR", "₹")
}


@st.cache_data(ttl=3600)
def get_exchange_rate(currency_code):
    url = "https://api.frankfurter.dev/v1/latest"

    response = requests.get(
        url,
        params={
            "base": "USD",
            "symbols": currency_code
        },
        timeout=10
    )

    response.raise_for_status()
    data = response.json()

    return float(data["rates"][currency_code]), data["date"]
# ------------------------------------------------------------
# Application heading
# ------------------------------------------------------------

st.title("💼 TalentPulse Salary Predictor")

st.write(
    "Estimate annual job salary using a machine learning "
    "model trained on job listings from five countries."
)

st.info(
    "Portfolio demonstration only. Predictions are estimates, "
    "not guaranteed salaries or compensation recommendations."
)

# ------------------------------------------------------------
# Job information
# ------------------------------------------------------------

st.subheader("Job Information")

country = st.selectbox(
    "Country",
    ["USA", "UK", "Canada", "Australia", "India"]
)

job_family = st.selectbox(
    "Job Family",
    [
        "Data Scientist",
        "Data Analyst",
        "Data Engineer",
        "Machine Learning Engineer",
        "Business Analyst",
        "Software Engineer",
        "Product Manager",
        "Other"
    ]
)

job_level = st.selectbox(
    "Job Level",
    ["Junior", "Mid", "Senior", "Lead"],
    index=1
)

# ------------------------------------------------------------
# Location selection
# ------------------------------------------------------------

location_options = {
    "UK": [
        "London, UK",
        "The City, Central London",
        "Manchester, Greater Manchester",
        "Newcastle Upon Tyne, Tyne & Wear",
        "Belfast, Northern Ireland"
    ],
    "USA": [
        "Pico Heights, Los Angeles County",
        "Atlanta, Fulton County",
        "Pimmit, Fairfax County",
        "Grand Central, Manhattan",
        "Washington, D.C., US",
        "Boston, Suffolk County"
    ],
    "Canada": [
        "Toronto, Ontario",
        "Canada"
    ],
    "Australia": [
        "Sydney, Sydney Region"
    ],
    "India": [
    "Bangalore, Karnataka",
    "Mumbai, Maharashtra",
    "Pune, Maharashtra",
    "Hyderabad, Telangana",
    "Delhi, India"
]
}

available_locations = location_options.get(country, [])

location = st.selectbox(
    "Location",
    options=available_locations + ["Other location"],
    help="Select a location or choose Other location to enter one manually."
)

if location == "Other location":
    location = st.text_input(
        "Enter your location",
        placeholder="Enter city or location"
    )

is_remote = st.checkbox("Remote position")

degree_required = st.selectbox(
    "Degree Requirement",
    ["Not specified", "Bachelor", "Master", "PhD"]
)

experience_tier = st.selectbox(
    "Experience Tier",
    [
        "Unknown",
        "Entry",
        "Intermediate",
        "Experienced",
        "Expert"
    ]
)
experience_missing = st.checkbox(
    "Experience requirement not specified",
    value=True,
    help=(
        "Check this when the original job listing does not "
        "specify an experience requirement."
    )
)

# ------------------------------------------------------------
# Skills
# ------------------------------------------------------------

st.subheader("Technical Skills")

skill_python = st.checkbox("Python")
skill_sql = st.checkbox("SQL")
skill_spark = st.checkbox("Spark")
skill_aws = st.checkbox("AWS")

num_skills = st.number_input(
    "Total number of skills listed in the job description",
    min_value=0,
    max_value=30,
    value=0,
    step=1,
    help=(
        "Enter the total skills count as defined in the "
        "original dataset. It may include skills other than "
        "the four listed above."
    )
)

# Explain what the total skills count means
st.caption(
    "Enter the total number of skills mentioned in the job posting, "
    "including skills other than Python, SQL, Spark and AWS."
)

# ------------------------------------------------------------
# Prediction
# ------------------------------------------------------------

if st.button("Predict Salary", type="primary"):

    if not location.strip():
        st.error("Please enter a location.")

    elif num_skills < sum([
        skill_python,
        skill_sql,
        skill_spark,
        skill_aws
    ]):
        st.error(
            "Total Number of Skills cannot be smaller than "
            "the number of selected technical skills."
        )

    else:
        input_data = pd.DataFrame([{
            "country": country,
            "job_family": job_family,
            "num_skills": int(num_skills),
            "job_level": job_level,
            "is_remote": bool(is_remote),
            "degree_required": degree_required,
            "experience_missing": int(experience_missing),
            "experience_tier": experience_tier,
            "skill_python": int(skill_python),
            "skill_sql": int(skill_sql),
            "skill_spark": int(skill_spark),
            "skill_aws": int(skill_aws),
            "is_senior": int(job_level in ["Senior", "Lead"]),
            "location": location.strip()
        }])

        # Match the model's expected column order
        input_data = input_data[
            list(model.feature_names_in_)
        ]

        try:
            prediction = float(model.predict(input_data)[0])

            st.success("Prediction completed!")

            st.metric(
                "Estimated Annual Salary (USD)",
                f"${prediction:,.2f}"
            )

            # Show approximate local currency equivalent
            if country in LOCAL_CURRENCIES:
                currency_code, currency_symbol = LOCAL_CURRENCIES[country]

                try:
                    exchange_rate, rate_date = get_exchange_rate(
                        currency_code
                    )

                    local_salary = prediction * exchange_rate

                    st.markdown(
                        f"**Approximate local equivalent:** "
                        f"({currency_symbol}{local_salary:,.2f} "
                        f"{currency_code})"
                    )

                    st.caption(
                        f"Exchange rate dated {rate_date}. "
                        "For reference only; rates may vary."
                    )

                except Exception:
                    st.info(
                        "Local currency conversion is temporarily "
                        "unavailable. The USD prediction is still shown."
                    )

            st.caption(
                "Predictions are in USD because the training "
                "salaries were standardized to USD. Model held-out "
                "test MAE was approximately $17,619; individual "
                "errors may be substantially larger."
            )

        except Exception as error:
            st.error(f"Prediction failed: {error}")

       
# ------------------------------------------------------------
# Model information
# ------------------------------------------------------------

st.divider()

st.subheader("About the Model")

st.write(
    "**Algorithm:** Optimized Random Forest Regressor"
)

st.write(
    "**Held-out Test MAE:** $17,619.30"
)

st.write(
    "**5-fold Cross-validation MAE:** $20,769.84"
)

st.write(
    "**Test R²:** 0.7337"
)

st.caption(
    "This model did not meet the project's $8,000 MAE target. "
    "It is intended for demonstration and further validation, "
    "not automated compensation decisions."
)
