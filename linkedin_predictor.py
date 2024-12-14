import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

# Load and preprocess the dataset
@st.cache_resource
def load_and_train_model():
    # Load the dataset
    s = pd.read_csv('finaldataset.csv')

    # Define the clean_sm function
    def clean_sm(x):
        return np.where(x == 1, 1, 0)

    # Preprocess the data
    s['sm_li'] = clean_sm(s['web1h'])
    s['income'] = np.where(s['income'] > 9, np.nan, s['income'])
    s['educ2'] = np.where(s['educ2'] > 8, np.nan, s['educ2'])
    s['parent'] = clean_sm(s['par'])
    s['married'] = clean_sm(np.where(s['marital'] == 1, 1, 0))
    s['female'] = clean_sm(np.where(s['gender'] == 2, 1, 0))
    s['age'] = np.where(s['age'] > 98, np.nan, s['age'])
    s = s.drop(columns=['web1h', 'par', 'marital', 'gender']).dropna()

    # Train the model
    X = s[['income', 'educ2', 'age', 'parent', 'married', 'female']]
    y = s['sm_li']
    model = LogisticRegression(class_weight='balanced', random_state=42)
    model.fit(X, y)

    return model

# Load the trained model
model = load_and_train_model()

# Streamlit app interface
st.title("LinkedIn User Predictor by Israel Izzy Mendoza")
st.write("This app predicts if someone is likely to use LinkedIn based on demographic data.")

# User input for the features
income_descriptions = {
    1: "Less than $10,000",
    2: "$10,000 to $20,000",
    3: "$20,000 to $30,000",
    4: "$30,000 to $40,000",
    5: "$40,000 to $50,000",
    6: "$50,000 to $75,000",
    7: "$75,000 to $100,000",
    8: "$100,000 to $150,000",
    9: "Over $150,000"
}
income = st.selectbox(
    "Income Level (1-9)",
    options=list(income_descriptions.keys()),  # Use keys from income_descriptions
    format_func=lambda x: f"{x}: {income_descriptions[x]}"  # Format with number and description
)
education_descriptions = {
    1: "Less than high school (Grades 1-8 or no formal schooling)",
    2: "High school incomplete (Grades 9-11 or Grade 12 with NO diploma)",
    3: "High school graduate (Grade 12 with diploma or GED certificate)",
    4: "Some college, no degree (includes some community college)",
    5: "Two-year associate degree from a college or university",
    6: "Four-year college or university degree/Bachelor’s degree (e.g., BS, BA, AB)",
    7: "Some postgraduate or professional schooling, no postgraduate degree (e.g., some graduate school)",
    8: "Postgraduate or professional degree, including master’s, doctorate, medical or law degree (e.g., MA, MS, PhD, MD, JD)",
    98: "Don’t know",
    99: "Refused"
}
education = st.selectbox(
    "Education Level (1-8)",
    options=list(education_descriptions.keys()),  # Use keys from education_descriptions
    format_func=lambda x: f"{x}: {education_descriptions[x]}"  # Format with number and description
)
age = st.slider("Age", min_value=18, max_value=98, value=30)
parent = st.radio("Are you a parent?", options=["Yes", "No"])
married = st.radio("Are you married?", options=["Yes", "No"])
gender = st.radio("Gender", options=["Female", "Male"])

# Format the input into a DataFrame for prediction
user_data = pd.DataFrame({
    'income': [income],
    'educ2': [education],
    'age': [age],
    'parent': [1 if parent == "Yes" else 0],
    'married': [1 if married == "Yes" else 0],
    'female': [1 if gender == "Female" else 0]
})

# Make predictions when a button is clicked
if st.button("Predict"):
    prediction = model.predict(user_data)[0]
    probability = model.predict_proba(user_data)[0, 1]

    # Display results
    st.write(f"Prediction: {'LinkedIn User' if prediction == 1 else 'Not a LinkedIn User'}")
    st.write(f"Probability of being a LinkedIn User: {probability:.2f}")
#Custom Color for Visual
st.markdown(
    """
    <style>
    /* Set the background color */
    .stApp {
        background-color: #blue; /* Replace with your desired color */
    }
    </style>
    """,
    unsafe_allow_html=True
)