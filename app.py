

import streamlit as st
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the model and vectorizer from the .pkl files
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Streamlit App UI
st.title("Loan Eligibility Prediction")

st.write("""
         This app predicts whether a person is eligible for a loan or not based on the bank data.
         Please provide the necessary information below to get a prediction.
         """)

# Input fields
gender = st.number_input("Gender", min_value=0, max_value=1, value=367)
married = st.number_input("Married", min_value=0 , max_value=1, value=367 )
education = st.number_input("Education", min_value=0, max_value=1, value=367) 
self_employed = st.number("Self_Employed", min_value=0, max_value=1, value=367)
applicant_income = st.number_input("ApplicantIncome ($)", min_value=150, max_value=81000, value=367)
coapplicant_income = st.number_input("CoapplicantIncome", min_value=0, max_value=41667, value=367)
loan_amount = st.number_input("LoanAmount ($)", min_value=9, max_value=700, value=367)
loan_term = st.selectbox("Loan_Amount_Term (Years)",  min_value=12, max_value=480, value=367)

# Assuming there is a text field for additional comments or other features
comments = st.text_area("Additional Comments", "")

# Prepare the features for prediction
def prepare_input_data(gender, married, education, self_employed, applicant_income, coapplicant_income, loan_amount, loan_term):
    # You can modify this based on how your model is structured
    data = {
        'gender': gender,
        'married': married,
        'education': education,
        'self_employed': self_employed,
        'applicant_income': applicant_income,
        'coapplicant_income': coapplicant_income,
        'loan_amount': loan_amount,
        'loan_term': loan_term,
    }
    
    df = pd.DataFrame([data])
    
    # If you have some text feature that needs transformation (like 'comments'), apply the vectorizer
    df['comments'] = vectorizer.transform(df['comments']).toarray()
    
    return df

# Prediction
if st.button('Predict Eligibility'):
    input_data = prepare_input_data(Gender, Married, Education, Self_Employed, ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History, Property_Area, Loan_Status)
    
    # Make prediction using the model
    prediction = model.predict(input_data)
    
    # Display result
    if prediction == 1:
        st.success("The person is eligible for a loan.")
    else:
        st.error("The person is not eligible for a loan.")

# Run the Streamlit app by executing this in the terminal:
# streamlit run app.py
