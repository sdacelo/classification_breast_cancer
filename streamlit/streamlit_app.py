import streamlit as st
import requests


# Define the FastAPI endpoint URL
backend = "http://fastapi:8000/predict"

# Set page configuration
st.set_page_config(page_title="Cancer Prediction", page_icon="✅")

# Create a Streamlit web form
st.title("Cancer Prediction App")
st.write("Enter a patient ID to predict cancer:")

patient_id = st.number_input("Patient ID", min_value=0, step=1)
submit = st.button("Predict")

if submit:
    # Define the request payload
    payload = {"patient_id": patient_id}

    try:
        # Send a POST request to the FastAPI endpoint
        response = requests.post(backend, json=payload)

        if response.status_code == 200:
            result = response.text
            st.write(f"Prediction Result: {result}")
        else:
            st.error(f"Error: {response.status_code} - {response.text}")

    except requests.exceptions.RequestException as e:
        st.error(f"Error: {e}")
