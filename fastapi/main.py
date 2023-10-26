import csv
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from joblib import load


# Import the model
model = load('breast_cancer_model.joblib')

# Set app
app = FastAPI()

# Initiate the class
class PatientData(BaseModel):
    patient_id: int


@app.post("/predict")
def predict(patient_data: PatientData):
    ''' Convert data.csv into a dictionary to get the patient id, as the key,
    and the features as values
    Returns string with the prediction of class and the probability of
    the cancer being malignant
    '''
    with open("data.csv", 'r') as file:
        reader = csv.reader(file)
        d = {i: np.array(value) for i, value in enumerate(reader)}

    if patient_data.patient_id in d:
        features = d[patient_data.patient_id]
        dict_cancer = {0: 'benign', 1: "malignant"}
        prediction = model.predict([features])[0]
        probability = model.predict_proba([features])
        malignancy_probability = probability[0][1]

        return f"Patient with ID {patient_data.patient_id}, the tumors prediction is {dict_cancer[prediction]}. There is {malignancy_probability* 100}% chance that the tumor is malignant"
    else:
        raise HTTPException(status_code=404, detail="Patient ID does not exit")
