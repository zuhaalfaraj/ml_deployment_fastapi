# Put the code for your API here.
from typing import Optional
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, Body
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import joblib
import os
import pandas as pd
from starter.ml.model import inference
from starter.ml.data import process_data
# Loading the saved model
model = joblib.load('model/trained_model.pkl')
encoder= joblib.load('model/encoder.pkl')

class DataItem(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(..., alias='education-num')
    marital_status: str = Field(..., alias='marital-status')
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(..., alias='capital-gain')
    capital_loss: int = Field(..., alias='capital-loss')
    hours_per_week: int = Field(..., alias='hours-per-week')
    native_country: str = Field(..., alias='native-country')
    class Config:
        populate_by_name = True
        json_schema_extra = {
            "examples": [
                {
                    "age": 39,
                    "workclass": "State-gov",
                    "fnlgt": 77516,
                    "education": "Bachelors",
                    "education-num": 13,
                    "marital-status": "Never-married",
                    "occupation": "Adm-clerical",
                    "relationship": "Not-in-family",
                    "race": "White",
                    "sex": "Male",
                    "capital-gain": 2174,
                    "capital-loss": 0,
                    "hours-per-week": 40,
                    "native-country": "United-States"
                }
            ]
        }

app = FastAPI()


@app.get("/")
def read_root():
    return {"message": "Welcome to the model inference API!"}


@app.post("/predict/")
def make_prediction(item: DataItem):

    try:
        # Convert the Pydantic object to a DataFrame
        data_dict = item.dict(by_alias=True)
        data_df = pd.DataFrame({k:[v] for k,v in data_dict.items()})

        cat_features = [
            "workclass",
            "education",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "native-country"
        ]

        # Process data
        X, _, _, _ = process_data(
            data_df, categorical_features=cat_features, encoder=encoder, training=False
        )

        # Perform prediction

        prediction = inference(model, X)
        print(prediction)

        return {"prediction": int(prediction[0])}

    except Exception as e:
        print(e)  # Log the exception for debugging
        raise HTTPException(status_code=500, detail=str(e))