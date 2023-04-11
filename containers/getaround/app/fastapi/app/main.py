# app/main.py
import mlflow 
import uvicorn
import json
import pandas as pd 
from pydantic import BaseModel
from typing import Literal, List, Union
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from app.db import database, User
import warnings
import platform
import numpy as np
import logging
import xgboost

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define app description and tags
app_description = """
This is a simple API for Getaround, a car-sharing platform. The API provides two endpoints: 

1. `/`: This is a simple default endpoint that displays a welcome message.

2. `/prediction`: This endpoint takes a set of car characteristics and returns a rental price suggestion based on those characteristics.
"""

app_tags = [
    {
        "name": "Default",
        "description": "Default endpoint",
    },
    {
        "name": "Prediction",
        "description": "Car rental price suggestion based on car characteristics"
    }
]

# Define the FastAPI app
app = FastAPI(
    title="Getaround API",
    description=app_description,
    version="1.0.0",
    openapi_tags=app_tags,
)

@app.get("/")
async def read_root():
    return await User.objects.all()

class PredictionFeatures(BaseModel):
    model_key: str = "Citroën"
    mileage: int = 150000
    engine_power: int = 100
    fuel: str = "diesel"
    paint_color: str = "green"
    car_type: str = "convertible"
    private_parking_available: bool = True
    has_gps: bool = True
    has_air_conditioning: bool = True
    automatic_car: bool = True
    has_getaround_connect: bool = True
    has_speed_regulator: bool = True
    winter_tires: bool = True


@app.post("/prediction", tags=["Rental price prediction"])
async def predict(features: PredictionFeatures):
    try:
        # Read data
        df = pd.DataFrame(dict(features), index=[0])

        logged_model = 'runs:/c518611fef064b2ea8fc3c4e3e011a51/getaround'

        # Load model as a PyFuncModel.
        loaded_model = mlflow.pyfunc.load_model(logged_model)
        prediction = loaded_model.predict(df)

        # Format response
        response = {"prediction": prediction.tolist()[0]}
        return response

    except Exception as e:
        # Log the error
        logger.error(str(e))
        # Raise an HTTPException with a 500 error code
        raise HTTPException(
            status_code=500,
            detail="Internal Server Error"
        )


@app.on_event("startup")
async def startup():
    try:
        if not database.is_connected:
            await database.connect()
        # create a dummy entry
        await User.objects.get_or_create(email="contact@pryda.dev")
        logger.info("Connected to the database")

    except Exception as e:
        # Log the error
        logger.error(str(e))
        # Raise an HTTPException with a 500 error code
        raise HTTPException(
            status_code=500,
            detail="Internal Server Error"
        )


@app.on_event("shutdown")
async def shutdown():
    try:
        if database.is_connected:
            await database.disconnect()
            logger.info("Disconnected from the database")
            
    except Exception as e:
        # Log the error
        logger.error(str(e))
        # Raise an HTTPException with a 500 error code
        raise HTTPException(
            status_code=500,
            detail="Internal Server Error"
        )