import os

import boto3
import mlflow
import numpy as np
import pandas as pd
from botocore.client import Config
from mlflow.models.signature import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

# Create a connection to S3 using the Boto3 library
s3 = boto3.resource(
    "s3",
    endpoint_url=os.getenv("MLFLOW_S3_ENDPOINT_URL"),
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    config=Config(signature_version="s3v4"),
    region_name="us-east-1",
)

EXPERIMENT_NAME = "getaround-bloc5"

mlflow.set_tracking_uri("https://mlflow.pryda.dev")

mlflow.set_experiment(EXPERIMENT_NAME)

experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

mlflow.sklearn.autolog()

# Load the data
df = pd.read_csv("data/get_around_pricing_project.csv", index_col=0)

# Extract the features
X = df.drop("rental_price_per_day", axis=1)

# Extract the target column
y = df.loc[:, "rental_price_per_day"]

# Train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, test_size=0.2
)

# determine categorical and numerical features
numerical_features = X.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X.select_dtypes(include=["object", "bool"]).columns

# Numerical Transformer
numerical_transformer = Pipeline(
    [("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())]
)

# Categorical Transformer
categorical_transformer = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numerical_transformer", numerical_transformer, numerical_features),
        ("categorical_transformer", categorical_transformer, categorical_features),
    ]
)

# List of models
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            GradientBoostingRegressor(
                learning_rate=0.1, max_depth=5, min_samples_split=10, n_estimators=200
            ),
        ),
    ]
)

# Log experiment to MLFlow
with mlflow.start_run(experiment_id=experiment.experiment_id):
    model.fit(X_train, y_train)
    predictions = model.predict(X_train)

    mlflow.log_metric("Train Score", model.score(X_train, y_train))
    mlflow.log_metric("Test Score", model.score(X_test, y_test))

    # Log model
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="getaround",
        registered_model_name="gradienboosting",
        signature=infer_signature(X_train, predictions),
    )
