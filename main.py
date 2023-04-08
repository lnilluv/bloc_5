import os

import boto3
import mlflow
import numpy as np
import pandas as pd
from botocore.client import Config
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

# Create a connection to S3 using the Boto3 library
s3 = boto3.resource('s3',
                    endpoint_url=os.getenv('MLFLOW_S3_ENDPOINT_URL'),
                    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
                    config=Config(signature_version='s3v4'),
                    region_name='us-east-1')

# Set your variables for your environment
EXPERIMENT_NAME = "getaround-bloc5"

# Set tracking URI to your Heroku application
mlflow.set_tracking_uri("https://mlflow.pryda.dev")

# Set experiment's info 
mlflow.set_experiment(EXPERIMENT_NAME)

# Get our experiment info
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

# Call mlflow autolog
mlflow.xgboost.autolog()

# Start the experiment run
with mlflow.start_run(experiment_id=experiment.experiment_id):

    # Load the data
    df = pd.read_csv('data/get_around_pricing_project.csv', index_col=0)

    # Extract the features
    X = df.drop('rental_price_per_day', axis=1)

    # Extract the target column
    y = df.loc[:, 'rental_price_per_day']

    # Train / test split 
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size = 0.2)

    # determine categorical and numerical features
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object', 'bool']).columns

    # Numerical Transformer
    numerical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
    ])

    # Categorical Transformer
    categorical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("numerical_transformer", numerical_transformer, numerical_features),
            ("categorical_transformer", categorical_transformer, categorical_features)
        ]
    )

    # List of models
    models = [
        XGBRegressor()
    ]

    # List of param_grids for each model
    param_grids = [
    {'model__gamma': 0, 'model__learning_rate': 0.1, 'model__max_depth': 10, 'model__min_child_weight': 5, 'model__n_estimators': 100}
    ]

    # Initialize an empty DataFrame to store the results
    results_df = pd.DataFrame(columns=['Model', 'Best_Params', 'Best_Score'])

    results = []

    for i, model in enumerate(models):
        param_grid = param_grids[i]
        
        # Create a pipeline with the preprocessor and the model
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])

        # Perform grid search with the current model and its param_grid
        grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)

        grid.fit(X_train, y_train)

