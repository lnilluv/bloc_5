import re
import warnings

import numpy as np
import pandas as pd
# from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

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
    CatBoostRegressor(verbose=0)
]

# List of param_grids for each model
param_grids = [
{
    'model__depth': [6, 8, 10],
    'model__learning_rate': [0.05, 0.1, 0.2],
    'model__iterations': [1000, 1500, 2000],
    'model__l2_leaf_reg': [2, 3, 4],
    'model__colsample_bylevel': [0.5, 0.8, 1],
    'model__subsample': [0.5, 0.8, 1],
    'model__border_count': [32, 64, 128],
}
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

