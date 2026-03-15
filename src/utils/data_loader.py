# Loading data set

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from src.utils.config_loader import load_config



# Load the tabular dataset
def load_tabular_data(config):
    print("Loading dataset.")

    # Read settings from config for tabular dataset
    dataset_name = config['datasets']['tabular_data']['name']
    dataset_file = config['datasets']['tabular_data']['file_name']
    target_column = config['datasets']['tabular_data']['target_column']
    test_size = config['datasets']['tabular_data']['test_size']
    seed = config['random_seeds']['primary_seed']

    # Load data into dataframe
    raw_path = os.path.join(config['paths']['data_raw'], dataset_file)
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Dataset file not found at {raw_path}")
    df = pd.read_csv(raw_path)

    # Separate features columns and target column
    y = df[target_column]
    X = df.drop(columns=[target_column])

    # Encode target labels to 0 and 1
    le = LabelEncoder()
    y = le.fit_transform(y)

    # Convert categorical columns to numbers
    categorical_cols = X.select_dtypes(include=['category', 'object']).columns
    for col in categorical_cols:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    # Fill missing values
    X = X.fillna(X.median())

    # Scale all features to similar ranges
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # Split data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)

    print(f"Data loaded for {X_train.shape[0]} train, {X_test.shape[0]} test")
    print(f" Features: {X_train.shape[1]}")

    return X_train, X_test, y_train, y_test


