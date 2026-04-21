# Loading data set

import os
from importlib.readers import remove_duplicates

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from src.utils.config_loader import load_config
from torchvision import datasets, transforms



# Load the tabular dataset
def load_tabular_data(config):
    print("Loading dataset.")

    # Read settings from config for tabular dataset
    dataset_name = config['datasets']['tabular_data']['name']
    dataset_file = config['datasets']['tabular_data']['file_name']
    target_column = config['datasets']['tabular_data']['target_column']
    test_size = config['datasets']['tabular_data']['test_size']
    seed = config['random_seeds']['primary_seed']
    missing_value = config['datasets']['tabular_data']['missing_value']
    remove_duplicates = config['datasets']['tabular_data']['remove_duplicates']

    # Load data into dataframe
    raw_path = os.path.join(config['paths']['data_raw'], dataset_file)
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Dataset file not found at {raw_path}")
    df = pd.read_csv(raw_path)

    # Remove duplicates according to the configuration value
    if remove_duplicates:
        before_count = len(df)
        df = df.drop_duplicates()
        after_count = len(df)

    # Separate features columns and target column
    y = df[target_column]
    X = df.drop(columns=[target_column])

    # Fill null values using user defined value
    missing_count = X.isnull().sum().sum()
    if missing_count > 0:
        if missing_value == 'mean':
            X = X.fillna(X.mean())
        elif missing_value == 'median':
            X = X.fillna(X.median())
        elif missing_value == 'mode':
            X = X.fillna(X.mode().iloc[0])
        else:
            raise ValueError("Please provide a valid missing value.")
    else:
        print("No missing value found.")

    # Encode target labels to 0 and 1
    le = LabelEncoder()
    y = le.fit_transform(y)

    # Convert categorical columns to numbers
    categorical_columns = X.select_dtypes(include=['category', 'object']).columns
    for column in categorical_columns:
        X[column] = LabelEncoder().fit_transform(X[column].astype(str))

    # Scale all features to similar ranges
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # Split data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)

    print(f"Data loaded for {X_train.shape[0]} train, {X_test.shape[0]} test")
    print(f" Features: {X_train.shape[1]}")

    return X_train, X_test, y_train, y_test

# Load the image dataset
def load_image_data(config ):
    print("Loading dataset.")

    dataset_config = config['datasets']['image_data']
    folder_name = dataset_config["folder_name"]
    num_channels = dataset_config["num_channels"]
    image_size = dataset_config["image_size"]

    # Set path for train and test
    train_path = os.path.join(config['paths']['data_raw'], folder_name, "train")
    test_path = os.path.join(config['paths']['data_raw'], folder_name, "test")

    # Check folders
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Dataset file not found at {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Dataset file not found at {test_path}")

    # Set normalisation using number of channels
    mean = tuple([0.5] * num_channels)
    std = tuple([0.5] * num_channels)

    # Resize all images to the same size for normalisation
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])

    # Load from folder structure using image folder
    train_set = datasets.ImageFolder(root = train_path, transform = transform)
    test_set = datasets.ImageFolder(root = test_path, transform = transform)

    print(f"Loaded: {len(train_set)} train and {len(test_set)} test")
    print(f"Image size: {image_size}")
    print(f"Classes: {len(train_set.classes)}")

    return train_set, test_set


