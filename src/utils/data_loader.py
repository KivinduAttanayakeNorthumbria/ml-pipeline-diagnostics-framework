# Loading data set

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torchvision import datasets, transforms

from src.utils.config_loader import get_tabular_config, get_image_config


# Load the tabular dataset
def load_tabular_data(config):
    print("Loading dataset.")

    # Read settings for the currently active tabular dataset
    dataset_config = get_tabular_config(config)

    dataset_name = dataset_config['name']
    dataset_file = dataset_config['file_name']
    target_column = dataset_config['target_column']
    test_size = dataset_config['test_size']
    missing_value = dataset_config['missing_value']
    remove_duplicates = dataset_config['remove_duplicates']
    na_values = dataset_config['na_values']
    seed = config['random_seeds']['primary_seed']

    print(f"Dataset: {dataset_name}")

    # Load data into dataframe
    raw_path = os.path.join(config['paths']['data_raw'], dataset_file)
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Dataset file not found at {raw_path}")

    if len(na_values) > 0:
        df = pd.read_csv(raw_path, na_values=na_values)
    else:
        df = pd.read_csv(raw_path)

    # Remove duplicates according to the configuration value
    if remove_duplicates:
        before_count = len(df)
        df = df.drop_duplicates()
        after_count = len(df)
        print(f"Removed {before_count - after_count} duplicate rows.")
    else:
        print(f"Duplicate rows kept ({df.duplicated().sum()} found).")

    # Separate features columns and target column
    y = df[target_column]
    X = df.drop(columns=[target_column])

    # Fill null values using user defined value
    missing_count = X.isnull().sum().sum()
    if missing_count > 0:
        print(f"Found {missing_count} missing values. Filling with {missing_value}.")
        if missing_value == 'mean':
            X = X.fillna(X.mean(numeric_only=True))
        elif missing_value == 'median':
            X = X.fillna(X.median(numeric_only=True))
        elif missing_value == 'mode':
            X = X.fillna(X.mode().iloc[0])
        else:
            raise ValueError("Please provide a valid missing value strategy: mean, median or mode.")
    else:
        print("No missing value found.")

    # Encode target labels to 0 and 1
    le = LabelEncoder()
    y = le.fit_transform(y)

    # Convert categorical columns to numbers
    categorical_columns = X.select_dtypes(include=['category', 'object']).columns
    for column in categorical_columns:
        X[column] = LabelEncoder().fit_transform(X[column].astype(str))

    remaining_missing = X.isnull().sum().sum()
    if remaining_missing > 0:
        raise ValueError(f"{remaining_missing} missing values remain after imputation. Check the fill strategy.")

    # Scale all features to similar ranges
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # Split data into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    print(f"Data loaded for {X_train.shape[0]} train, {X_test.shape[0]} test")
    print(f" Features: {X_train.shape[1]}")

    return X_train, X_test, y_train, y_test


# Load the image dataset
def load_image_data(config):
    print("Loading dataset.")

    # Read settings for the currently active image dataset
    dataset_config = get_image_config(config)

    dataset_name = dataset_config['name']
    folder_name = dataset_config['folder_name']
    num_channels = dataset_config['num_channels']
    image_size = dataset_config['image_size']

    print(f"Dataset: {dataset_name}")

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

    # Build the transform list
    transform_list = [transforms.Resize((image_size, image_size))]

    if num_channels == 1:
        transform_list.append(transforms.Grayscale(num_output_channels=1))

    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(mean, std))

    transform = transforms.Compose(transform_list)

    train_set = datasets.ImageFolder(root=train_path, transform=transform)
    test_set = datasets.ImageFolder(root=test_path, transform=transform)

    sample_image, sample_label = train_set[0]
    if sample_image.shape[0] != num_channels:
        raise ValueError(
            f"Channel mismatch. Config says {num_channels} channels "
            f"but the loaded image has {sample_image.shape[0]}."
        )

    print(f"Loaded: {len(train_set)} train and {len(test_set)} test")
    print(f"Image size: {image_size}, Channels: {num_channels}")
    print(f"Classes: {len(train_set.classes)}")
    print(f"Class names: {train_set.classes}")

    return train_set, test_set