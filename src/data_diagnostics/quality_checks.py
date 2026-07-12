# Data quality check (EDA and health report)

import numpy as np
from src.utils.config_loader import get_tabular_config, get_image_config

# Run quality checks on tabular data
def check_tabular_quality(X, y, config):
    print("EDA started for tabular data.")
    dataset_name = get_tabular_config(config)['name']
    report = {}

    # Dataset basic shape
    report['num_samples'] = X.shape[0]
    report['num_features'] = X.shape[1]
    print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}")

    # Class distribution
    unique, counts = np.unique(y, return_counts = True)
    # Avoid numpy int64
    class_distribution = dict(zip(unique.astype(int).tolist(), counts.astype(int).tolist()))
    report['class_distribution'] = class_distribution

    # Imbalance ratio
    majority = counts.max()
    minority = counts.min()
    imbalance_ratio = round(minority / majority, 4)
    report['imbalance_ratio'] = imbalance_ratio
    print(f"Class distribution: {class_distribution}")
    print(f"Imbalance ratio: {imbalance_ratio}")

    # Duplicate rows
    duplicates = X.duplicated().sum()
    report['duplicate_rows'] = int(duplicates)
    print(f"Duplicate rows: {duplicates}")

    # Feature min, max, mean, median, std
    feature_stats = {}
    for col in X.columns:
        feature_stats[col] = {
            'min': round(float(X[col].min()), 4),
            'max': round(float(X[col].max()), 4),
            'mean': round(float(X[col].mean()), 4),
            'std': round(float(X[col].std()), 4),
            'median': round(float(X[col].median()), 4),
        }
    report['feature_stats'] = feature_stats

    # Outliers (values more than 3 standard deviations from mean)
    outlier_count = {}
    for col in X.columns:
        mean = X[col].mean()
        std = X[col].std()
        outliers = ((X[col] < mean - 3 * std) | (X[col] > mean + 3 * std)).sum()
        outlier_count[col] = int(outliers)

    report['outlier_counts'] = outlier_count
    total_outliers = sum(outlier_count.values())
    print(f"Total outlier count: {total_outliers}")

    # Correlation matrix
    correlation = X.corr()
    report['correlation'] = correlation

    print(f"EDA completed for {dataset_name}")
    return report

# Run quality checks on image data
def check_image_quality(dataset, config):
    print("EDA started for image data.")
    dataset_name = get_image_config(config)['name']
    report = {}

    # Basic counts
    report['total_images'] = len(dataset)
    print(f"Total images: {len(dataset)}")

    # Class distribution
    targets = np.array(dataset.targets)
    unique, counts = np.unique(targets, return_counts = True)
    class_dist = dict(zip(unique.astype(int).tolist(), counts.astype(int).tolist()))
    report['class_distribution'] = class_dist
    print(f"Classes: {len(unique)}")
    print(f"Distribution: {class_dist}")

    # Imbalance ratio
    majority = counts.max()
    minority = counts.min()
    imbalance_ratio = round(float(minority / majority), 4)
    report['imbalance_ratio'] = imbalance_ratio
    print(f"Imbalance ratio: {imbalance_ratio}")

    print(f"EDA completed for {dataset_name}")
    return report

