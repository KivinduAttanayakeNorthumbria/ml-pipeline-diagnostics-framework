# Configuration loader

import os
import json
import yaml


# Load the YAML configuration file
def load_config(config_path=None):
    if config_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        config_path = os.path.join(project_root, 'config', 'experiment_config.yaml')

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


# Get the currently active tabular dataset settings
def get_tabular_config(config):
    active = config['active_tabular']
    return config['datasets']['tabular'][active]


# Get the currently active image dataset settings
def get_image_config(config):
    active = config['active_image']
    return config['datasets']['image'][active]


# Build the key used to store and look up a threshold
def build_threshold_key(dataset_short_name, model_name, uq_method):
    return f"{dataset_short_name}_{model_name}_{uq_method}"


# Save a computed threshold into the thresholds file
def save_threshold(config, dataset_short_name, model_name, uq_method, threshold_value):
    path = config['paths']['thresholds_file']

    if os.path.exists(path):
        with open(path, 'r') as f:
            thresholds = json.load(f)
    else:
        thresholds = {}

    key = build_threshold_key(dataset_short_name, model_name, uq_method)
    thresholds[key] = round(float(threshold_value), 6)

    with open(path, 'w') as f:
        json.dump(thresholds, f, indent=2)

    print(f"Threshold saved: {key} = {thresholds[key]}")


# Load a previously computed threshold
def load_threshold(config, dataset_short_name, model_name, uq_method):
    path = config['paths']['thresholds_file']

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Thresholds file not found at {path}. "
            f"Run the clean baseline notebook for this dataset and model first."
        )

    with open(path, 'r') as f:
        thresholds = json.load(f)

    key = build_threshold_key(dataset_short_name, model_name, uq_method)

    if key not in thresholds:
        raise KeyError(
            f"Threshold '{key}' not found. "
            f"Run the clean baseline notebook for {dataset_short_name} / {model_name} first. "
            f"Available keys: {list(thresholds.keys())}"
        )

    return thresholds[key]