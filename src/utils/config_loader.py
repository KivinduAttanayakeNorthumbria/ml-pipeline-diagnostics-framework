# Load experiment settings from the YAML configuration file.

import os
import yaml

# Set the path, check, open and read the YAML file.
def load_config(config_path = None):
    if config_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        config_path = os.path.join(project_root, "config", "experiment_config.yaml")

    # Check the file.
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at: {config_path}")

    # Open the YAML file and read.
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config