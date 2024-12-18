# src/config.py

import os
import logging

# Attempt to locate project directories and handle errors
try:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
    PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
    CLEANED_DATA_DIR = os.path.join(BASE_DIR, "data", "cleaned")

    # Check existence of directories
    for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, CLEANED_DATA_DIR]:
        if not os.path.exists(d):
            raise FileNotFoundError(f"Required directory not found: {d}")

    # Filenames
    CART_FILENAME = "cart.csv"
    ORDER_FILENAME = "order.csv"

except Exception as e:
    print(f"Error in config setup: {e}")
    # Depending on the project needs, we could exit or handle differently.
    # For now, just print the error.


def load_logger():
    logs_dir = os.path.join(BASE_DIR, "logs")
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    log_file = os.path.join(logs_dir, "app.log")
    logging.basicConfig(filename=log_file, level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    return logger


def load_config():
    """
    Load configuration from env.yaml file.
    Returns a dictionary containing configuration values.
    """
    import yaml

    try:
        config_path = os.path.join(BASE_DIR, "env.yaml")
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        logger = load_logger()
        logger.error("env.yaml configuration file not found")
        return {}
    except yaml.YAMLError as e:
        logger = load_logger()
        logger.error(f"Error parsing env.yaml: {e}")
        return {}
    except Exception as e:
        logger = load_logger()
        logger.error(f"Unexpected error loading config: {e}")
        return {}
