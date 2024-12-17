# src/config.py

import os

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
