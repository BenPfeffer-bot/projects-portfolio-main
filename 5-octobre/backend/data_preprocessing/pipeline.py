# pipeline.py
"""
Data Preprocessing Pipeline

This script orchestrates the entire data preprocessing pipeline:
- Loading raw or intermediate data from specified directories.
- Applying cleaning, validation, and transformations to the data.
- Saving cleaned datasets to a designated directory for downstream use.

Adjust configuration paths and filenames in config.py as needed.
"""

import os
import pandas as pd
from typing import Optional, Tuple
import sys

sys.path.append(
    "/Users/benpfeffer/Library/Mobile Documents/com~apple~CloudDocs/projects-portfolio-main/5-octobre/"
)
from backend.data_preprocessing.config import (
    PROCESSED_DATA_DIR,
    CLEANED_DATA_DIR,
    CART_FILENAME,
    ORDER_FILENAME,
    INVENTORY_FILENAME,
    RETAIL_FILENAME,
)
from backend.data_preprocessing.loaders import load_data
from backend.data_preprocessing.cleaners import (
    standardize_column_names,
    remove_abandoned_carts,
    convert_to_datetime,
    remove_rows_before_date,
    remove_specific_clients,
    remove_missing_required,
    remove_duplicates,
    clean_currency_column,
    # transform_abandoned_cart_order_id,  # Import the new function
)
from backend.data_preprocessing.validators import validate_schema
from backend.data_preprocessing.schemas import (
    CartData,
    OrderData,
    InventoryData,
    RetailData,
)


def preprocess_cart_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the Cart dataset.
    Steps:
    - Standardize column names
    - Remove abandoned carts with zero totals
    - Convert date columns to datetime
    - Remove missing required fields and duplicates
    """
    # Original column names mapping
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

    # Map French column names to standardized names
    column_mapping = {
        "id_commande": "order_id",
        "référence": "reference",
        "client": "customer",
        "total": "total_amount",
        "transporteur": "carrier",
        "date": "order_date",
    }

    # Rename columns if they exist
    existing_cols = set(df.columns) & set(column_mapping.keys())
    df = df.rename(columns={col: column_mapping[col] for col in existing_cols})

    # Transform into float
    if "total_amount" in df.columns:
        df = clean_currency_column(df, "total_amount")

    # Remove abandoned carts
    if "order_id" in df.columns and "total_amount" in df.columns:
        df = remove_abandoned_carts(df, id_col="order_id", total_col="total_amount")

    # Convert date
    if "order_date" in df.columns:
        df = convert_to_datetime(df, date_col="order_date")

    # Remove missing required fields
    required_cols = [
        col for col in ["order_id", "total_amount", "order_date"] if col in df.columns
    ]
    if required_cols:
        df = remove_missing_required(df, required_cols)

    # Remove duplicates
    subset_cols = [col for col in ["order_id", "order_date"] if col in df.columns]
    if subset_cols:
        df = remove_duplicates(df, subset=subset_cols)

    return df


def preprocess_order_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the Order dataset.
    Steps:
    - Standardize column names.
    - Convert date columns to datetime.
    - Remove old orders before a cutoff date.
    - Remove unwanted clients.
    - Remove missing required fields and duplicates.
    """
    df = standardize_column_names(df)
    df = convert_to_datetime(df, date_col="date")
    df = remove_rows_before_date(df, date_col="date", cutoff="2021-03-31")
    df = remove_specific_clients(
        df, client_col="client", unwanted_list=["L. Pfeffer", "M. Vincent"]
    )
    df = remove_missing_required(df, ["id", "total", "date"])
    df = remove_duplicates(df, subset=["id", "date"])

    # Transform into float
    if "total" in df.columns:
        df = clean_currency_column(df, "total")

    return df


def preprocess_inventory_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the Inventory dataset.
    Steps:
    - Standardize column names.
    - Clean currency columns.
    - Remove missing required fields and duplicates.
    """
    df = standardize_column_names(df)
    for col in ["factory_price", "retail", "retail_us"]:
        if col in df.columns:
            df = clean_currency_column(df, col)
    df = remove_missing_required(df, ["id", "qty", "factory_price", "retail"])
    df = remove_duplicates(df, subset=["id", "ean"])
    # df = df.drop(columns=["ean"])
    return df


def preprocess_retail_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the Retail dataset.
    Steps:
    - Standardize column names.
    - Clean currency columns.
    - Convert date columns to datetime.
    - Remove missing required fields and duplicates.
    """
    df = standardize_column_names(df)
    for col in ["pv_ttc", "ca_ttc"]:
        if col in df.columns:
            df = clean_currency_column(df, col)
    df = convert_to_datetime(df, date_col="date")
    df = remove_missing_required(df, ["date", "pv_ttc", "ca_ttc"])
    df = remove_duplicates(df, subset=["date", "ref", "cust"])
    df = df.drop(columns=["ca_ttc"])
    return df


def save_data(df: pd.DataFrame, file_path: str) -> None:
    """
    Save a DataFrame to a CSV file.
    """
    try:
        df.to_csv(file_path, index=False)
        print(f"[INFO] Saved cleaned dataset to {file_path}")
    except Exception as e:
        print(f"[ERROR] Could not save data to {file_path}: {e}")


def run_pipeline() -> Tuple[Optional[pd.DataFrame], ...]:
    """
    Orchestrates the entire data preprocessing pipeline.
    - Loads data from raw or processed directories.
    - Applies preprocessing to each dataset (cart, order, inventory, retail).
    - Validates datasets (optional).
    - Saves cleaned datasets to the cleaned directory.

    Returns:
        A tuple of DataFrames (cart_df, order_df, inventory_df, retail_df).
    """
    # Build file paths for raw or processed data
    cart_path = os.path.join(PROCESSED_DATA_DIR, CART_FILENAME)
    order_path = os.path.join(PROCESSED_DATA_DIR, ORDER_FILENAME)
    inventory_path = os.path.join(PROCESSED_DATA_DIR, INVENTORY_FILENAME)
    retail_path = os.path.join(PROCESSED_DATA_DIR, RETAIL_FILENAME)

    # 1. Load datasets
    print("[INFO] Loading datasets...")
    cart_df = load_data(cart_path)
    order_df = load_data(order_path)
    inventory_df = load_data(inventory_path)
    retail_df = load_data(retail_path)

    if not all(
        [
            cart_df is not None,
            order_df is not None,
            inventory_df is not None,
            retail_df is not None,
        ]
    ):
        print("[ERROR] Failed to load one or more datasets.")
        return None, None, None, None

    # 2. Preprocess datasets
    print("[INFO] Preprocessing Cart data...")
    cart_df = preprocess_cart_data(cart_df)
    # cart_df = transform_abandoned_cart_order_id(cart_df)  # Apply the new transformation

    print("[INFO] Preprocessing Order data...")
    order_df = preprocess_order_data(order_df)

    print("[INFO] Preprocessing Inventory data...")
    inventory_df = preprocess_inventory_data(inventory_df)

    print("[INFO] Preprocessing Retail data...")
    retail_df = preprocess_retail_data(retail_df)

    # 3. Validate schemas (optional step)
    print("[INFO] Validating schemas...")
    cart_valid, cart_err_idx, cart_err_msgs = validate_schema(
        cart_df, CartData, raise_errors=False
    )
    if not cart_valid:
        print(f"[WARN] Cart data had {len(cart_err_idx)} invalid rows.")

    order_valid, order_err_idx, order_err_msgs = validate_schema(
        order_df, OrderData, raise_errors=False
    )
    if not order_valid:
        print(f"[WARN] Order data had {len(order_err_idx)} invalid rows.")

    inventory_valid, inventory_err_idx, inventory_err_msgs = validate_schema(
        inventory_df, InventoryData, raise_errors=False
    )
    if not inventory_valid:
        print(f"[WARN] Inventory data had {len(inventory_err_idx)} invalid rows.")

    retail_valid, retail_err_idx, retail_err_msgs = validate_schema(
        retail_df, RetailData, raise_errors=False
    )
    if not retail_valid:
        print(f"[WARN] Retail data had {len(retail_err_idx)} invalid rows.")

    # 4. Save cleaned datasets
    print("[INFO] Saving cleaned datasets...")
    save_data(cart_df, os.path.join(CLEANED_DATA_DIR, CART_FILENAME))
    save_data(order_df, os.path.join(CLEANED_DATA_DIR, ORDER_FILENAME))
    save_data(inventory_df, os.path.join(CLEANED_DATA_DIR, INVENTORY_FILENAME))
    save_data(retail_df, os.path.join(CLEANED_DATA_DIR, RETAIL_FILENAME))

    return cart_df, order_df, inventory_df, retail_df


if __name__ == "__main__":
    """
    Run the pipeline when executed as a standalone script.
    Example usage:
        python pipeline.py
    """
    print("[INFO] Starting preprocessing pipeline...")
    cart, order, inventory, retail = run_pipeline()
    if any(df is None for df in [cart, order, inventory, retail]):
        print("[ERROR] Preprocessing pipeline encountered errors.")
    else:
        print("[INFO] Preprocessing pipeline completed successfully!")
