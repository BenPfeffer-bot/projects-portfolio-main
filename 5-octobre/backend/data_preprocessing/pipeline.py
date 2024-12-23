import os
import pandas as pd
from typing import Optional, Tuple

from config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    CLEANED_DATA_DIR,
    CART_FILENAME,
    ORDER_FILENAME,
    INVENTORY_FILENAME,
    RETAIL_FILENAME,
)
from loaders import load_data
from cleaners import (
    standardize_column_names,
    remove_abandoned_carts,
    convert_to_datetime,
    remove_rows_before_date,
    remove_specific_clients,
    remove_missing_required,
    remove_duplicates,
    clean_currency_column,
)
from validators import validate_schema
from schemas import CartData, OrderData, InventoryData, RetailData


def preprocess_cart_data(df: pd.DataFrame) -> pd.DataFrame:
    # Example: Remove abandoned, convert date, remove duplicates, etc.
    df = standardize_column_names(df)
    df = remove_abandoned_carts(df, "id_commande", "total")
    df = convert_to_datetime(df, "date")
    df = remove_missing_required(df, ["id_commande", "total", "date"])
    df = remove_duplicates(df, ["id_commande", "date"])
    return df


def preprocess_order_data(df: pd.DataFrame) -> pd.DataFrame:
    df = standardize_column_names(df)
    df = convert_to_datetime(df, "date")
    df = remove_rows_before_date(df, "date", cutoff="2021-03-31")
    df = remove_specific_clients(df, "client", ["L. Pfeffer", "M. Vincent"])
    df = remove_missing_required(df, ["id", "total", "date"])
    df = remove_duplicates(df, ["id", "date"])
    return df


def preprocess_inventory_data(df: pd.DataFrame) -> pd.DataFrame:
    df = standardize_column_names(df)
    # Example: Clean currency columns if you have them
    for col in ["factory_price", "retail", "retail_us"]:
        if col in df.columns:
            df = clean_currency_column(df, col)
    df = remove_missing_required(df, ["id", "qty", "factory_price", "retail"])
    df = remove_duplicates(df, ["id", "ean"])
    return df


def preprocess_retail_data(df: pd.DataFrame) -> pd.DataFrame:
    df = standardize_column_names(df)
    # Clean currency columns
    for col in ["pv_ttc", "ca_ttc"]:
        if col in df.columns:
            df = clean_currency_column(df, col)
    df = convert_to_datetime(df, "date")
    df = remove_missing_required(df, ["date", "pv_ttc", "ca_ttc"])
    df = remove_duplicates(df, ["date", "ref", "cust"])
    return df


def save_data(df: pd.DataFrame, file_path: str) -> None:
    """
    Save a DataFrame to CSV.
    """
    try:
        df.to_csv(file_path, index=False)
        print(f"[INFO] Saved cleaned dataset to {file_path}")
    except Exception as e:
        print(f"[ERROR] Could not save data to {file_path}: {e}")


def run_pipeline() -> Tuple[Optional[pd.DataFrame], ...]:
    """
    Orchestrates the entire data preprocessing pipeline for
    cart, order, inventory, and retail.

    :return: tuple of DataFrames (cart_df, order_df, inventory_df, retail_df),
             or (None, None, None, None) if there's a failure.
    """
    # Build file paths for raw or processed data
    cart_path = os.path.join(PROCESSED_DATA_DIR, CART_FILENAME)
    order_path = os.path.join(PROCESSED_DATA_DIR, ORDER_FILENAME)
    inventory_path = os.path.join(PROCESSED_DATA_DIR, INVENTORY_FILENAME)
    retail_path = os.path.join(PROCESSED_DATA_DIR, RETAIL_FILENAME)

    # 1. Load each dataset
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

    # 2. Preprocess each dataset
    cart_df = preprocess_cart_data(cart_df)
    order_df = preprocess_order_data(order_df)
    inventory_df = preprocess_inventory_data(inventory_df)
    retail_df = preprocess_retail_data(retail_df)

    # 3. Validate schemas (optional)
    #    If you want strict Pydantic validation, you can do:
    cart_valid, cart_err_idx, cart_err_msgs = validate_schema(
        cart_df, CartData, raise_errors=False
    )
    if not cart_valid:
        print(
            f"[WARN] Cart data had {len(cart_err_idx)} invalid rows according to CartData schema."
        )

    # Similarly for orders, etc.:
    # order_valid, order_err_idx, order_err_msgs = validate_schema(order_df, OrderData, raise_errors=False)
    # if not order_valid:
    #     ...

    # 4. Save cleaned data
    cleaned_cart_path = os.path.join(CLEANED_DATA_DIR, CART_FILENAME)
    cleaned_order_path = os.path.join(CLEANED_DATA_DIR, ORDER_FILENAME)
    cleaned_inventory_path = os.path.join(CLEANED_DATA_DIR, INVENTORY_FILENAME)
    cleaned_retail_path = os.path.join(CLEANED_DATA_DIR, RETAIL_FILENAME)

    save_data(cart_df, cleaned_cart_path)
    save_data(order_df, cleaned_order_path)
    save_data(inventory_df, cleaned_inventory_path)
    save_data(retail_df, cleaned_retail_path)

    return cart_df, order_df, inventory_df, retail_df
