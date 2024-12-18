import os
import pandas as pd
import numpy as np
from datetime import datetime
from config import PROCESSED_DATA_DIR, CLEANED_DATA_DIR, CART_FILENAME, ORDER_FILENAME


def standardize_column_names(df):
    """
    Standardize column names by stripping whitespace and possibly renaming columns
    to a uniform format if needed.
    """
    df.columns = [col.strip() for col in df.columns]
    return df


def clean_total_column(df, total_col="Total"):
    """
    Clean and convert the Total column to float by removing currency symbols,
    spaces, and converting locale-specific formats (commas to dots).

    We also report how many values could not be converted.
    """
    if total_col not in df.columns:
        print(f"Warning: '{total_col}' column not found in DataFrame.")
        return df

    def clean_amount(x):
        if pd.isna(x):
            return np.nan
        if isinstance(x, str):
            # Remove currency symbols and various whitespace/formatting characters
            cleaned = (
                x.replace("€", "")
                .replace("$", "")
                .replace("£", "")
                .replace("¥", "")
                .replace("\xa0", "")
                .replace(" ", "")
            )
            cleaned = cleaned.replace(",", ".")
            # Attempt float conversion
            try:
                return float(cleaned)
            except ValueError:
                return np.nan
        return x

    # Apply cleaning
    original_nonnull_count = df[total_col].notnull().sum()
    df[total_col] = df[total_col].apply(clean_amount)
    converted_nonnull_count = df[total_col].notnull().sum()
    print(
        f"Converted {converted_nonnull_count}/{original_nonnull_count} non-null '{total_col}' values to float successfully."
    )

    return df


def convert_date_column(df, date_col="Date"):
    """
    Convert the Date column to datetime format.
    Tries multiple common date formats if the direct conversion fails.
    Logs any rows that cannot be converted.
    """
    if date_col not in df.columns:
        print(f"Warning: '{date_col}' column not found in DataFrame.")
        return df

    # Try to convert with pandas default
    try:
        df[date_col] = pd.to_datetime(
            df[date_col], errors="coerce", infer_datetime_format=True
        )
    except Exception as e:
        print(f"Error converting {date_col} to datetime: {e}")

    # Count invalid dates
    invalid_dates = df[date_col].isna().sum()
    if invalid_dates > 0:
        print(
            f"Warning: {invalid_dates} rows have invalid or missing {date_col} and will be dropped."
        )
        df = df.dropna(subset=[date_col])

    return df


def remove_specific_clients(df, client_col="Client"):
    """
    Remove rows where Client is 'L. Pfeffer' or 'M. Vincent'.
    Logs how many rows were removed.
    """
    if client_col not in df.columns:
        print(f"Warning: '{client_col}' column not found in DataFrame.")
        return df

    initial_count = df.shape[0]
    clients_to_remove = ["L. Pfeffer", "M. Vincent"]
    df = df[~df[client_col].isin(clients_to_remove)]
    final_count = df.shape[0]
    removed = initial_count - final_count

    if removed > 0:
        print(f"Removed {removed} rows for clients: {', '.join(clients_to_remove)}")

    return df


def remove_old_orders(df, date_col="Date", cutoff_date="2021-03-31"):
    """
    Remove orders before a specified cutoff date.
    Args:
        df: DataFrame containing order data
        date_col: Name of the date column
        cutoff_date: Date string in YYYY-MM-DD format - orders before this will be removed
    Returns:
        DataFrame with old orders removed
    """
    if date_col not in df.columns:
        print(f"Warning: '{date_col}' column not found in DataFrame.")
        return df

    # Convert cutoff_date to datetime
    cutoff = pd.to_datetime(cutoff_date)

    # Get initial count
    initial_count = len(df)

    # Filter to keep only rows >= cutoff_date
    df = df[df[date_col] >= cutoff]

    # Calculate and log removed rows
    removed = initial_count - len(df)
    print(f"Removed {removed} orders from before {cutoff_date}")

    return df


def remove_abandoned_carts(df, id_col="ID commande", total_col="Total"):
    """
    Remove rows from cart DataFrame where 'ID commande' is 'Panier abandonné' and 'Total' is 0.
    Logs how many rows were removed.
    """
    if id_col not in df.columns or total_col not in df.columns:
        print(
            "Warning: Cannot remove abandoned carts since required columns are missing."
        )
        return df

    condition = (df[id_col] == "Panier abandonné") & (df[total_col] == 0)
    to_remove = df[condition].shape[0]
    df = df[~condition]
    print(f"Removed {to_remove} rows of abandoned carts with zero total.")
    return df


def handle_missing_values(df, required_columns):
    """
    Drop rows where required columns are missing.
    Logs how many rows are dropped.
    """
    initial_count = df.shape[0]
    for col in required_columns:
        if col not in df.columns:
            print(f"Warning: Required column '{col}' not found in DataFrame.")
    df = df.dropna(subset=required_columns)
    final_count = df.shape[0]
    dropped = initial_count - final_count
    if dropped > 0:
        print(
            f"Dropped {dropped} rows due to missing required columns: {required_columns}"
        )
    return df


def remove_duplicates(df, subset_cols):
    """
    Remove duplicate rows based on a subset of columns.
    Logs how many duplicates were removed.
    """
    initial_count = df.shape[0]
    df = df.drop_duplicates(subset=subset_cols)
    final_count = df.shape[0]
    removed = initial_count - final_count
    if removed > 0:
        print(f"Removed {removed} duplicate rows based on columns {subset_cols}.")
    return df


def load_data(file_path):
    """
    Load CSV data from a specified file path.
    Handles various common errors.
    """
    try:
        df = pd.read_csv(file_path, encoding="utf-8", low_memory=False)
        return df
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except pd.errors.EmptyDataError:
        print(f"No data: {file_path} is empty.")
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
    return None


def save_data(df, file_path):
    """
    Save a dataframe to a CSV file at the specified file path.
    """
    try:
        df.to_csv(file_path, index=False)
        print(f"Data saved to: {file_path}")
    except Exception as e:
        print(f"Error saving file {file_path}: {e}")


def preprocess_data():
    """
    Main preprocessing function:
    1. Load cart and order data from the raw directory.
    2. Standardize column names.
    3. Clean the Total column and convert it to float.
    4. Remove abandoned carts where Total = 0.
    5. Convert Date column to datetime.
    6. Handle missing required values.
    7. Remove duplicates if needed.
    8. Save cleaned data to the cleaned directory.
    """
    cart_path = os.path.join(PROCESSED_DATA_DIR, CART_FILENAME)
    order_path = os.path.join(PROCESSED_DATA_DIR, ORDER_FILENAME)

    cart_df = load_data(cart_path)
    order_df = load_data(order_path)

    if cart_df is None or order_df is None:
        print("Data loading failed. Cannot proceed with preprocessing.")
        return

    # Standardize columns
    cart_df = standardize_column_names(cart_df)
    order_df = standardize_column_names(order_df)

    # Clean Total columns
    cart_df = clean_total_column(cart_df, total_col="Total")
    order_df = clean_total_column(order_df, total_col="Total")

    # Remove abandoned carts
    cart_df = remove_abandoned_carts(cart_df, id_col="ID commande", total_col="Total")

    # Convert dates
    cart_df = convert_date_column(cart_df, date_col="Date")
    order_df = convert_date_column(order_df, date_col="Date")

    # Remove specific clients
    cart_df = remove_specific_clients(cart_df, client_col="Client")
    order_df = remove_specific_clients(order_df, client_col="Client")

    # Remove old orders
    order_df = remove_old_orders(order_df, date_col="Date", cutoff_date="2021-03-31")

    # Handle missing values - assume these columns must be present
    required_cart_cols = ["ID commande", "Total", "Date"]
    required_order_cols = [
        "id",
        "Total",
        "Date",
    ]  # Adjust as needed based on your data schema

    cart_df = handle_missing_values(cart_df, required_cart_cols)
    order_df = handle_missing_values(order_df, required_order_cols)

    # Remove duplicates if it makes sense (e.g., by 'ID commande')
    if "ID commande" in cart_df.columns:
        cart_df = remove_duplicates(cart_df, subset_cols=["ID commande", "Date"])
    if "id" in order_df.columns:
        order_df = remove_duplicates(order_df, subset_cols=["id", "Date"])

    # Save cleaned data
    try:
        cleaned_cart_path = os.path.join(CLEANED_DATA_DIR, CART_FILENAME)
        cleaned_order_path = os.path.join(CLEANED_DATA_DIR, ORDER_FILENAME)

        save_data(cart_df, cleaned_cart_path)
        save_data(order_df, cleaned_order_path)
    except Exception as e:
        print(f"Error saving cleaned data: {e}")
        return None


if __name__ == "__main__":
    preprocess_data()
