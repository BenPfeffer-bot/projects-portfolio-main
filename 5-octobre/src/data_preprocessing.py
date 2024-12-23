import os
import pandas as pd
import numpy as np
from datetime import datetime
import sys
from pydantic import BaseModel, Field, validator
from typing import Optional, List
from decimal import Decimal

sys.path.append(
    "/Users/benpfeffer/Library/Mobile Documents/com~apple~CloudDocs/projects-portfolio-main/5-octobre"
)
from src.config import (
    PROCESSED_DATA_DIR,
    CLEANED_DATA_DIR,
    CART_FILENAME,
    ORDER_FILENAME,
    INVENTORY_FILENAME,
    RETAIL_FILENAME,
)


# Data Models for Validation
class OrderData(BaseModel):
    id: int
    reference: str = Field(alias="Référence")
    new_customer: int = Field(alias="Nouveau client")
    delivery: str = Field(alias="Livraison")
    client: Optional[str] = Field(alias="Client")
    total: Decimal = Field(alias="Total")
    payment: str = Field(alias="Paiement")
    status: str = Field(alias="État")
    date: datetime = Field(alias="Date")

    @validator("total")
    def validate_total(cls, v):
        if v < 0:
            raise ValueError("Total amount cannot be negative")
        return v

    @validator("date")
    def validate_date(cls, v):
        if v > datetime.now():
            raise ValueError("Date cannot be in the future")
        return v


class CartData(BaseModel):
    id: int
    order_id: str = Field(alias="ID commande")
    client: Optional[str] = Field(alias="Client")
    total: Decimal = Field(alias="Total")
    carrier: Optional[str] = Field(alias="Transporteur")
    date: datetime = Field(alias="Date")

    @validator("total")
    def validate_total(cls, v):
        if v < 0:
            raise ValueError("Total amount cannot be negative")
        return v

    @validator("date")
    def validate_date(cls, v):
        if v > datetime.now():
            raise ValueError("Date cannot be in the future")
        return v


class InventoryData(BaseModel):
    id: int
    sfa: str
    lib: str
    ean: str
    qty: int
    factory_price: Decimal
    retail: Decimal
    retail_us: Decimal

    @validator("qty")
    def validate_qty(cls, v):
        if not isinstance(v, int):
            raise ValueError("Quantity must be an integer")
        return v

    @validator("factory_price", "retail", "retail_us")
    def validate_prices(cls, v):
        if v < 0:
            raise ValueError("Price cannot be negative")
        return v


class RetailData(BaseModel):
    date: datetime = Field(alias="Date")
    ref: str = Field(alias="Ref")
    libelle: str = Field(alias="Libellé")
    customer: str = Field(alias="Cust")
    quantity: int = Field(alias="Qté")
    pv_ttc: Decimal = Field(alias="PV TTC")
    ca_ttc: Decimal = Field(alias="CA TTC")

    @validator("date")
    def validate_date(cls, v):
        if v > datetime.now():
            raise ValueError("Date cannot be in the future")
        return v

    @validator("quantity")
    def validate_quantity(cls, v):
        if v <= 0:
            raise ValueError("Quantity must be positive")
        return v

    @validator("pv_ttc", "ca_ttc")
    def validate_amounts(cls, v):
        if v < 0:
            raise ValueError("Amount cannot be negative")
        return v


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
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
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
    Load data from CSV file with proper error handling
    """
    try:
        if "savs.csv" in file_path:  # For retail data
            df = pd.read_csv(file_path)
        else:
            df = pd.read_csv(file_path, delimiter=";")
        return df
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return None


def save_data(df, file_path):
    """
    Save DataFrame to CSV with proper error handling
    """
    try:
        df.to_csv(file_path, index=False)
        print(f"Successfully saved data to {file_path}")
    except Exception as e:
        print(f"Error saving data to {file_path}: {e}")


def validate_data_schema(df, model_class, raise_errors=True):
    """
    Validate DataFrame against a Pydantic model schema.
    Returns tuple (is_valid, error_records, error_messages)
    """
    errors = []
    error_records = []

    for idx, row in df.iterrows():
        try:
            model_class(**row.to_dict())
        except Exception as e:
            errors.append(f"Row {idx}: {str(e)}")
            error_records.append(idx)
            if raise_errors:
                raise ValueError(f"Data validation error at row {idx}: {str(e)}")

    return len(errors) == 0, error_records, errors


def check_data_quality(df, required_cols=None, numeric_cols=None, date_cols=None):
    """
    Perform data quality checks on DataFrame
    Returns tuple (quality_score, issues)
    """
    issues = []
    checks_passed = 0
    total_checks = 0

    # Check for required columns
    if required_cols:
        total_checks += 1
        missing_cols = [col for col in required_cols if col not in df.columns]
        if not missing_cols:
            checks_passed += 1
        else:
            issues.append(f"Missing required columns: {missing_cols}")

    # Check for duplicate rows
    total_checks += 1
    duplicates = df.duplicated().sum()
    if duplicates == 0:
        checks_passed += 1
    else:
        issues.append(f"Found {duplicates} duplicate rows")

    # Check numeric columns
    if numeric_cols:
        for col in numeric_cols:
            if col in df.columns:
                total_checks += 2
                # Check for negative values
                neg_values = (df[col] < 0).sum()
                if neg_values == 0:
                    checks_passed += 1
                else:
                    issues.append(f"Found {neg_values} negative values in {col}")

                # Check for outliers (using IQR method)
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = (
                    (df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))
                ).sum()
                if outliers == 0:
                    checks_passed += 1
                else:
                    issues.append(f"Found {outliers} potential outliers in {col}")

    # Check date columns
    if date_cols:
        for col in date_cols:
            if col in df.columns:
                total_checks += 2
                # Convert dates if they're not already datetime
                try:
                    dates = pd.to_datetime(df[col], errors="coerce")
                    # Check for future dates
                    future_dates = (dates > pd.Timestamp.now()).sum()
                    if future_dates == 0:
                        checks_passed += 1
                    else:
                        issues.append(f"Found {future_dates} future dates in {col}")

                    # Check for very old dates (before 2020)
                    old_dates = (dates < pd.Timestamp("2020-01-01")).sum()
                    if old_dates == 0:
                        checks_passed += 1
                    else:
                        issues.append(f"Found {old_dates} dates before 2020 in {col}")
                except Exception as e:
                    issues.append(f"Error processing dates in column {col}: {str(e)}")
                    total_checks -= (
                        2  # Subtract these checks since they couldn't be performed
                    )

    quality_score = checks_passed / total_checks if total_checks > 0 else 0
    return quality_score, issues


def preprocess_retail_data(df):
    """
    Preprocess retail dataset by cleaning currency values and dates
    """
    # Convert currency columns
    currency_cols = ["PV TTC", "CA TTC"]
    for col in currency_cols:
        # Remove currency symbols and extract numeric values
        df[col] = df[col].str.replace("€", "").str.replace("$", "")
        df[col] = df[col].apply(lambda x: x.split("(")[0] if "(" in str(x) else x)
        # Convert to float, replacing commas with periods
        df[col] = df[col].str.replace(",", ".").astype(float)

    # Convert Date column to datetime
    df["Date"] = pd.to_datetime(df["Date"], format="%d %b %Y")

    # Remove duplicates
    df.drop_duplicates(inplace=True)

    return df


def preprocess_inventory_data(df):
    """
    Preprocess inventory dataset by cleaning currency values and standardizing column names
    """
    # Clean Factory Price column
    df["Factory Price"] = df["Factory Price"].str.replace(",", ".").astype(float)

    # Normalize column names to snake_case
    df.columns = df.columns.str.lower().str.replace(" ", "_")

    # Drop unnecessary columns
    if "group_price" in df.columns:
        df = df.drop(columns=["group_price"])

    # Remove duplicates
    df.drop_duplicates(inplace=True)

    return df


def preprocess_data():
    """
    Main preprocessing function with enhanced validation and quality checks
    """
    # Load all data files
    cart_path = os.path.join(PROCESSED_DATA_DIR, CART_FILENAME)
    order_path = os.path.join(PROCESSED_DATA_DIR, ORDER_FILENAME)
    inventory_path = os.path.join(PROCESSED_DATA_DIR, INVENTORY_FILENAME)
    retail_path = os.path.join(PROCESSED_DATA_DIR, RETAIL_FILENAME)

    cart_df = load_data(cart_path)
    order_df = load_data(order_path)
    inventory_df = load_data(inventory_path)
    retail_df = load_data(retail_path)

    if any(df is None for df in [cart_df, order_df, inventory_df, retail_df]):
        print("Data loading failed. Cannot proceed with preprocessing.")
        return

    # Process retail data
    retail_df = standardize_column_names(retail_df)
    retail_df = preprocess_retail_data(retail_df)
    retail_df = handle_missing_values(retail_df, ["Date", "PV TTC", "CA TTC"])
    retail_df = remove_duplicates(retail_df)

    # Process inventory data
    inventory_df = standardize_column_names(inventory_df)
    inventory_df = preprocess_inventory_data(inventory_df)
    inventory_df = handle_missing_values(
        inventory_df, ["id", "factory_price", "retail"]
    )
    inventory_df = remove_duplicates(inventory_df)

    # Process cart and order data
    cart_df = remove_abandoned_carts(cart_df, id_col="ID commande", total_col="Total")
    cart_df = convert_date_column(cart_df, date_col="Date")
    order_df = convert_date_column(order_df, date_col="Date")
    cart_df = remove_specific_clients(cart_df, client_col="Client")
    order_df = remove_specific_clients(order_df, client_col="Client")
    order_df = remove_old_orders(order_df, date_col="Date", cutoff_date="2021-03-31")

    # Handle missing values
    cart_df = handle_missing_values(cart_df, ["ID commande", "Total", "Date"])
    order_df = handle_missing_values(order_df, ["id", "Total", "Date"])

    # Remove duplicates
    cart_df = remove_duplicates(cart_df, subset_cols=["ID commande", "Date"])
    order_df = remove_duplicates(order_df, subset_cols=["id", "Date"])

    try:
        # Save all cleaned datasets
        cleaned_cart_path = os.path.join(CLEANED_DATA_DIR, CART_FILENAME)
        cleaned_order_path = os.path.join(CLEANED_DATA_DIR, ORDER_FILENAME)
        cleaned_inventory_path = os.path.join(CLEANED_DATA_DIR, INVENTORY_FILENAME)
        cleaned_retail_path = os.path.join(CLEANED_DATA_DIR, RETAIL_FILENAME)

        save_data(cart_df, cleaned_cart_path)
        save_data(order_df, cleaned_order_path)
        save_data(inventory_df, cleaned_inventory_path)
        save_data(retail_df, cleaned_retail_path)

        return cart_df, order_df, inventory_df, retail_df
    except Exception as e:
        print(f"Error saving cleaned data: {e}")
        return None


if __name__ == "__main__":
    preprocess_data()
