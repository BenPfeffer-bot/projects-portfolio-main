# my_project/backend/data_preprocessing/cleaners.py

import pandas as pd
import numpy as np
from datetime import datetime
from typing import List


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Trim whitespace and ensure consistent formatting for column names.
    """
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
    return df


def remove_abandoned_carts(
    df: pd.DataFrame, id_col: str, total_col: str
) -> pd.DataFrame:
    """
    Remove rows that represent abandoned carts (assuming some pattern, e.g. 'Panier abandonné' with zero total).
    """
    if id_col not in df.columns or total_col not in df.columns:
        print("[WARN] Required columns for removing abandoned carts are missing.")
        return df

    condition = (df[id_col] == "Panier abandonné") & (df[total_col] == 0)
    removed_count = condition.sum()
    df = df[~condition]
    print(f"[INFO] Removed {removed_count} abandoned cart rows.")
    return df


def convert_to_datetime(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """
    Convert a column to datetime. Invalid rows are dropped.
    """
    if date_col not in df.columns:
        print(f"[WARN] Date column '{date_col}' not found.")
        return df

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    invalid_count = df[date_col].isna().sum()

    if invalid_count > 0:
        df.dropna(subset=[date_col], inplace=True)
        print(
            f"[INFO] Dropped {invalid_count} invalid date rows in column '{date_col}'."
        )

    return df


def remove_rows_before_date(
    df: pd.DataFrame, date_col: str, cutoff: str = "2021-03-31"
) -> pd.DataFrame:
    """
    Remove rows older than a certain date.
    """
    if date_col not in df.columns:
        print(f"[WARN] Date column '{date_col}' not found.")
        return df

    cutoff_date = pd.to_datetime(cutoff)
    before_count = df.shape[0]
    df = df[df[date_col] >= cutoff_date]
    after_count = df.shape[0]
    removed = before_count - after_count
    print(f"[INFO] Removed {removed} rows before {cutoff_date}.")
    return df


def remove_specific_clients(
    df: pd.DataFrame, client_col: str, unwanted_list: List[str]
) -> pd.DataFrame:
    """
    Remove rows with certain client names (e.g. test clients).
    """
    if client_col not in df.columns:
        print(f"[WARN] Client column '{client_col}' not found.")
        return df

    before_count = df.shape[0]
    df = df[~df[client_col].isin(unwanted_list)]
    after_count = df.shape[0]
    removed = before_count - after_count
    if removed > 0:
        print(f"[INFO] Removed {removed} rows for clients in {unwanted_list}.")
    return df


def remove_missing_required(df: pd.DataFrame, required_cols: List[str]) -> pd.DataFrame:
    """
    Drop rows where required columns are null.
    """
    before_count = df.shape[0]
    df.dropna(subset=required_cols, inplace=True)
    after_count = df.shape[0]
    removed = before_count - after_count
    if removed > 0:
        print(
            f"[INFO] Dropped {removed} rows with missing required columns {required_cols}."
        )
    return df


def remove_duplicates(df: pd.DataFrame, subset: List[str]) -> pd.DataFrame:
    """
    Remove duplicates based on a subset of columns.
    """
    before_count = df.shape[0]
    df.drop_duplicates(subset=subset, inplace=True)
    after_count = df.shape[0]
    removed = before_count - after_count
    if removed > 0:
        print(f"[INFO] Removed {removed} duplicate rows based on {subset}.")
    return df


def clean_currency_column(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Remove currency symbols, convert commas to dots, and cast to float.
    """
    if col not in df.columns:
        print(f"[WARN] Column '{col}' not found.")
        return df

    def parse_currency(val):
        if pd.isna(val):
            return np.nan
        # Remove currency symbols
        val = (
            str(val)
            .replace("€", "")
            .replace("$", "")
            .replace("£", "")
            .replace("\xa0", "")
            .replace(" ", "")
        )
        val = val.replace(",", ".")
        try:
            return float(val)
        except ValueError:
            return np.nan

    nonnull_before = df[col].notnull().sum()
    df[col] = df[col].apply(parse_currency)
    nonnull_after = df[col].notnull().sum()
    print(
        f"[INFO] Cleaned '{col}' from currency format. Valid non-null: {nonnull_after}/{nonnull_before}."
    )
    return df
