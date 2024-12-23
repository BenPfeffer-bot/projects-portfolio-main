import os
import pandas as pd
from typing import Optional


def load_csv(file_path: str, delimiter: str = ";") -> Optional[pd.DataFrame]:
    """
    Load a CSV file with a given delimiter.
    Returns a DataFrame or None if an error occurs.
    """
    try:
        df = pd.read_csv(file_path, delimiter=delimiter)
        return df
    except Exception as e:
        print(f"[ERROR] Could not load file {file_path}: {e}")
        return None


def load_data(file_path: str) -> Optional[pd.DataFrame]:
    """
    Load data with proper encoding for French characters
    """
    try:
        # Try UTF-8 first
        df = pd.read_csv(file_path, encoding="utf-8")
    except UnicodeDecodeError:
        try:
            # Fall back to Latin-1 encoding
            df = pd.read_csv(file_path, encoding="latin-1")
        except Exception as e:
            print(f"[ERROR] Could not load file {file_path}: {e}")
            return None

    print(f"[INFO] Successfully loaded {file_path} with {len(df)} rows")
    return df
