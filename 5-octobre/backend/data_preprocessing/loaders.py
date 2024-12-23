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
    A more specialized loader that detects the file name and chooses
    an appropriate delimiter or logic. For example, we might handle
    'retail.csv' differently than the others.
    """
    if "retail.csv" in os.path.basename(file_path).lower():
        return load_csv(file_path, delimiter=",")
    else:
        return load_csv(file_path, delimiter=";")
