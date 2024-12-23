import pandas as pd
from typing import List, Tuple
from pydantic import BaseModel


def validate_schema(
    df: pd.DataFrame, model_class: BaseModel, raise_errors: bool = False
) -> Tuple[bool, List[int], List[str]]:
    """
    Validate each row of a DataFrame against a Pydantic model.
    Returns (is_valid, error_indices, error_messages).
    If raise_errors=True, stops on the first invalid row.
    """
    errors = []
    error_indices = []
    for idx, row in df.iterrows():
        try:
            model_class(**row.to_dict())
        except Exception as e:
            errors.append(str(e))
            error_indices.append(idx)
            if raise_errors:
                raise ValueError(f"Validation failed on row {idx}: {e}")

    return (len(errors) == 0, error_indices, errors)


def check_duplicates(df: pd.DataFrame, subset: List[str]) -> int:
    """
    Returns the number of duplicate rows based on a subset of columns.
    """
    return df.duplicated(subset=subset).sum()


def check_negative_values(df: pd.DataFrame, col: str) -> int:
    """
    Returns the count of negative values in a numeric column.
    """
    if col not in df.columns:
        return 0
    return (df[col] < 0).sum()
