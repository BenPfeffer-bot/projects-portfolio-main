"""
Utility functions and decorators for metrics optimization.
"""

import functools
import time
from typing import Any, Callable, Dict, Optional
import pandas as pd
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
from functools import wraps

logger = logging.getLogger(__name__)

# Cache for storing frequently accessed data
_CACHE: Dict[str, Any] = {}


def cache_dataframe(key: str, df: pd.DataFrame, max_age_seconds: int = 3600) -> None:
    """Cache a DataFrame with timestamp."""
    _CACHE[key] = {"data": df, "timestamp": time.time(), "max_age": max_age_seconds}


def get_cached_dataframe(key: str) -> Optional[pd.DataFrame]:
    """Retrieve cached DataFrame if it exists and hasn't expired."""
    if key in _CACHE:
        cache_entry = _CACHE[key]
        if time.time() - cache_entry["timestamp"] < cache_entry["max_age"]:
            return cache_entry["data"]
        else:
            del _CACHE[key]
    return None


def clear_cache() -> None:
    """Clear all cached data."""
    _CACHE.clear()


def with_progress_bar(desc: str = "Processing"):
    """Decorator to add a progress bar to a function."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            with tqdm(total=1, desc=desc) as pbar:
                result = func(*args, **kwargs)
                pbar.update(1)
                return result

        return wrapper

    return decorator


def parallelize_dataframe(df: pd.DataFrame, func: Callable, n_cores: int = 4) -> pd.DataFrame:
    """
    Parallelize operations on DataFrame by splitting it into chunks.

    Args:
        df: Input DataFrame
        func: Function to apply to each chunk
        n_cores: Number of CPU cores to use

    Returns:
        Processed DataFrame with results combined
    """
    df_split = np.array_split(df, n_cores)

    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        futures = [executor.submit(func, chunk) for chunk in df_split]
        results = [future.result() for future in as_completed(futures)]

    return pd.concat(results)


def vectorized_operation(func: Callable) -> Callable:
    """
    Decorator to ensure operations are vectorized when possible.
    Will log a warning if operation seems to be using iterative approach.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time

        # Check if the function took longer than expected
        if execution_time > 1.0:  # threshold in seconds
            logger.warning(f"Function {func.__name__} took {execution_time:.2f} seconds. " "Consider vectorizing operations for better performance.")
        return result

    return wrapper


def cache_result(expire_after: int = 3600):
    """
    Cache function results with expiration.

    Args:
        expire_after: Cache expiration time in seconds
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create a cache key from function name and arguments
            cache_key = f"{func.__name__}:{str(args)}:{str(kwargs)}"
            cached_result = get_cached_dataframe(cache_key)

            if cached_result is not None:
                logger.debug(f"Cache hit for {func.__name__}")
                return cached_result

            result = func(*args, **kwargs)
            cache_dataframe(cache_key, result, expire_after)
            return result

        return wrapper

    return decorator


def batch_process(batch_size: int = 1000):
    """
    Decorator to process large DataFrames in batches.

    Args:
        batch_size: Number of rows to process in each batch
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(df: pd.DataFrame, *args, **kwargs):
            if len(df) <= batch_size:
                return func(df, *args, **kwargs)

            results = []
            with tqdm(total=len(df), desc=f"Processing {func.__name__}") as pbar:
                for start_idx in range(0, len(df), batch_size):
                    end_idx = min(start_idx + batch_size, len(df))
                    batch_df = df.iloc[start_idx:end_idx]
                    batch_result = func(batch_df, *args, **kwargs)
                    results.append(batch_result)
                    pbar.update(len(batch_df))

            # Combine results based on return type
            if isinstance(results[0], pd.DataFrame):
                return pd.concat(results, ignore_index=True)
            elif isinstance(results[0], dict):
                combined = {}
                for r in results:
                    for k, v in r.items():
                        if k not in combined:
                            combined[k] = v
                        else:
                            if isinstance(v, (int, float)):
                                combined[k] += v
                            elif isinstance(v, pd.Series):
                                combined[k] = combined[k].add(v, fill_value=0)
                return combined
            else:
                return pd.concat(results)

        return wrapper

    return decorator
