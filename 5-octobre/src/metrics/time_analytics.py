"""
Time-based analytics and metrics.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any
from .utils import vectorized_operation, cache_result, with_progress_bar, parallelize_dataframe, batch_process

logger = logging.getLogger(__name__)


@vectorized_operation
@cache_result(expire_after=3600)
@with_progress_bar("Computing Revenue Over Time")
def compute_revenue_over_time(order_df: pd.DataFrame, freq: str = "M", date_col: str = "Date", total_col: str = "Total") -> pd.Series:
    """
    Compute total revenue aggregated by time frequency using vectorized operations.
    """
    logger.info(f"Computing revenue over time with frequency {freq}...")
    if date_col not in order_df.columns or total_col not in order_df.columns:
        logger.error("Required columns for revenue over time not found")
        return pd.Series(dtype=float)
    try:
        # Vectorized date and aggregation operations
        order_df[date_col] = pd.to_datetime(order_df[date_col], errors="coerce")
        revenue_series = order_df.set_index(date_col).resample(freq)[total_col].sum()
        logger.debug(f"Revenue series shape: {revenue_series.shape}")
        return revenue_series
    except Exception as e:
        logger.error(f"Error computing revenue over time: {e}")
        return pd.Series(dtype=float)


@vectorized_operation
@cache_result(expire_after=3600)
def revenue_growth(order_df: pd.DataFrame, freq: str = "M", date_col: str = "Date", total_col: str = "Total") -> pd.Series:
    """
    Compute period-over-period revenue growth using vectorized operations.
    """
    logger.info(f"Computing revenue growth with frequency {freq}...")
    revenue_series = compute_revenue_over_time(order_df, freq=freq, date_col=date_col, total_col=total_col)
    if revenue_series.empty:
        logger.warning("Empty revenue series - cannot compute growth")
        return pd.Series(dtype=float)

    # Vectorized growth calculation
    growth = revenue_series.pct_change() * 100
    logger.debug(f"Revenue growth series shape: {growth.shape}")
    return growth


@vectorized_operation
@cache_result(expire_after=3600)
@with_progress_bar("Analyzing Customer Count")
def analyze_customer_count(order_df: pd.DataFrame, date_col: str = "Date", client_col: str = "Client", freq: str = "M") -> pd.Series:
    """
    Analyze unique customers over time using vectorized operations.
    """
    logger.info(f"Analyzing customer count with frequency {freq}...")
    if date_col not in order_df.columns or client_col not in order_df.columns:
        logger.error("Required columns for customer analysis not found")
        return pd.Series(dtype=float)

    try:
        # Vectorized date and customer counting
        order_df[date_col] = pd.to_datetime(order_df[date_col], errors="coerce")
        customer_series = order_df.set_index(date_col).groupby(pd.Grouper(freq=freq))[client_col].nunique()
        logger.debug(f"Customer count series shape: {customer_series.shape}")
        return customer_series
    except Exception as e:
        logger.error(f"Error analyzing customer count: {e}")
        return pd.Series(dtype=float)


@vectorized_operation
@cache_result(expire_after=3600)
@with_progress_bar("Performing Cohort Analysis")
def cohort_analysis(order_df: pd.DataFrame, client_col: str = "Client", date_col: str = "Date", freq: str = "M") -> pd.DataFrame:
    """
    Analyze retention using vectorized cohort operations.
    """
    if client_col not in order_df.columns or date_col not in order_df.columns:
        logger.error("Required columns for cohort analysis not found")
        return pd.DataFrame()

    try:
        # Vectorized date operations
        order_df[date_col] = pd.to_datetime(order_df[date_col])

        # Vectorized cohort calculations
        order_df["CohortMonth"] = order_df.groupby(client_col)[date_col].transform("min").dt.to_period(freq)
        order_df["OrderMonth"] = order_df[date_col].dt.to_period(freq)

        # Vectorized pivot and retention calculation
        cohort_data = order_df.groupby(["CohortMonth", "OrderMonth"])[client_col].nunique().reset_index()
        cohort_data["CohortIndex"] = (cohort_data["OrderMonth"] - cohort_data["CohortMonth"]).apply(lambda x: x.n)

        cohort_pivot = cohort_data.pivot_table(index="CohortMonth", columns="CohortIndex", values=client_col)

        # Vectorized retention calculation
        cohort_size = cohort_pivot.iloc[:, 0]
        retention = cohort_pivot.divide(cohort_size, axis=0) * 100
        return retention
    except Exception as e:
        logger.error(f"Error in cohort analysis: {e}")
        return pd.DataFrame()


@vectorized_operation
@cache_result(expire_after=3600)
def day_of_week_analysis(order_df: pd.DataFrame, date_col: str = "Date", total_col: str = "Total") -> pd.Series:
    """
    Analyze revenue by day of the week using vectorized operations.
    """
    if date_col not in order_df.columns or total_col not in order_df.columns:
        logger.error("Required columns for day_of_week analysis not found")
        return pd.Series(dtype=float)
    try:
        # Vectorized date and aggregation operations
        order_df[date_col] = pd.to_datetime(order_df[date_col])
        return order_df.assign(DayOfWeek=order_df[date_col].dt.day_name()).groupby("DayOfWeek")[total_col].sum().sort_values(ascending=False)
    except Exception as e:
        logger.error(f"Error in day_of_week analysis: {e}")
        return pd.Series(dtype=float)


@vectorized_operation
@cache_result(expire_after=3600)
def hour_of_day_analysis(order_df: pd.DataFrame, date_col: str = "Date", total_col: str = "Total") -> pd.Series:
    """
    Analyze revenue by hour using vectorized operations.
    """
    if date_col not in order_df.columns or total_col not in order_df.columns:
        logger.error("Required columns for hour_of_day analysis not found")
        return pd.Series(dtype=float)
    try:
        # Vectorized hour extraction and aggregation
        return order_df.assign(Hour=pd.to_datetime(order_df[date_col]).dt.hour).groupby("Hour")[total_col].sum().sort_values(ascending=False)
    except Exception as e:
        logger.error(f"Error in hour_of_day analysis: {e}")
        return pd.Series(dtype=float)


@vectorized_operation
@cache_result(expire_after=3600)
@with_progress_bar("Computing Year-over-Year Metrics")
def year_over_year_metrics(order_df: pd.DataFrame, date_col: str = "Date", total_col: str = "Total") -> Dict[str, pd.DataFrame]:
    """
    Compute year-over-year metrics using vectorized operations.
    """
    if date_col not in order_df.columns or total_col not in order_df.columns:
        logger.error("Required columns for year-over-year analysis not found")
        return {}

    try:
        # Vectorized date conversion
        order_df[date_col] = pd.to_datetime(order_df[date_col], errors="coerce")

        # Vectorized revenue analysis
        yearly_revenue = order_df.groupby(pd.Grouper(key=date_col, freq="Y")).agg({total_col: "sum"}).reset_index()
        yearly_revenue["year"] = yearly_revenue[date_col].dt.year
        yearly_revenue = yearly_revenue.rename(columns={total_col: "total_revenue"})
        yearly_revenue["yoy_growth"] = yearly_revenue["total_revenue"].pct_change() * 100

        # Vectorized orders analysis
        yearly_orders = order_df.set_index(date_col).resample("Y").size().reset_index(name="total_orders")
        yearly_orders["year"] = yearly_orders[date_col].dt.year
        yearly_orders["yoy_growth"] = yearly_orders["total_orders"].pct_change() * 100

        # Vectorized AOV analysis
        yearly_aov = yearly_revenue.copy()
        yearly_aov["yearly_aov"] = yearly_revenue["total_revenue"] / yearly_orders["total_orders"]
        yearly_aov["yoy_growth"] = yearly_aov["yearly_aov"].pct_change() * 100
        yearly_aov = yearly_aov[["year", "yearly_aov", "yoy_growth"]]

        return {"revenue": yearly_revenue[["year", "total_revenue", "yoy_growth"]], "orders": yearly_orders[["year", "total_orders", "yoy_growth"]], "aov": yearly_aov}
    except Exception as e:
        logger.error(f"Error in year_over_year analysis: {e}")
        return {}
