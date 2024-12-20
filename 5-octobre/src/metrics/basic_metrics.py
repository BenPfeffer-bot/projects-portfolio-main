"""
Basic metrics and KPIs for e-commerce analysis.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any
from .utils import vectorized_operation, cache_result, with_progress_bar, parallelize_dataframe, batch_process

logger = logging.getLogger(__name__)


@vectorized_operation
@cache_result(expire_after=3600)
@with_progress_bar("Computing Basic KPIs")
def basic_kpis(order_df: pd.DataFrame, total_col: str = "Total", client_col: str = "Client") -> Dict[str, Any]:
    """
    Compute basic KPIs using vectorized operations:
    - Total orders
    - Total revenue
    - Unique customers
    - Average orders per customer
    """
    logger.info("Computing basic KPIs...")
    if total_col not in order_df.columns or client_col not in order_df.columns:
        logger.error("Required columns for basic KPIs not found")
        return {}

    try:
        # Vectorized calculations
        total_orders = len(order_df)
        total_revenue = order_df[total_col].sum()
        unique_customers = order_df[client_col].nunique()
        avg_orders_per_customer = total_orders / unique_customers if unique_customers > 0 else 0

        kpis = {
            "total_orders": total_orders,
            "total_revenue": total_revenue,
            "unique_customers": unique_customers,
            "avg_orders_per_customer": avg_orders_per_customer,
        }

        logger.debug(f"Basic KPIs calculated: {kpis}")
        return kpis
    except Exception as e:
        logger.error(f"Error computing basic KPIs: {e}")
        return {}


@vectorized_operation
@cache_result(expire_after=3600)
def compute_average_order_value(order_df: pd.DataFrame, total_col: str = "Total") -> float:
    """
    Compute the Average Order Value (AOV) using vectorized operations.
    """
    logger.info("Computing average order value...")
    if total_col not in order_df.columns:
        logger.error(f"Column {total_col} not found in orders data")
        return np.nan
    try:
        # Vectorized mean calculation
        aov = order_df[total_col].mean()
        logger.debug(f"Calculated AOV: {aov}")
        return aov
    except Exception as e:
        logger.error(f"Error computing average order value: {e}")
        return np.nan


@vectorized_operation
@cache_result(expire_after=3600)
def order_value_distribution(order_df: pd.DataFrame, total_col: str = "Total") -> Dict[str, float]:
    """
    Provide descriptive statistics for order values using vectorized operations.
    """
    logger.info("Analyzing order value distribution...")
    if total_col not in order_df.columns:
        logger.error(f"{total_col} not found in orders data")
        return {}
    try:
        # Vectorized descriptive statistics
        desc = order_df[total_col].describe()
        stats = {
            "min": desc["min"],
            "max": desc["max"],
            "median": desc["50%"],
            "mean": desc["mean"],
            "std": desc["std"],
            "25%_quartile": desc["25%"],
            "75%_quartile": desc["75%"],
        }
        logger.debug(f"Order value distribution stats: {stats}")
        return stats
    except Exception as e:
        logger.error(f"Error computing order value distribution: {e}")
        return {}


@vectorized_operation
@cache_result(expire_after=3600)
@with_progress_bar("Computing Cart Abandonment Rate")
def compute_cart_abandonment_rate(cart_df: pd.DataFrame, order_df: pd.DataFrame, cart_id_col: str = "ID commande", order_ref_col: str = "Référence") -> float:
    """
    Compute cart abandonment rate using vectorized operations.
    """
    logger.info("Computing cart abandonment rate...")
    if cart_id_col not in cart_df.columns or order_ref_col not in order_df.columns:
        logger.error("Required columns for cart abandonment rate not found")
        return np.nan

    try:
        # Vectorized set operations
        total_carts = cart_df[cart_id_col].nunique()
        completed_orders = cart_df[cart_id_col].isin(order_df[order_ref_col].unique()).sum()

        if total_carts == 0:
            logger.warning("No carts found in data")
            return 0.0

        abandonment_rate = (1 - (completed_orders / total_carts)) * 100
        logger.debug(f"Cart abandonment rate: {abandonment_rate}%")
        return abandonment_rate
    except Exception as e:
        logger.error(f"Error computing cart abandonment rate: {e}")
        return np.nan


@vectorized_operation
@cache_result(expire_after=3600)
@with_progress_bar("Analyzing Order States")
def order_state_analysis(order_df: pd.DataFrame, state_col: str = "État", total_col: str = "Total") -> pd.DataFrame:
    """
    Break down orders by their state using vectorized operations.
    """
    if state_col not in order_df.columns or total_col not in order_df.columns:
        logger.error("Required columns for state analysis not found")
        return pd.DataFrame()

    try:
        # Vectorized aggregation and percentage calculations
        total_orders = len(order_df)
        total_revenue = order_df[total_col].sum()

        # Group by state with multiple aggregations
        state_groups = order_df.groupby(state_col).agg({total_col: ["sum", "count"]}).reset_index()

        # Flatten column names
        state_groups.columns = [state_col, "total_revenue", "order_count"]

        # Vectorized percentage calculations
        state_groups["order_pct"] = (state_groups["order_count"] / total_orders * 100) if total_orders > 0 else 0
        state_groups["revenue_pct"] = (state_groups["total_revenue"] / total_revenue * 100) if total_revenue > 0 else 0

        return state_groups.sort_values("total_revenue", ascending=False)
    except Exception as e:
        logger.error(f"Error in order state analysis: {e}")
        return pd.DataFrame()
