"""
Revenue-focused analytics and metrics.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any
from .utils import vectorized_operation, cache_result, with_progress_bar, parallelize_dataframe, batch_process

logger = logging.getLogger(__name__)


@vectorized_operation
@cache_result(expire_after=3600)
def revenue_concentration(order_df: pd.DataFrame, total_col: str = "Total") -> Dict[str, float]:
    """
    Check revenue concentration using vectorized operations.
    """
    if total_col not in order_df.columns:
        logger.error(f"Column {total_col} not found")
        return {}

    try:
        # Vectorized sorting and calculations
        sorted_orders = order_df[total_col].sort_values(ascending=False)
        total_revenue = sorted_orders.sum()
        top_10pct_count = int(len(sorted_orders) * 0.1)
        top_10pct_revenue = sorted_orders.iloc[:top_10pct_count].sum() if top_10pct_count > 0 else 0
        concentration_pct = (top_10pct_revenue / total_revenue * 100) if total_revenue > 0 else 0

        return {
            "top_10pct_revenue_concentration": concentration_pct,
            "top_10pct_revenue": top_10pct_revenue,
        }
    except Exception as e:
        logger.error(f"Error in revenue concentration analysis: {e}")
        return {}


@vectorized_operation
@cache_result(expire_after=3600)
@with_progress_bar("Analyzing Payment Methods")
def payment_method_analysis(order_df: pd.DataFrame, total_col: str = "Total", payment_col: str = "Paiement") -> pd.DataFrame:
    """
    Analyze revenue and AOV by payment method using vectorized operations.
    """
    logger.info("Analyzing payment methods...")
    if total_col not in order_df.columns or payment_col not in order_df.columns:
        logger.error("Required columns for payment method analysis not found")
        return pd.DataFrame()
    try:
        # Vectorized aggregation
        payment_stats = (
            order_df.groupby(payment_col)[total_col]
            .agg(["sum", "mean", "count"])
            .rename(
                columns={
                    "sum": "total_revenue",
                    "mean": "avg_order_value",
                    "count": "order_count",
                }
            )
        )
        stats = payment_stats.sort_values("total_revenue", ascending=False)
        logger.debug(f"Payment method analysis shape: {stats.shape}")
        return stats
    except Exception as e:
        logger.error(f"Error analyzing payment methods: {e}")
        return pd.DataFrame()


@vectorized_operation
@cache_result(expire_after=3600)
@with_progress_bar("Analyzing Country Data")
def country_analysis(order_df: pd.DataFrame, total_col: str = "Total", country_col: str = "Livraison", client_col: str = "Client") -> pd.DataFrame:
    """
    Analyze top countries by revenue using vectorized operations.
    """
    logger.info("Analyzing country data...")
    if total_col not in order_df.columns or country_col not in order_df.columns or client_col not in order_df.columns:
        logger.error("Required columns for country analysis not found")
        return pd.DataFrame()
    try:
        # Vectorized aggregation
        country_stats = order_df.groupby(country_col).agg({total_col: ["sum", "mean", "count"], client_col: "nunique"})

        # Flatten column names
        country_stats.columns = ["total_revenue", "avg_order_value", "order_count", "unique_customers"]

        stats = country_stats.sort_values("total_revenue", ascending=False)
        logger.debug(f"Country analysis shape: {stats.shape}")
        return stats
    except Exception as e:
        logger.error(f"Error analyzing countries: {e}")
        return pd.DataFrame()


@vectorized_operation
@cache_result(expire_after=3600)
@with_progress_bar("Analyzing Refunds and Cancellations")
def refund_cancellation_analysis(order_df: pd.DataFrame, state_col: str = "État", total_col: str = "Total") -> Dict[str, float]:
    """
    Analyze refund and cancellation rates using vectorized operations.
    """
    logger.info("Analyzing refunds and cancellations...")
    if state_col not in order_df.columns or total_col not in order_df.columns:
        logger.error("Required columns for refund/cancellation analysis not found")
        return {}

    try:
        total_orders = len(order_df)
        total_revenue = order_df[total_col].sum()

        # Vectorized filtering
        state_mask = order_df[state_col].str
        refunded = order_df[state_mask.contains("Remboursé", na=False)]
        canceled = order_df[state_mask.contains("Annulée", na=False)]

        # Vectorized calculations
        refund_rate = (len(refunded) / total_orders * 100) if total_orders > 0 else 0
        cancellation_rate = (len(canceled) / total_orders * 100) if total_orders > 0 else 0

        revenue_lost_refunds = refunded[total_col].sum()
        revenue_lost_cancellations = canceled[total_col].sum()

        revenue_refund_pct = (revenue_lost_refunds / total_revenue * 100) if total_revenue > 0 else 0
        revenue_cancel_pct = (revenue_lost_cancellations / total_revenue * 100) if total_revenue > 0 else 0

        return {
            "refund_rate": refund_rate,
            "cancellation_rate": cancellation_rate,
            "revenue_lost_refunds": revenue_lost_refunds,
            "revenue_refund_pct": revenue_refund_pct,
            "revenue_lost_cancellations": revenue_lost_cancellations,
            "revenue_cancel_pct": revenue_cancel_pct,
        }
    except Exception as e:
        logger.error(f"Error in refund/cancellation analysis: {e}")
        return {}


@vectorized_operation
@cache_result(expire_after=3600)
@with_progress_bar("Analyzing Monthly Trends")
def monthly_cancellation_refund_trends(order_df: pd.DataFrame, state_col: str = "État", date_col: str = "Date") -> pd.DataFrame:
    """
    Analyze monthly trends using vectorized operations.
    """
    if state_col not in order_df.columns or date_col not in order_df.columns:
        logger.error("Required columns for monthly cancellation/refund trends not found")
        return pd.DataFrame()

    try:
        # Vectorized date operations
        order_df[date_col] = pd.to_datetime(order_df[date_col])
        order_df["Month"] = order_df[date_col].dt.to_period("M").astype(str)

        # Vectorized pivot and percentage calculation
        monthly_data = pd.crosstab(order_df["Month"], order_df[state_col], normalize="index") * 100

        # Focus on specific states
        interesting_states = ["Annulée", "Remboursé", "Remboursement partiel"]
        columns_to_keep = [c for c in interesting_states if c in monthly_data.columns]
        return monthly_data[columns_to_keep].fillna(0)
    except Exception as e:
        logger.error(f"Error in monthly cancellation/refund trends: {e}")
        return pd.DataFrame()
