"""
Customer-focused analytics and metrics.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import logging
from typing import Dict, Any
from .utils import (
    vectorized_operation,
    cache_result,
    with_progress_bar,
    parallelize_dataframe,
    batch_process,
)

logger = logging.getLogger(__name__)


@vectorized_operation
@cache_result(expire_after=3600)
@with_progress_bar("Performing RFM Analysis")
def rfm_analysis(
    order_df: pd.DataFrame,
    client_col: str = "Client",
    date_col: str = "Date",
    total_col: str = "Total",
    analysis_date: datetime = None,
) -> pd.DataFrame:
    """
    Conduct RFM analysis (Recency, Frequency, Monetary).
    Uses vectorized operations for better performance.
    """
    logger.info("Performing RFM analysis...")
    if (
        client_col not in order_df.columns
        or date_col not in order_df.columns
        or total_col not in order_df.columns
    ):
        logger.error("Required columns for RFM analysis not found")
        return pd.DataFrame()

    try:
        order_df[date_col] = pd.to_datetime(order_df[date_col])
        if analysis_date is None:
            analysis_date = order_df[date_col].max() + pd.Timedelta(days=1)

        # Vectorized aggregation
        rfm = (
            order_df.groupby(client_col)
            .agg(
                {
                    date_col: lambda x: (analysis_date - x.max()).days,
                    total_col: "sum",
                    client_col: "count",
                }
            )
            .rename(
                columns={
                    date_col: "Recency",
                    total_col: "Monetary",
                    client_col: "Frequency",
                }
            )
        )

        # Vectorized scoring using quantiles
        for metric in ["Recency", "Frequency", "Monetary"]:
            quantiles = rfm[metric].quantile([0.25, 0.5, 0.75])
            if metric == "Recency":  # Lower is better
                rfm[f"{metric[0]}_score"] = pd.qcut(
                    rfm[metric], q=4, labels=[4, 3, 2, 1]
                )
            else:  # Higher is better
                rfm[f"{metric[0]}_score"] = pd.qcut(
                    rfm[metric], q=4, labels=[1, 2, 3, 4]
                )

        # Vectorized string concatenation
        rfm["RFM_score"] = (
            rfm["R_score"].astype(str)
            + rfm["F_score"].astype(str)
            + rfm["M_score"].astype(str)
        )

        logger.debug(f"RFM analysis shape: {rfm.shape}")
        return rfm
    except Exception as e:
        logger.error(f"Error in RFM analysis: {e}")
        return pd.DataFrame()


@vectorized_operation
@cache_result(expire_after=3600)
@with_progress_bar("Performing Customer Segmentation")
def customer_segmentation_by_value(
    order_df: pd.DataFrame, client_col: str = "Client", total_col: str = "Total"
) -> pd.DataFrame:
    """
    Segment customers into tiers based on total spend using vectorized operations.
    """
    logger.info("Performing customer segmentation...")
    if client_col not in order_df.columns or total_col not in order_df.columns:
        logger.error("Required columns for customer segmentation not found")
        return pd.DataFrame()

    try:
        # Vectorized aggregation and sorting
        customer_spend = order_df.groupby(client_col)[total_col].sum()
        total = customer_spend.sum()

        # Calculate cumulative percentages
        seg_df = pd.DataFrame(customer_spend).sort_values(total_col, ascending=False)
        seg_df["cumsum_pct"] = (seg_df[total_col].cumsum() / total) * 100

        # Vectorized segmentation
        seg_df["segment"] = pd.cut(
            seg_df["cumsum_pct"],
            bins=[0, 20, 50, 100],
            labels=["High-value", "Mid-value", "Low-value"],
        )

        logger.debug(f"Customer segmentation shape: {seg_df.shape}")
        return seg_df
    except Exception as e:
        logger.error(f"Error in customer segmentation: {e}")
        return pd.DataFrame()


@vectorized_operation
@cache_result(expire_after=3600)
def calculate_clv(
    order_df: pd.DataFrame,
    total_col: str = "Total",
    client_col: str = "Client",
    date_col: str = "Date",
    lifetime_months: int = 12,
) -> float:
    """
    Calculate Customer Lifetime Value (CLV) using vectorized operations.
    """
    if total_col not in order_df.columns or client_col not in order_df.columns:
        logger.error("Required columns for CLV calculation not found")
        return np.nan

    try:
        # Vectorized calculations
        avg_order_value = order_df[total_col].mean()
        purchase_frequency = order_df.groupby(client_col).size().mean()
        clv = avg_order_value * purchase_frequency * lifetime_months
        return clv
    except Exception as e:
        logger.error(f"Error calculating CLV: {e}")
        return np.nan


@vectorized_operation
@cache_result(expire_after=3600)
def analyze_customer_behavior(
    order_df: pd.DataFrame,
    client_col: str = "Client",
    total_col: str = "Total",
    status_col: str = "Status",
) -> Dict[str, Any]:
    """
    Analyze customer behavior including repeat purchases, revenue, and order cancellations.
    """
    if not all(col in order_df.columns for col in [client_col, total_col, status_col]):
        logger.error("Required columns for customer behavior analysis not found")
        return {}

    try:
        # Count orders per customer
        customer_orders = (
            order_df.groupby(client_col)
            .agg(
                {
                    total_col: "sum",  # Total revenue
                    client_col: "count",  # Number of orders
                    status_col: lambda x: (
                        x == "Annulée"
                    ).sum(),  # Count cancelled orders
                }
            )
            .rename(
                columns={
                    total_col: "total_revenue",
                    client_col: "order_count",
                    status_col: "cancelled_orders",
                }
            )
        )

        # Categorize customers
        customer_orders["customer_type"] = np.where(
            customer_orders["order_count"] > 1, "repeat", "one_time"
        )

        # Calculate metrics
        results = {
            "repeat_customers": {
                "count": (customer_orders["customer_type"] == "repeat").sum(),
                "total_revenue": customer_orders[
                    customer_orders["customer_type"] == "repeat"
                ]["total_revenue"].sum(),
                "cancelled_orders": customer_orders[
                    customer_orders["customer_type"] == "repeat"
                ]["cancelled_orders"].sum(),
            },
            "one_time_customers": {
                "count": (customer_orders["customer_type"] == "one_time").sum(),
                "total_revenue": customer_orders[
                    customer_orders["customer_type"] == "one_time"
                ]["total_revenue"].sum(),
                "cancelled_orders": customer_orders[
                    customer_orders["customer_type"] == "one_time"
                ]["cancelled_orders"].sum(),
            },
            "top_repeat_customers": customer_orders[
                customer_orders["customer_type"] == "repeat"
            ]
            .nlargest(10, "total_revenue")
            .to_dict(),
        }

        # Calculate percentages
        total_customers = len(customer_orders)
        total_revenue = customer_orders["total_revenue"].sum()

        results["metrics"] = {
            "repeat_customer_rate": (
                results["repeat_customers"]["count"] / total_customers * 100
            ),
            "repeat_revenue_rate": (
                results["repeat_customers"]["total_revenue"] / total_revenue * 100
            ),
            "avg_order_value_repeat": (
                results["repeat_customers"]["total_revenue"]
                / customer_orders[customer_orders["customer_type"] == "repeat"][
                    "order_count"
                ].sum()
            ),
            "avg_order_value_one_time": (
                results["one_time_customers"]["total_revenue"]
                / results["one_time_customers"]["count"]
            ),
        }

        return results

    except Exception as e:
        logger.error(f"Error analyzing customer behavior: {e}")
        return {}


@vectorized_operation
@cache_result(expire_after=3600)
def churn_rate(
    order_df: pd.DataFrame,
    client_col: str = "Client",
    date_col: str = "Date",
    inactivity_period: int = 90,
) -> float:
    """
    Calculate churn rate using vectorized operations.
    """
    if client_col not in order_df.columns or date_col not in order_df.columns:
        logger.error("Required columns for churn rate not found")
        return np.nan

    try:
        # Vectorized date operations
        order_df[date_col] = pd.to_datetime(order_df[date_col])
        analysis_date = order_df[date_col].max() + pd.Timedelta(days=1)

        # Vectorized customer analysis
        last_purchase = order_df.groupby(client_col)[date_col].max()
        inactive_customers = (analysis_date - last_purchase) > pd.Timedelta(
            days=inactivity_period
        )
        churned = inactive_customers.sum()
        total_customers = len(last_purchase)

        return (churned / total_customers * 100) if total_customers > 0 else np.nan
    except Exception as e:
        logger.error(f"Error calculating churn rate: {e}")
        return np.nan
