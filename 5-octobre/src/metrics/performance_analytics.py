"""
Performance analytics and metrics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Dict, Any, Tuple
from src.metrics.utils import vectorized_operation, cache_result

logger = logging.getLogger(__name__)


def margins(
    order_df: pd.DataFrame,
    cost_col: str = "factory_price",
    price_col: str = "retail",
    quantity_col: str = "qty",
) -> pd.DataFrame:
    """
    Calculate margin per transaction.

    For each sold item, computes gross profit as:
    (Price Paid - Factory Cost) * Quantity

    Args:
        order_df: DataFrame containing order data
        cost_col: Column name for factory/cost price
        price_col: Column name for retail price
        quantity_col: Column name for quantity sold

    Returns:
        DataFrame with gross profit per line item
    """
    try:
        # Validate required columns exist
        required_cols = [cost_col, price_col, quantity_col]
        if not all(col in order_df.columns for col in required_cols):
            raise ValueError(f"Missing required columns. Need: {required_cols}")

        # Calculate margin per transaction
        margins = (order_df[price_col] - order_df[cost_col]) * order_df[quantity_col]

        return margins

    except Exception as e:
        logger.error(f"Error calculating margins: {e}")
        return pd.DataFrame()


def gross_margin_rate(
    order_df: pd.DataFrame,
    cost_col: str = "factory_price",
    price_col: str = "retail",
    quantity_col: str = "qty",
    groupby: str = None,
    time_freq: str = None,
) -> pd.DataFrame:
    """
    Evaluate margin rates by product, category, or timeframe.

    Args:
        order_df: DataFrame containing order data
        cost_col: Column name for cost/factory price
        price_col: Column name for retail price
        quantity_col: Column name for quantity
        groupby: Column to group margins by (e.g. 'product', 'category')
        time_freq: Time frequency for analysis ('D'=daily, 'W'=weekly, 'M'=monthly)

    Returns:
        DataFrame with margin analysis grouped by specified dimension, including:
        - Total Revenue
        - Total Cost
        - Gross Profit
        - Margin Rate %
        - Unit Volume
    """
    # Input validation
    required_cols = [cost_col, price_col, quantity_col]
    if groupby:
        required_cols.append(groupby)
    if time_freq and "Date" not in order_df.columns:
        raise ValueError("Date column required for time-based analysis")
    if not all(col in order_df.columns for col in required_cols):
        raise ValueError(f"Missing required columns. Need: {required_cols}")

    # Calculate metrics
    df = order_df.copy()
    df["revenue"] = df[price_col] * df[quantity_col]
    df["total_cost"] = df[cost_col] * df[quantity_col]
    df["gross_profit"] = df["revenue"] - df["total_cost"]

    # Setup grouping
    group_cols = []
    if groupby:
        group_cols.append(groupby)
    if time_freq:
        df["period"] = pd.to_datetime(df["Date"]).dt.to_period(time_freq)
        group_cols.append("period")

    # Group and aggregate
    if group_cols:
        results = (
            df.groupby(group_cols)
            .agg(
                {
                    "revenue": "sum",
                    "total_cost": "sum",
                    "gross_profit": "sum",
                    quantity_col: "sum",
                }
            )
            .reset_index()
        )
    else:
        results = pd.DataFrame(
            {
                "revenue": [df["revenue"].sum()],
                "total_cost": [df["total_cost"].sum()],
                "gross_profit": [df["gross_profit"].sum()],
                quantity_col: [df[quantity_col].sum()],
            }
        )

    # Calculate margin rate
    results["margin_rate"] = (results["gross_profit"] / results["revenue"] * 100).round(
        2
    )

    # Add summary stats
    stats = {
        "avg_margin": results["margin_rate"].mean(),
        "median_margin": results["margin_rate"].median(),
        "margin_std": results["margin_rate"].std(),
        "min_margin": results["margin_rate"].min(),
        "max_margin": results["margin_rate"].max(),
    }

    # Sort by margin rate descending
    results = results.sort_values("margin_rate", ascending=False)

    return results, stats


@vectorized_operation
@cache_result(expire_after=3600)
def calculate_inventory_metrics(
    retail_df: pd.DataFrame, inventory_df: pd.DataFrame, period: str = "W"
) -> pd.DataFrame:
    """
    Calculate stock turnover ratio and days of inventory for products.

    Args:
        retail_df: DataFrame containing retail sales data
        inventory_df: DataFrame containing current inventory data
        period: Frequency for sales velocity calculation ('D' for daily, 'W' for weekly)

    Returns:
        DataFrame with inventory metrics including:
        - Daily and weekly sales velocity
        - Stock turnover ratio
        - Days of inventory remaining
        - Stock status indicators
    """
    # Convert date column and ensure proper format
    retail_df["Date"] = pd.to_datetime(retail_df["Date"])

    # Calculate sales velocity (daily and weekly)
    daily_sales = (
        retail_df.groupby(["Ref", pd.Grouper(key="Date", freq="D")])["Qté"]
        .sum()
        .reset_index()
    )
    weekly_sales = (
        retail_df.groupby(["Ref", pd.Grouper(key="Date", freq="W")])["Qté"]
        .sum()
        .reset_index()
    )

    # Calculate average daily and weekly sales
    daily_avg = (
        daily_sales.groupby("Ref")["Qté"].agg(["mean", "std", "count"]).reset_index()
    )
    daily_avg.columns = ["Ref", "daily_avg_sales", "daily_sales_std", "days_with_sales"]

    weekly_avg = (
        weekly_sales.groupby("Ref")["Qté"].agg(["mean", "std", "count"]).reset_index()
    )
    weekly_avg.columns = [
        "Ref",
        "weekly_avg_sales",
        "weekly_sales_std",
        "weeks_with_sales",
    ]

    # Merge with inventory data
    inventory_metrics = inventory_df[["id", "qty"]].copy()
    inventory_metrics = inventory_metrics.rename(columns={"id": "Ref"})

    # Merge daily and weekly metrics
    inventory_metrics = inventory_metrics.merge(daily_avg, on="Ref", how="left").merge(
        weekly_avg, on="Ref", how="left"
    )

    # Fill NaN values with 0 for products without sales
    inventory_metrics = inventory_metrics.fillna(0)

    # Calculate inventory metrics
    inventory_metrics["daily_turnover_ratio"] = (
        inventory_metrics["daily_avg_sales"] * 30
    ) / inventory_metrics["qty"].clip(lower=0.1)
    inventory_metrics["weekly_turnover_ratio"] = (
        inventory_metrics["weekly_avg_sales"] * 4
    ) / inventory_metrics["qty"].clip(lower=0.1)

    inventory_metrics["days_of_inventory"] = np.where(
        inventory_metrics["daily_avg_sales"] > 0,
        inventory_metrics["qty"] / inventory_metrics["daily_avg_sales"],
        float("inf"),
    )

    # Add stock status indicators
    inventory_metrics["stock_status"] = pd.cut(
        inventory_metrics["days_of_inventory"],
        bins=[-float("inf"), 15, 30, 60, float("inf")],
        labels=["Critical", "Low", "Adequate", "Excess"],
    )

    # Calculate coefficient of variation to measure sales consistency
    inventory_metrics["daily_sales_cv"] = np.where(
        inventory_metrics["daily_avg_sales"] > 0,
        inventory_metrics["daily_sales_std"] / inventory_metrics["daily_avg_sales"],
        0,
    )

    # Add risk indicators
    inventory_metrics["stock_risk"] = np.where(
        (inventory_metrics["days_of_inventory"] < 15)
        & (inventory_metrics["daily_sales_cv"] < 1),
        "High Risk - Consistent Sales",
        np.where(
            (inventory_metrics["days_of_inventory"] < 15)
            & (inventory_metrics["daily_sales_cv"] >= 1),
            "Medium Risk - Volatile Sales",
            "Low Risk",
        ),
    )

    return inventory_metrics


def enrich_retail_data(
    retail_df: pd.DataFrame,
    inventory_df: pd.DataFrame,
    retail_id_col: str = "Ref",
    inventory_id_col: str = "id",
) -> pd.DataFrame:
    """
    Enrich retail data with inventory information.

    Args:
        retail_df: DataFrame containing retail/sales data
        inventory_df: DataFrame containing inventory data
        retail_id_col: Column name for product ID in retail data
        inventory_id_col: Column name for product ID in inventory data

    Returns:
        DataFrame with enriched retail data including inventory information
    """
    try:
        # Create copy to avoid modifying original
        enriched_df = retail_df.copy()

        # Merge retail with inventory data
        enriched_df = enriched_df.merge(
            inventory_df[
                [inventory_id_col, "factory_price", "retail", "retail_us", "sfa", "lib"]
            ],
            left_on=retail_id_col,
            right_on=inventory_id_col,
            how="left",
        )

        # Log any unmatched products
        unmatched = enriched_df[enriched_df["factory_price"].isna()][
            retail_id_col
        ].unique()
        if len(unmatched) > 0:
            logger.warning(f"Found {len(unmatched)} products without inventory data")

        return enriched_df

    except Exception as e:
        logger.error(f"Error enriching retail data: {e}")
        return pd.DataFrame()


def product_profitability_analysis(
    retail_df: pd.DataFrame,
    inventory_df: pd.DataFrame,
    period: str = "M",
    min_sales: int = 1,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Detailed product-level profitability analysis.

    Args:
        retail_df: DataFrame containing retail/sales data
        inventory_df: DataFrame containing inventory data
        period: Time frequency for analysis ('D'=daily, 'W'=weekly, 'M'=monthly)
        min_sales: Minimum number of sales to include in analysis

    Returns:
        Tuple containing:
        - DataFrame with product profitability metrics
        - Dictionary with summary statistics
    """
    try:
        # Enrich retail data with inventory information
        enriched_df = enrich_retail_data(retail_df, inventory_df)
        if enriched_df.empty:
            return pd.DataFrame(), {}

        # Convert date and set time period
        enriched_df["Date"] = pd.to_datetime(enriched_df["Date"])
        enriched_df["period"] = enriched_df["Date"].dt.to_period(period)

        # Calculate key metrics per product
        product_metrics = (
            enriched_df.groupby(["Ref", "lib", "sfa"])
            .agg(
                {
                    "Qté": ["sum", "count"],
                    "factory_price": "first",
                    "retail": "first",
                    "retail_us": "first",
                    "period": "nunique",
                }
            )
            .reset_index()
        )

        # Rename columns
        product_metrics.columns = [
            "product_id",
            "product_name",
            "category",
            "total_units",
            "total_transactions",
            "cost_price",
            "retail_price",
            "retail_price_us",
            "active_periods",
        ]

        # Calculate profitability metrics
        product_metrics["total_revenue"] = (
            product_metrics["total_units"] * product_metrics["retail_price"]
        )
        product_metrics["total_cost"] = (
            product_metrics["total_units"] * product_metrics["cost_price"]
        )
        product_metrics["gross_profit"] = (
            product_metrics["total_revenue"] - product_metrics["total_cost"]
        )
        product_metrics["margin_rate"] = (
            product_metrics["gross_profit"] / product_metrics["total_revenue"] * 100
        ).round(2)
        product_metrics["avg_units_per_period"] = (
            product_metrics["total_units"] / product_metrics["active_periods"]
        ).round(2)

        # Filter by minimum sales threshold
        product_metrics = product_metrics[product_metrics["total_units"] >= min_sales]

        # Calculate summary statistics
        stats = {
            "total_products": len(product_metrics),
            "total_revenue": product_metrics["total_revenue"].sum(),
            "total_profit": product_metrics["gross_profit"].sum(),
            "avg_margin_rate": product_metrics["margin_rate"].mean(),
            "median_margin_rate": product_metrics["margin_rate"].median(),
            "top_margin_rate": product_metrics["margin_rate"].max(),
            "bottom_margin_rate": product_metrics["margin_rate"].min(),
        }

        # Sort by gross profit descending
        product_metrics = product_metrics.sort_values("gross_profit", ascending=False)

        return product_metrics, stats

    except Exception as e:
        logger.error(f"Error in product profitability analysis: {e}")
        return pd.DataFrame(), {}


def abc_inventory_analysis(
    retail_df: pd.DataFrame,
    inventory_df: pd.DataFrame,
    value_thresholds: Dict[str, float] = {"A": 0.8, "B": 0.95},
    period: str = "M",
) -> pd.DataFrame:
    """
    Perform ABC analysis on inventory based on revenue contribution.

    Args:
        retail_df: DataFrame containing retail/sales data
        inventory_df: DataFrame containing inventory data
        value_thresholds: Dictionary defining cumulative value thresholds for A and B categories
        period: Time period for analysis

    Returns:
        DataFrame with ABC classification and metrics for each product
    """
    try:
        # Get product profitability metrics
        product_metrics, _ = product_profitability_analysis(
            retail_df, inventory_df, period=period
        )
        if product_metrics.empty:
            return pd.DataFrame()

        # Calculate cumulative revenue contribution
        product_metrics = product_metrics.sort_values("total_revenue", ascending=False)
        total_revenue = product_metrics["total_revenue"].sum()
        product_metrics["revenue_contribution"] = (
            product_metrics["total_revenue"] / total_revenue
        )
        product_metrics["cumulative_contribution"] = product_metrics[
            "revenue_contribution"
        ].cumsum()

        # Assign ABC categories
        def get_category(cum_value):
            if cum_value <= value_thresholds["A"]:
                return "A"
            elif cum_value <= value_thresholds["B"]:
                return "B"
            return "C"

        product_metrics["abc_category"] = product_metrics[
            "cumulative_contribution"
        ].apply(get_category)

        # Add category statistics
        category_stats = (
            product_metrics.groupby("abc_category")
            .agg(
                {
                    "product_id": "count",
                    "total_revenue": "sum",
                    "total_units": "sum",
                    "gross_profit": "sum",
                }
            )
            .reset_index()
        )

        category_stats = category_stats.rename(columns={"product_id": "product_count"})
        category_stats["revenue_percent"] = (
            category_stats["total_revenue"] / total_revenue * 100
        ).round(2)

        # Add inventory metrics
        current_stock = inventory_df.groupby("id")["qty"].sum()
        product_metrics["current_stock"] = product_metrics["product_id"].map(
            current_stock
        )
        product_metrics["stock_value"] = (
            product_metrics["current_stock"] * product_metrics["cost_price"]
        )

        return product_metrics, category_stats

    except Exception as e:
        logger.error(f"Error in ABC analysis: {e}")
        return pd.DataFrame(), pd.DataFrame()


def calculate_stock_recommendations(
    retail_df: pd.DataFrame,
    inventory_df: pd.DataFrame,
    safety_stock_days: Dict[str, int] = {"A": 14, "B": 21, "C": 30},
    lead_time_days: int = 7,
    period: str = "D",
) -> pd.DataFrame:
    """
    Calculate stock recommendations based on ABC analysis and sales patterns.

    Args:
        retail_df: DataFrame containing retail/sales data
        inventory_df: DataFrame containing inventory data
        safety_stock_days: Dictionary defining safety stock days by ABC category
        lead_time_days: Lead time for replenishment in days
        period: Time period for analysis

    Returns:
        DataFrame with stock recommendations per product
    """
    try:
        # Get ABC analysis results
        abc_results, _ = abc_inventory_analysis(retail_df, inventory_df, period=period)
        if abc_results.empty:
            return pd.DataFrame()

        # Calculate daily sales statistics
        enriched_df = enrich_retail_data(retail_df, inventory_df)
        daily_sales = enriched_df.groupby(["Date", "Ref"])["Qté"].sum().reset_index()

        sales_stats = (
            daily_sales.groupby("Ref")
            .agg({"Qté": ["mean", "std", "max"]})
            .reset_index()
        )

        sales_stats.columns = [
            "product_id",
            "avg_daily_sales",
            "sales_std",
            "max_daily_sales",
        ]

        # Merge with ABC results
        recommendations = abc_results.merge(sales_stats, on="product_id", how="left")

        # Calculate reorder points and quantities
        recommendations["safety_stock_days"] = recommendations["abc_category"].map(
            safety_stock_days
        )
        recommendations["safety_stock"] = (
            recommendations["avg_daily_sales"] * recommendations["safety_stock_days"]
            + recommendations["sales_std"] * np.sqrt(lead_time_days)
        ).round(0)

        recommendations["reorder_point"] = (
            recommendations["avg_daily_sales"] * lead_time_days
            + recommendations["safety_stock"]
        ).round(0)

        recommendations["optimal_order_qty"] = (
            recommendations["avg_daily_sales"]
            * np.sqrt(2 * lead_time_days * recommendations["safety_stock_days"])
        ).round(0)

        # Add stock status
        recommendations["stock_status"] = np.where(
            recommendations["current_stock"] <= recommendations["safety_stock"],
            "Critical",
            np.where(
                recommendations["current_stock"] <= recommendations["reorder_point"],
                "Reorder",
                "Adequate",
            ),
        )

        # Calculate days of stock remaining
        recommendations["days_of_stock"] = np.where(
            recommendations["avg_daily_sales"] > 0,
            (
                recommendations["current_stock"] / recommendations["avg_daily_sales"]
            ).round(0),
            999,
        )

        return recommendations

    except Exception as e:
        logger.error(f"Error calculating stock recommendations: {e}")
        return pd.DataFrame()


def analyze_product_movement(
    retail_df: pd.DataFrame,
    inventory_df: pd.DataFrame,
    analysis_period: str = "M",
    slow_mover_threshold: int = 1,  # Units per period
    bestseller_threshold: float = 0.8,  # Top 20% by sales volume
    stock_coverage_days: int = 90,  # Days to analyze stock coverage
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Analyze product movement patterns and stock efficiency.
    Identifies best-sellers, slow movers, and dead stock while tracking stock availability.

    Args:
        retail_df: DataFrame containing retail sales data
        inventory_df: DataFrame containing inventory data
        analysis_period: Time period for analysis ('D'=daily, 'W'=weekly, 'M'=monthly)
        slow_mover_threshold: Units sold per period below which product is considered slow-moving
        bestseller_threshold: Percentile threshold for bestseller classification
        stock_coverage_days: Number of days to analyze stock coverage

    Returns:
        Tuple containing:
        - DataFrame with product movement analysis
        - Dictionary with summary statistics
    """
    try:
        # Ensure date format
        retail_df = retail_df.copy()
        retail_df["Date"] = pd.to_datetime(retail_df["Date"])
        analysis_end_date = retail_df["Date"].max()
        analysis_start_date = analysis_end_date - pd.Timedelta(days=stock_coverage_days)

        # Filter data for analysis period
        period_sales = retail_df[retail_df["Date"] >= analysis_start_date]

        # Calculate sales metrics
        sales_analysis = (
            period_sales.groupby("Ref")
            .agg({"Qté": ["sum", "mean", "count"], "Date": ["min", "max"]})
            .reset_index()
        )

        sales_analysis.columns = [
            "product_id",
            "total_units_sold",
            "avg_units_per_period",
            "sales_frequency",
            "first_sale",
            "last_sale",
        ]

        # Calculate days since last sale
        sales_analysis["days_since_last_sale"] = (
            analysis_end_date - sales_analysis["last_sale"]
        ).dt.days

        # Merge with current inventory
        product_analysis = sales_analysis.merge(
            inventory_df[["id", "qty", "retail", "factory_price"]],
            left_on="product_id",
            right_on="id",
            how="outer",
        ).fillna(
            {"total_units_sold": 0, "avg_units_per_period": 0, "sales_frequency": 0}
        )

        # Calculate stock value
        product_analysis["stock_value"] = (
            product_analysis["qty"] * product_analysis["factory_price"]
        )

        # Calculate stock coverage
        product_analysis["daily_sales_rate"] = (
            product_analysis["total_units_sold"] / stock_coverage_days
        )
        product_analysis["days_of_stock"] = np.where(
            product_analysis["daily_sales_rate"] > 0,
            product_analysis["qty"] / product_analysis["daily_sales_rate"],
            float("inf"),
        )

        # Determine bestseller threshold
        bestseller_min_units = product_analysis["total_units_sold"].quantile(
            1 - bestseller_threshold
        )

        # Categorize products
        def get_movement_category(row):
            if row["total_units_sold"] >= bestseller_min_units:
                return "Best Seller"
            elif row["total_units_sold"] == 0:
                return "Dead Stock"
            elif row["avg_units_per_period"] <= slow_mover_threshold:
                return "Slow Mover"
            else:
                return "Regular Mover"

        product_analysis["movement_category"] = product_analysis.apply(
            get_movement_category, axis=1
        )

        # Add stock efficiency status
        def get_stock_status(row):
            if row["qty"] == 0 and row["total_units_sold"] > 0:
                return "Out of Stock"
            elif row["days_of_stock"] <= 30:
                return "Low Stock"
            elif row["days_of_stock"] >= 180:
                return "Overstocked"
            else:
                return "Optimal"

        product_analysis["stock_status"] = product_analysis.apply(
            get_stock_status, axis=1
        )

        # Calculate summary statistics
        summary_stats = {
            "category_counts": product_analysis["movement_category"]
            .value_counts()
            .to_dict(),
            "stock_status_counts": product_analysis["stock_status"]
            .value_counts()
            .to_dict(),
            "total_stock_value": product_analysis["stock_value"].sum(),
            "dead_stock_value": product_analysis[
                product_analysis["movement_category"] == "Dead Stock"
            ]["stock_value"].sum(),
            "bestseller_stats": {
                "count": (product_analysis["movement_category"] == "Best Seller").sum(),
                "total_units": product_analysis[
                    product_analysis["movement_category"] == "Best Seller"
                ]["total_units_sold"].sum(),
                "stock_availability": product_analysis[
                    product_analysis["movement_category"] == "Best Seller"
                ]["stock_status"]
                .value_counts()
                .to_dict(),
            },
        }

        # Add recommendations
        def get_recommendation(row):
            if row["movement_category"] == "Dead Stock":
                return "Consider clearance/promotion or discontinuation"
            elif row["movement_category"] == "Best Seller" and row["stock_status"] in [
                "Out of Stock",
                "Low Stock",
            ]:
                return "Urgent restock needed"
            elif (
                row["movement_category"] == "Slow Mover"
                and row["stock_status"] == "Overstocked"
            ):
                return "Consider promotion or inventory reduction"
            elif row["stock_status"] == "Overstocked":
                return "Review order quantities"
            elif (
                row["stock_status"] == "Low Stock"
                and row["movement_category"] != "Slow Mover"
            ):
                return "Plan restock"
            else:
                return "Maintain current strategy"

        product_analysis["recommendation"] = product_analysis.apply(
            get_recommendation, axis=1
        )

        # Sort by total units sold descending
        product_analysis = product_analysis.sort_values(
            "total_units_sold", ascending=False
        )

        return product_analysis, summary_stats

    except Exception as e:
        logger.error(f"Error in product movement analysis: {e}")
        return pd.DataFrame(), {}


if __name__ == "__main__":
    retail_df = pd.read_csv(
        "/Users/benpfeffer/Library/Mobile Documents/com~apple~CloudDocs/projects-portfolio-main/5-octobre/data/cleaned/retail.csv"
    )
    inventory_df = pd.read_csv(
        "/Users/benpfeffer/Library/Mobile Documents/com~apple~CloudDocs/projects-portfolio-main/5-octobre/data/cleaned/inventory.csv"
    )
    # Get product profitability insights
    product_smetrics, summary = product_profitability_analysis(retail_df, inventory_df)
    # Perform ABC analysis
    abc_results, category_stats = abc_inventory_analysis(retail_df, inventory_df)
    # Get stock recommendations
    stock_recommendations = calculate_stock_recommendations(retail_df, inventory_df)
