import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys

# Adjust the system path and imports as needed for your project structure
sys.path.append("/Users/benpfeffer/Library/Mobile Documents/com~apple~CloudDocs/projects-portfolio-main/5-octobre")
from src.config import CLEANED_DATA_DIR, CART_FILENAME, ORDER_FILENAME
from src.config import load_logger

logger = load_logger()

##############################################################
# NOTE:
# This updated code acknowledges the current limitations:
# - We have an 'order_df' with aggregated order-level data (no product breakdown).
# - We have a 'cart_df' with abandoned cart info.
# - We have a 'products_df' (5octobre_products.csv) with product listings.
#
# Without line-item detail (which product was sold on which order, and in what quantity),
# we cannot compute product-level metrics like units sold per product, top-selling products, etc.
#
# Below, we provide a framework. If in the future you add line-item level data (order_items_df),
# you can reintegrate the product-level metrics functions.
##############################################################


def load_product_data(products_file_path):
    """
    Load product data from the provided CSV file.

    Parameters
    ----------
    products_file_path : str
        Path to the CSV file containing product data, e.g., '5octobre_products.csv'.

    Returns
    -------
    pd.DataFrame
        DataFrame of products with at least 'name' and 'price' columns.
    """
    logger.info(f"Loading product data from {products_file_path}...")
    try:
        products_df = pd.read_csv(products_file_path)
        logger.debug(f"Products data shape: {products_df.shape}")
        return products_df
    except FileNotFoundError as e:
        logger.error(f"Product file not found: {e}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error loading product data: {e}")
        return pd.DataFrame()


def load_cleaned_data():
    """
    Load the cleaned cart and order datasets.

    Returns
    -------
    tuple of (pd.DataFrame, pd.DataFrame)
        cart_df and order_df respectively.
    """
    logger.info("Loading cleaned data files...")
    cart_path = os.path.join(CLEANED_DATA_DIR, CART_FILENAME)
    order_path = os.path.join(CLEANED_DATA_DIR, ORDER_FILENAME)

    try:
        cart_df = pd.read_csv(cart_path)
        order_df = pd.read_csv(order_path)
        logger.info("Successfully loaded cleaned data files")
        logger.debug(f"Cart data shape: {cart_df.shape}")
        logger.debug(f"Order data shape: {order_df.shape}")
        return cart_df, order_df
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return None, None
    except pd.errors.EmptyDataError:
        logger.error("One of the cleaned files is empty")
        return None, None
    except Exception as e:
        logger.error(f"Error loading cleaned data: {e}")
        return None, None


#############################################
# Basic / Order-Level Metrics
#############################################


def basic_kpis(order_df, total_col="Total", client_col="Client"):
    """
    Compute basic KPIs:
    - Total orders
    - Total revenue
    - Unique customers
    - Average orders per customer
    """
    logger.info("Computing basic KPIs...")
    if total_col not in order_df.columns or client_col not in order_df.columns:
        logger.error("Required columns for basic KPIs not found")
        return {}

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
    logger.info("Basic KPIs computation complete")
    return kpis


def compute_revenue_over_time(order_df, freq="M", date_col="Date", total_col="Total"):
    """Compute total revenue aggregated by a specified time frequency."""
    logger.info(f"Computing revenue over time with frequency {freq}...")
    if date_col not in order_df.columns or total_col not in order_df.columns:
        logger.error("Required columns for revenue over time not found")
        return pd.Series(dtype=float)
    try:
        order_df[date_col] = pd.to_datetime(order_df[date_col], errors="coerce")
        revenue_series = order_df.set_index(date_col).resample("ME")[total_col].sum()
        logger.debug(f"Revenue series shape: {revenue_series.shape}")
        logger.info("Revenue over time computation complete")
        return revenue_series
    except Exception as e:
        logger.error(f"Error computing revenue over time: {e}")
        return pd.Series(dtype=float)


def compute_average_order_value(order_df, total_col="Total"):
    """Compute the Average Order Value (AOV)."""
    logger.info("Computing average order value...")
    if total_col not in order_df.columns:
        logger.error(f"Column {total_col} not found in orders data")
        return np.nan
    try:
        aov = order_df[total_col].mean()
        logger.debug(f"Calculated AOV: {aov}")
        logger.info("AOV computation complete")
        return aov
    except Exception as e:
        logger.error(f"Error computing average order value: {e}")
        return np.nan


def compute_cart_abandonment_rate(cart_df, order_df, cart_id_col="ID commande", order_ref_col="Référence"):
    """Compute cart abandonment rate."""
    logger.info("Computing cart abandonment rate...")
    if cart_id_col not in cart_df.columns:
        logger.error(f"Column {cart_id_col} not found in cart data")
        return np.nan
    if order_ref_col not in order_df.columns:
        logger.error(f"Column {order_ref_col} not found in order data")
        return np.nan

    try:
        total_carts = cart_df[cart_id_col].nunique()
        completed_orders = cart_df[cart_id_col].isin(order_df[order_ref_col].unique()).sum()
        if total_carts == 0:
            logger.warning("No carts found in data")
            return 0.0
        abandonment_rate = (1 - (completed_orders / total_carts)) * 100
        logger.debug(f"Cart abandonment rate: {abandonment_rate}%")
        logger.info("Cart abandonment rate computation complete")
        return abandonment_rate
    except Exception as e:
        logger.error(f"Error computing cart abandonment rate: {e}")
        return np.nan


def analyze_customer_count(order_df, date_col="Date", client_col="Client", freq="M"):
    """Analyze unique customers over time."""
    logger.info(f"Analyzing customer count with frequency {freq}...")
    if date_col not in order_df.columns or client_col not in order_df.columns:
        logger.error("Required columns for customer analysis not found")
        return pd.Series(dtype=float)

    try:
        order_df[date_col] = pd.to_datetime(order_df[date_col], errors="coerce")
        customer_series = order_df.set_index(date_col).groupby(pd.Grouper(freq="ME"))[client_col].nunique()
        logger.debug(f"Customer count series shape: {customer_series.shape}")
        logger.info("Customer count analysis complete")
        return customer_series
    except Exception as e:
        logger.error(f"Error analyzing customer count: {e}")
        return pd.Series(dtype=float)


#######################################
# Product-Level Metrics (Not Available)
#######################################
# Without line-item (order_items_df) data linking orders to products, we cannot compute:
# - Top Selling Products by Volume & Revenue
# - Product-level repeat purchase rate
# - Product mix & cross-sell
#
# If you obtain `order_items_df`, you can re-enable the following functions.
# For now, they will return empty DataFrames or placeholders.


def compute_product_sales_metrics(order_items_df, products_df, top_n=10):
    """
    Placeholder function. Returns empty results since we lack line-item data.
    """
    logger.warning("Cannot compute product-level sales metrics without line-item data.")
    return {"product_sales_summary": pd.DataFrame(), "top_products_by_revenue": pd.DataFrame(), "top_10_revenue_share": 0.0}


def product_level_repeat_purchase_rate(order_items_df):
    """
    Placeholder for product-level repeat purchase rate. Returns empty DataFrame.
    """
    logger.warning("Cannot compute product-level repeat purchase rates without line-item data.")
    return pd.DataFrame()


def average_items_per_order(order_items_df):
    """
    Placeholder function, returns NaN without line-item data.
    """
    logger.warning("Cannot compute average items per order without line-item data.")
    return np.nan


def product_mix_cross_sell_analysis(order_items_df, min_frequency=2):
    """
    Placeholder for cross-sell analysis. Returns empty DataFrame.
    """
    logger.warning("Cannot compute product mix/cross-sell without line-item data.")
    return pd.DataFrame()


#############################################
# Customer Behavioral & Advanced Metrics
#############################################
# Some advanced metrics (churn, retention, RFM, etc.) can still be computed at the order level.


def customer_lifecycle_stages(order_df, recency_days=30, frequency_threshold=2):
    """
    Identify lifecycle stages of customers based on recency and frequency.
    This can be done with order-level data alone.
    """
    logger.info("Determining customer lifecycle stages...")
    if "Date" not in order_df.columns or "Client" not in order_df.columns:
        logger.error("Date and Client columns are required.")
        return pd.DataFrame()

    order_df["Date"] = pd.to_datetime(order_df["Date"])
    now = order_df["Date"].max()
    cust_agg = order_df.groupby("Client").agg(first_purchase_date=("Date", "min"), last_purchase_date=("Date", "max"), total_orders=("Référence", "count")).reset_index()

    cust_agg["days_since_last_purchase"] = (now - cust_agg["last_purchase_date"]).dt.days
    # Define conditions
    conditions = [
        (cust_agg["total_orders"] == 1) & (cust_agg["days_since_last_purchase"] <= recency_days),
        (cust_agg["total_orders"] >= frequency_threshold) & (cust_agg["days_since_last_purchase"] <= recency_days),
        (cust_agg["days_since_last_purchase"] > recency_days),
    ]
    choices = ["New", "Active", "At-Risk"]

    cust_agg["stage"] = np.select(conditions, choices, default="At-Risk")
    return cust_agg


def net_revenue_retention(order_df, date_col="Date", client_col="Client", total_col="Total", freq="M"):
    """
    Compute Net Revenue Retention (NRR) at cohort level.
    Requires multiple months of data and tracking cohorts over time.
    If your dataset supports it, this can be done with just orders.
    """
    logger.info("Calculating Net Revenue Retention (NRR)...")
    if client_col not in order_df.columns or date_col not in order_df.columns or total_col not in order_df.columns:
        logger.error("Required columns not found for NRR calculation.")
        return pd.DataFrame()

    order_df[date_col] = pd.to_datetime(order_df[date_col])
    first_purchase = order_df.groupby(client_col)[date_col].min().dt.to_period(freq)
    order_df["Cohort"] = order_df[client_col].map(first_purchase)
    order_df["OrderPeriod"] = order_df[date_col].dt.to_period(freq)

    cohort_revenue = order_df.groupby(["Cohort", "OrderPeriod"])[total_col].sum().reset_index()
    cohort_pivot = cohort_revenue.pivot(index="Cohort", columns="OrderPeriod", values=total_col)
    base_revenue = cohort_pivot.iloc[:, 0] if cohort_pivot.shape[1] > 0 else pd.Series()
    if base_revenue.empty:
        logger.warning("Not enough data for NRR calculation.")
        return pd.DataFrame()

    nrr = cohort_pivot.divide(base_revenue, axis=0) * 100
    return nrr


#############################################
# If you have additional metrics (churn, CLV, etc.) you can implement them here
# Many of these can be derived from order-level data.
#############################################


def churn_rate(order_df, date_col="Date", client_col="Client", freq="M"):
    """
    Compute a simple churn rate based on month-over-month retention.
    This requires identifying how many customers in one month did not purchase again the next month.
    """
    logger.info("Computing churn rate...")
    if date_col not in order_df.columns or client_col not in order_df.columns:
        logger.error("Date or Client column not found for churn analysis.")
        return pd.Series(dtype=float)

    order_df[date_col] = pd.to_datetime(order_df[date_col])
    order_df["OrderMonth"] = order_df[date_col].dt.to_period(freq)
    month_groups = order_df.groupby("OrderMonth")[client_col].apply(set)

    months = month_groups.index.sort_values()
    churn_rates = pd.Series(dtype=float)
    for i in range(1, len(months)):
        prev_set = month_groups.loc[months[i - 1]]
        curr_set = month_groups.loc[months[i]]
        if len(prev_set) > 0:
            churn = len(prev_set - curr_set) / len(prev_set)
        else:
            churn = np.nan
        churn_rates[months[i].start_time] = churn * 100

    return churn_rates


def calculate_clv(order_df, total_col="Total", client_col="Client", date_col="Date"):
    """
    Calculate a simple Customer Lifetime Value (CLV) approximation:
    CLV = Average Order Value * Purchase Frequency * Average Customer Lifetime (assume a fixed lifetime, e.g. 12 months)
    This is a rough estimate without product-level detail.
    """
    logger.info("Calculating CLV...")
    if total_col not in order_df.columns or client_col not in order_df.columns or date_col not in order_df.columns:
        logger.error("Required columns not found for CLV calculation.")
        return np.nan
    try:
        aov = order_df[total_col].mean()
        purchase_freq = order_df.groupby(client_col).size().mean()
        # Assume 12-month lifetime
        lifetime_months = 12
        clv = aov * purchase_freq * lifetime_months
        return clv
    except Exception as e:
        logger.error(f"Error calculating CLV: {e}")
        return np.nan


##############################################################
# MAIN FOR TESTING
##############################################################
if __name__ == "__main__":
    logger.info("Starting metrics computation test...")
    products_df = load_product_data("/path/to/5octobre_products.csv")
    cart_df, order_df = load_cleaned_data()

    if order_df is not None and not order_df.empty:
        # Compute some basic metrics
        kpis = basic_kpis(order_df)
        logger.info(f"Basic KPIs: {kpis}")

        monthly_revenue = compute_revenue_over_time(order_df, freq="M")
        logger.info(f"Monthly Revenue:\n{monthly_revenue}")

        aov = compute_average_order_value(order_df)
        logger.info(f"Average Order Value: {aov}")

        cart_abandon_rate = compute_cart_abandonment_rate(cart_df, order_df)
        logger.info(f"Cart Abandonment Rate: {cart_abandon_rate}%")

        monthly_customers = analyze_customer_count(order_df)
        logger.info(f"Monthly Unique Customers:\n{monthly_customers}")

        # Lifecycle stages
        lifecycle_df = customer_lifecycle_stages(order_df)
        logger.info(f"Customer Lifecycle Stages:\n{lifecycle_df.head()}")

        # Churn Rate
        churn = churn_rate(order_df)
        logger.info(f"Monthly Churn Rates:\n{churn}")

        # CLV
        clv = calculate_clv(order_df)
        logger.info(f"Estimated CLV: {clv}")

    else:
        logger.warning("No order data available for metric computations.")
