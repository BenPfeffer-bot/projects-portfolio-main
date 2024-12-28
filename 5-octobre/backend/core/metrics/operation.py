"""
Sales and Revenue Analysis
------------------------------------
- Daily, Monthly, and Yearly Revenue
- Number of Orders
- Average Order Value (AOV)
- Conversion Rate (via GA) -- to be integrated later
- Global Gross Profit Margin
"""

import sys
import os
import logging
import pandas as pd

# Add custom library path if needed
sys.path.append(
    "/Users/benpfeffer/Library/Mobile Documents/com~apple~CloudDocs/projects-portfolio-main/5-octobre/"
)

from backend.data_preprocessing.pipeline import run_pipeline


def calculate_daily_revenue(df_retail: pd.DataFrame) -> pd.Series:
    """
    Calculates the daily revenue by summing 'pv_ttc' for each day.

    Args:
        df_retail (pd.DataFrame): Retail transactions DataFrame.
                                  Must contain 'date' and 'pv_ttc' columns.

    Returns:
        pd.Series: A time series (daily) of total revenue.
    """
    df_retail["date"] = pd.to_datetime(df_retail["date"])
    daily_revenue = df_retail.groupby(df_retail["date"].dt.date)["pv_ttc"].sum()
    return daily_revenue


def calculate_monthly_revenue(df_retail: pd.DataFrame) -> pd.Series:
    """
    Calculates the monthly revenue by summing 'pv_ttc' for each month.

    Args:
        df_retail (pd.DataFrame): Retail transactions DataFrame.
                                  Must contain 'date' and 'pv_ttc' columns.

    Returns:
        pd.Series: A time series (monthly) of total revenue, indexed by year-month.
    """
    df_retail["date"] = pd.to_datetime(df_retail["date"])
    monthly_revenue = df_retail.groupby(df_retail["date"].dt.to_period("M"))[
        "pv_ttc"
    ].sum()
    return monthly_revenue


def calculate_yearly_revenue(df_retail: pd.DataFrame) -> pd.Series:
    """
    Calculates the yearly revenue by summing 'pv_ttc' for each year.

    Args:
        df_retail (pd.DataFrame): Retail transactions DataFrame.
                                  Must contain 'date' and 'pv_ttc' columns.

    Returns:
        pd.Series: A time series (yearly) of total revenue, indexed by year.
    """
    df_retail["date"] = pd.to_datetime(df_retail["date"])
    yearly_revenue = df_retail.groupby(df_retail["date"].dt.to_period("Y"))[
        "pv_ttc"
    ].sum()
    return yearly_revenue


def calculate_number_of_orders(df_retail: pd.DataFrame) -> int:
    """
    Calculates the number of unique orders in the retail DataFrame.
    Assumes each row represents a line item and 'ref' is a unique order identifier.

    Args:
        df_retail (pd.DataFrame): Retail transactions DataFrame.
                                  Must contain 'ref' column representing unique order references.

    Returns:
        int: The total number of unique orders.
    """
    return df_retail["ref"].nunique()


def calculate_average_order_value(df_retail: pd.DataFrame) -> float:
    """
    Calculates the Average Order Value (AOV) by dividing total revenue ('pv_ttc')
    by the number of unique orders (based on 'ref').

    Args:
        df_retail (pd.DataFrame): Retail transactions DataFrame.
                                  Must contain 'pv_ttc' (revenue) and 'ref' (order reference).

    Returns:
        float: The Average Order Value. Returns 0 if there are no orders.
    """
    total_revenue = df_retail["pv_ttc"].sum()
    num_orders = calculate_number_of_orders(df_retail)

    if num_orders > 0:
        return total_revenue / num_orders
    return 0.0


def calculate_gross_profit_margin(
    df_retail: pd.DataFrame, df_inventory: pd.DataFrame
) -> float:
    """
    Calculates the gross profit margin using merged data from retail and inventory.
    Assumes:
      - 'libellé' in df_retail corresponds to 'lib' in df_inventory (for matching products).
      - 'factory_price' (cost) and 'qty' (quantity) exist in df_inventory.
      - 'retail' in df_inventory is the retail price per unit,
        which corresponds to 'pv_ttc' in df_retail if needed.

    Args:
        df_retail (pd.DataFrame): Retail transactions DataFrame.
        df_inventory (pd.DataFrame): Inventory DataFrame with factory_price, qty, retail columns.

    Returns:
        float: The gross profit margin percentage. Returns 0 if total_revenue is zero or
               if there's an error in merging/columns.
    """
    try:
        merged_df = pd.merge(
            df_retail, df_inventory, left_on="libellé", right_on="lib", how="left"
        )
        # Cost: factory_price * qty sold
        merged_df["cost"] = merged_df["factory_price"] * merged_df["qty"]
        # Revenue: retail price per unit * quantity
        merged_df["revenue"] = merged_df["retail"] * merged_df["qty"]

        total_revenue = merged_df["revenue"].sum()
        total_cost = merged_df["cost"].sum()

        if total_revenue > 0:
            return ((total_revenue - total_cost) / total_revenue) * 100
        return 0.0
    except KeyError as e:
        logging.error(f"Key error while calculating gross profit margin: {e}")
        return 0.0
    except Exception as e:
        logging.error(f"Error calculating gross profit margin: {e}")
        return 0.0


def calculate_avg_order_value(order_df: pd.DataFrame) -> float:
    """
    Calculates the average order value (AOV) specifically from the 'order' DataFrame,
    which may differ from the 'retail' DataFrame.

    Args:
        order_df (pd.DataFrame): The order DataFrame. Must contain 'total' column.

    Returns:
        float: The average order value.
    """
    try:
        total_revenue = order_df["total"].sum()
        nb_sales = order_df["total"].count()

        logging.info(f"Total revenue: {total_revenue}")
        logging.info(f"Number of sales: {nb_sales}")

        if nb_sales == 0:
            logging.warning("No sales found in the order dataframe.")
            return 0.0

        aov = total_revenue / nb_sales
        logging.info(f"Average Order Value: {aov}")
        return aov

    except ZeroDivisionError:
        logging.error("No sales found in the order dataframe (division by zero).")
        return 0.0
    except Exception as e:
        logging.error(f"An error occurred while calculating AOV: {e}")
        return 0.0


def calculate_inventory_turnover(inventory_df: pd.DataFrame) -> str:
    """
    Calculates and interprets the inventory turnover ratio.

    Inventory Turnover Ratio = Cost of Goods Sold (COGS) / Average Inventory Value.

    Args:
        inventory_df (pd.DataFrame): Must contain 'factory_price' and 'qty' for calculating total cost,
                                     and 'retail' to approximate inventory value.

    Returns:
        str: A human-readable interpretation of the inventory turnover ratio.
    """
    try:
        inventory_df["total_cost"] = inventory_df["factory_price"] * inventory_df["qty"]
        cogs = inventory_df["total_cost"].sum()
        logging.info(f"Cost of Goods Sold (COGS): {cogs}")

        average_inventory_value = inventory_df["retail"].mean()
        logging.info(f"Average Inventory Value: {average_inventory_value}")

        if average_inventory_value == 0:
            logging.warning(
                "Average inventory value is zero; cannot calculate turnover ratio."
            )
            return (
                "Cannot calculate inventory turnover: Average inventory value is zero."
            )

        inventory_turnover = cogs / average_inventory_value
        logging.info(f"Inventory Turnover Ratio: {inventory_turnover:.2f}")

        interpretation = f"Inventory Turnover Ratio: {inventory_turnover:.2f}. "
        if inventory_turnover > 5:
            interpretation += (
                "This indicates a healthy turnover, suggesting efficient inventory management "
                "and strong sales."
            )
        elif 2 <= inventory_turnover <= 5:
            interpretation += "This suggests a reasonable turnover rate."
        else:
            interpretation += (
                "This relatively low turnover might indicate overstocking or slow-moving inventory. "
                "Further investigation is recommended."
            )

        return interpretation

    except KeyError as e:
        logging.error(f"Column '{e}' not found in the inventory DataFrame.")
        return f"Error calculating inventory turnover: Column '{e}' not found."
    except Exception as e:
        logging.error(f"An error occurred while calculating inventory turnover: {e}")
        return f"An error occurred while calculating inventory turnover: {e}"


def calculate_cart_abandonment_rate(cart_df: pd.DataFrame) -> float:
    """
    Calculates the cart abandonment rate.

    Formula: (Initiated Carts - Completed Carts) / Initiated Carts * 100

    Args:
        cart_df (pd.DataFrame): DataFrame containing cart data,
                                where one of the columns indicates cart status.
                                Assumes 'Panier abandonné' denotes an abandoned cart.

    Returns:
        float: The cart abandonment rate as a percentage.
               Returns 0.0 if no carts are found or if an error occurs.
    """
    try:
        if cart_df is None or cart_df.empty:
            logging.warning("Cart DataFrame is empty or None.")
            return 0.0

        total_initiated_carts = len(cart_df)
        abandoned_carts = cart_df[
            cart_df.iloc[:, 1].str.contains("Panier abandonné", na=False)
        ].shape[0]

        if total_initiated_carts > 0:
            return (abandoned_carts / total_initiated_carts) * 100
        return 0.0

    except Exception as e:
        logging.error(f"Error calculating cart abandonment rate: {e}")
        return 0.0


import logging
import pandas as pd


def refund_rate(orders: pd.DataFrame) -> float:
    """
    Calculates the refund (or return) rate.

    Refund Rate = (Number of Refunded/Returned Orders / Total Orders) * 100

    Assumptions:
      - 'état' column in the orders DataFrame indicates order status.
      - Any status different from "Livré" is counted as refunded/returned.

    Args:
        orders (pd.DataFrame): DataFrame containing order information, must have an 'état' column.

    Returns:
        float: The refund/return rate as a percentage. Returns 0.0 if the DataFrame is empty or
               all rows have a "Livré" status.
    """
    try:
        if orders is None or orders.empty:
            logging.warning("Orders DataFrame is empty or None.")
            return 0.0

        total_orders = len(orders)
        if total_orders == 0:
            return 0.0

        # Count orders that are not 'Livré'
        total_refunded = orders[orders["état"] != "Livré"].shape[0]
        return (total_refunded / total_orders) * 100

    except KeyError as e:
        logging.error(f"KeyError: Column '{e}' not found in the orders DataFrame.")
        return 0.0
    except Exception as e:
        logging.error(f"An error occurred while calculating refund_rate: {e}")
        return 0.0


def calculate_stock_out_rate(inventory_df: pd.DataFrame) -> float:
    """
    Calculates the stock-out rate.

    Stock-Out Rate = (Number of Products Out of Stock / Total Active Products) * 100

    Assumptions:
      - 'qty' column indicates the quantity in stock for each product.
      - Products with qty <= 0 are considered out of stock.

    Args:
        inventory_df (pd.DataFrame): Inventory DataFrame with a 'qty' column.

    Returns:
        float: The stock-out rate as a percentage. Returns 0.0 if there are no products or
               if the DataFrame is empty.
    """
    try:
        if inventory_df is None or inventory_df.empty:
            logging.warning("Inventory DataFrame is empty or None.")
            return 0.0

        total_active_products = len(inventory_df)
        if total_active_products == 0:
            return 0.0

        out_of_stock_products = inventory_df[inventory_df["qty"] <= 0].shape[0]
        return (out_of_stock_products / total_active_products) * 100

    except KeyError as e:
        logging.error(f"KeyError: Column '{e}' not found in the inventory DataFrame.")
        return 0.0
    except Exception as e:
        logging.error(f"An error occurred while calculating stock-out rate: {e}")
        return 0.0


def calculate_fill_rate(order_df: pd.DataFrame) -> float:
    """
    Calculates the fill rate.

    Fill Rate = (Number of Items Shipped on Time / Total Items Ordered) * 100

    Assumptions:
      - 'état' column indicates whether an order item is "Livré" (shipped on time)
        or some other status.

    Args:
        order_df (pd.DataFrame): DataFrame of order items, must include an 'état' column.

    Returns:
        float: The fill rate as a percentage. Returns 0.0 if there are no orders.
    """
    try:
        if order_df is None or order_df.empty:
            logging.warning("Order DataFrame is empty or None.")
            return 0.0

        total_items_ordered = len(order_df)
        if total_items_ordered == 0:
            return 0.0

        items_shipped_on_time = order_df[order_df["état"] == "Livré"].shape[0]
        return (items_shipped_on_time / total_items_ordered) * 100

    except KeyError as e:
        logging.error(f"KeyError: Column '{e}' not found in the order DataFrame.")
        return 0.0
    except Exception as e:
        logging.error(f"An error occurred while calculating fill rate: {e}")
        return 0.0


def calculate_contribution_margin(inventory_df: pd.DataFrame) -> pd.Series:
    """
    Calculates the contribution margin per product.

    Contribution Margin (per unit) = Retail Price per Unit - Factory Price per Unit

    Assumptions:
      - 'retail' column: retail price per unit.
      - 'factory_price' column: variable (direct) cost per unit.

    Args:
        inventory_df (pd.DataFrame): Inventory DataFrame containing 'retail'
                                     and 'factory_price' columns.

    Returns:
        pd.Series: A Series of contribution margins for each product,
                   indexed by the original DataFrame's index.
    """
    try:
        if inventory_df is None or inventory_df.empty:
            logging.warning("Inventory DataFrame is empty or None.")
            return pd.Series(dtype=float)

        # Calculate contribution margin for each row
        return inventory_df["retail"] - inventory_df["factory_price"]

    except KeyError as e:
        logging.error(f"KeyError: Column '{e}' not found in the inventory DataFrame.")
        return pd.Series(dtype=float)
    except Exception as e:
        logging.error(f"An error occurred while calculating contribution margin: {e}")
        return pd.Series(dtype=float)


def calculate_sales_velocity(
    retail_df: pd.DataFrame, start_date: str, end_date: str
) -> float:
    """
    Calculates the sales velocity (units per day) over a given time period.

    Sales Velocity = Total Units Sold in the Period / Length of the Period (Days)

    Assumptions:
      - 'date' column in retail_df can be converted to datetime.
      - 'Qté' column indicates the quantity sold.

    Args:
        retail_df (pd.DataFrame): Retail DataFrame containing 'date' and 'Qté' columns.
        start_date (str): Start date of the period in 'YYYY-MM-DD' format.
        end_date (str): End date of the period in 'YYYY-MM-DD' format.

    Returns:
        float: The average units sold per day over the specified period.
               Returns 0.0 if the period length is zero or the DataFrame is empty.
    """
    try:
        if retail_df is None or retail_df.empty:
            logging.warning("Retail DataFrame is empty or None.")
            return 0.0

        retail_df["date"] = pd.to_datetime(retail_df["date"], errors="coerce")
        if retail_df["date"].isnull().all():
            logging.error("No valid dates found in the 'date' column.")
            return 0.0

        # Filter rows based on the input date range
        period_df = retail_df[
            (retail_df["date"] >= start_date) & (retail_df["date"] <= end_date)
        ]
        total_units_sold = period_df["qté"].sum()

        # Calculate the number of days in the period
        period_length = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
        if period_length > 0:
            return total_units_sold / period_length
        return 0.0

    except KeyError as e:
        logging.error(f"KeyError: Column '{e}' not found in the retail DataFrame.")
        return 0.0
    except Exception as e:
        logging.error(f"An error occurred while calculating sales velocity: {e}")
        return 0.0


import logging
import pandas as pd


def calculate_clv(order_df: pd.DataFrame) -> float:
    """
    Calculates the Customer Lifetime Value (CLV) using a simplified model.

    Formula:
        CLV = AOV * (Purchase Frequency per Year) * (Customer Lifespan in Years)

    Args:
        aov (float): Average Order Value.
        purchase_frequency (float): Average number of purchases per year by a typical customer.
        customer_lifespan (float): Average customer lifespan in years.

    Returns:
        float: The Customer Lifetime Value. Returns 0.0 if any argument is non-positive.
    """
    order_df = order_df.copy()
    aov = calculate_avg_order_value(order_df)
    purchase_frequency = calculate_repeat_purchase_rate(order_df)
    customer_lifespan = 10
    try:
        if aov <= 0 or purchase_frequency <= 0 or customer_lifespan <= 0:
            logging.warning("One of the input parameters is non-positive.")
            return 0.0
        return aov * purchase_frequency * customer_lifespan
    except Exception as e:
        logging.error(f"Error calculating CLV: {e}")
        return 0.0


def calculate_cac(order_df: pd.DataFrame) -> float:
    """
    Calculates the Customer Acquisition Cost (CAC).

    Formula:
        CAC = Total Marketing & Sales Expense / Number of New Customers Acquired

    Args:
        total_marketing_expense (float): Total marketing and sales expense.
        new_customers (int): Number of new customers acquired in the period.

    Returns:
        float: The Customer Acquisition Cost. Returns 0.0 if new_customers is zero or negative.
    """
    order_df = order_df.copy()
    total_marketing_expense = 10000
    new_customers = order_df["client"].nunique()
    try:
        if new_customers <= 0:
            logging.warning(
                "Cannot calculate CAC because the number of new customers is zero or negative."
            )
            return 0.0
        return total_marketing_expense / new_customers
    except Exception as e:
        logging.error(f"Error calculating CAC: {e}")
        return 0.0


def calculate_rfm_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assigns an RFM segment to each row based on recency, frequency, and monetary scores.

    Example Segmentation Logic:
        1. Compute an 'RFM_Score' as the sum of 'recency', 'frequency', and 'monetary'.
        2. Use pd.qcut() to segment the 'RFM_Score' into quartiles: ['Low', 'Medium', 'High', 'Very High'].

    Assumptions:
        - df contains 'recency', 'frequency', and 'monetary' columns.

    Args:
        df (pd.DataFrame): DataFrame containing customer purchase data with 'recency',
                           'frequency', and 'monetary' columns.

    Returns:
        pd.DataFrame: A copy of the original DataFrame with additional columns:
            - 'RFM_Score': The sum of recency, frequency, and monetary.
            - 'Segment': The categorical segment based on quartiles.
        Returns an empty DataFrame if input is invalid or missing columns.
    """
    try:
        if df is None or df.empty:
            logging.warning("The input DataFrame is empty or None.")
            return pd.DataFrame()

        # Verify required columns
        required_cols = {"recency", "frequency", "monetary"}
        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            logging.error(f"Missing columns for RFM analysis: {missing}")
            return pd.DataFrame()

        rfm_df = df.copy()
        rfm_df["RFM_Score"] = (
            rfm_df["recency"] + rfm_df["frequency"] + rfm_df["monetary"]
        )

        # Segment by quartiles
        rfm_df["Segment"] = pd.qcut(
            rfm_df["RFM_Score"], q=4, labels=["Low", "Medium", "High", "Very High"]
        )
        return rfm_df

    except Exception as e:
        logging.error(f"Error calculating RFM distribution: {e}")
        return pd.DataFrame()


def calculate_repeat_purchase_rate(order_df: pd.DataFrame) -> float:
    """
    Calculates the Repeat Purchase Rate (RPR).

    Formula:
        RPR = (Number of Returning Customers / Total Number of Customers) * 100

    Args:
        returning_customers (int): Number of returning customers in a given period.
        total_customers (int): Total number of customers in that period.

    Returns:
        float: The Repeat Purchase Rate as a percentage. Returns 0.0 if total_customers is zero or negative.
    """
    order_customers = order_df.copy()
    order_customers["returning_customer"] = 0
    order_customers["returning_customer"] = order_customers["nouveau_client"].apply(
        lambda x: 1 if x == 0 else 0
    )
    returning_customers = order_customers["returning_customer"].sum()
    total_customers = order_customers["client"].nunique()

    try:
        if total_customers <= 0:
            logging.warning(
                "Total customers is zero or negative; cannot calculate RPR."
            )
            return 0.0
        return (returning_customers / total_customers) * 100
    except Exception as e:
        logging.error(f"Error calculating repeat purchase rate: {e}")
        return 0.0


def calculate_on_time_delivery_rate(orders_df: pd.DataFrame) -> float:
    """
    Calculates the On-Time Delivery Rate.

    Formula:
        On-Time Delivery Rate =
            (Number of Orders Delivered on or Before Estimated Date / Total Orders Shipped) * 100

    Assumptions:
        - orders_df contains 'delivery_date' and 'estimated_delivery_date' columns.

    Args:
        orders_df (pd.DataFrame): DataFrame with 'delivery_date' and 'estimated_delivery_date' columns.

    Returns:
        float: The On-Time Delivery Rate as a percentage. Returns 0.0 if the DataFrame is empty or
               if the columns are missing.
    """
    try:
        if orders_df is None or orders_df.empty:
            logging.warning("Orders DataFrame is empty or None.")
            return 0.0

        required_cols = {"delivery_date", "estimated_delivery_date"}
        if not required_cols.issubset(orders_df.columns):
            missing_cols = required_cols - set(orders_df.columns)
            logging.error(
                f"Missing columns for on-time delivery calculation: {missing_cols}"
            )
            return 0.0

        # Ensure columns are date-compatible
        orders_df["delivery_date"] = pd.to_datetime(
            orders_df["delivery_date"], errors="coerce"
        )
        orders_df["estimated_delivery_date"] = pd.to_datetime(
            orders_df["estimated_delivery_date"], errors="coerce"
        )

        # Filter out rows where dates are not valid
        valid_orders = orders_df.dropna(
            subset=["delivery_date", "estimated_delivery_date"]
        )
        total_orders_shipped = len(valid_orders)
        if total_orders_shipped == 0:
            logging.warning(
                "No valid shipping data to calculate on-time delivery rate."
            )
            return 0.0

        on_time_deliveries = valid_orders[
            valid_orders["delivery_date"] <= valid_orders["estimated_delivery_date"]
        ].shape[0]

        return (on_time_deliveries / total_orders_shipped) * 100
    except Exception as e:
        logging.error(f"Error calculating on-time delivery rate: {e}")
        return 0.0


if __name__ == "__main__":
    # Run your data pipeline to retrieve the necessary DataFrames
    cart_df, order_df, inventory_df, retail_df = run_pipeline()

    # Calculate various metrics
    daily_revenue = calculate_daily_revenue(retail_df)
    monthly_revenue = calculate_monthly_revenue(retail_df)
    yearly_revenue = calculate_yearly_revenue(retail_df)
    gross_profit_margin = calculate_gross_profit_margin(retail_df, inventory_df)
    aov_orders_table = calculate_avg_order_value(order_df)  # from 'order.csv'
    inventory_turnover_result = calculate_inventory_turnover(inventory_df)
    cart_abandonment_rate = calculate_cart_abandonment_rate(cart_df)

    # Example of printing or logging results
    logging.info(f"Daily Revenue:\n{daily_revenue}\n")
    logging.info(f"Monthly Revenue:\n{monthly_revenue}\n")
    logging.info(f"Yearly Revenue:\n{yearly_revenue}\n")
    logging.info(f"Gross Profit Margin: {gross_profit_margin:.2f}%")
    logging.info(f"AOV (Orders Table): {aov_orders_table:.2f}")
    logging.info(inventory_turnover_result)
    logging.info(f"Cart Abandonment Rate: {cart_abandonment_rate:.2f}%")

    print("[INFO] Cart Abandonment Rate:", cart_abandonment_rate)
    print("[INFO] Daily Revenue:", daily_revenue)
    print("[INFO] Monthly Revenue:", monthly_revenue)
    print("[INFO] Yearly Revenue:", yearly_revenue)
    print("[INFO] Gross Profit Margin:", gross_profit_margin)
    print("[INFO] Average Order Value:", aov_orders_table)
    print("[INFO] Inventory Turnover:", inventory_turnover_result)
    # Calculate additional metrics
    refund_rate_result = refund_rate(order_df)
    stock_out_rate = calculate_stock_out_rate(inventory_df)
    fill_rate = calculate_fill_rate(order_df)
    contribution_margins = calculate_contribution_margin(inventory_df)
    sales_velocity = calculate_sales_velocity(retail_df, "2023-01-01", "2023-12-31")

    # Log additional metrics
    logging.info(f"Refund Rate: {refund_rate_result:.2f}%")
    logging.info(f"Stock-Out Rate: {stock_out_rate:.2f}%")
    logging.info(f"Fill Rate: {fill_rate:.2f}%")
    logging.info(f"Average Contribution Margin: {contribution_margins.mean():.2f}")
    logging.info(f"Sales Velocity (units/day): {sales_velocity:.2f}")

    # Print additional metrics
    print("[INFO] Refund Rate:", refund_rate_result)
    print("[INFO] Stock-Out Rate:", stock_out_rate)
    print("[INFO] Fill Rate:", fill_rate)
    print("[INFO] Average Contribution Margin:", contribution_margins.mean())
    print("[INFO] Sales Velocity (units/day):", sales_velocity)

    print(calculate_on_time_delivery_rate(order_df))
    print(calculate_repeat_purchase_rate(order_df))
    print(calculate_clv(order_df))
    print(calculate_cac(order_df))
    print(calculate_rfm_distribution(order_df))

    # Example of exporting to CSV if needed
    # output_dir = "data/analysis/sales_revenues/revenues/"
    # os.makedirs(output_dir, exist_ok=True)
    # daily_revenue.to_csv(os.path.join(output_dir, "daily_revenue.csv"), index=True)
    # monthly_revenue.to_csv(os.path.join(output_dir, "monthly_revenue.csv"), index=True)
    # yearly_revenue.to_csv(os.path.join(output_dir, "yearly_revenue.csv"), index=True)

    # Confirm end of script
    logging.info("Metrics calculation completed successfully.")
