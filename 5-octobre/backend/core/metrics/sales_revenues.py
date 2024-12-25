"""
Ventes et revenus
------------------------------------
- CA (journalier, mensuel, cumul annuel)
- Nombre de commandes
- Average Order Value (AOV)
- Taux de conversion (via GA)
- Marge brute globale
"""

from asyncio import shield
import pandas as pd
import sys

sys.path.append(
    "/Users/benpfeffer/Library/Mobile Documents/com~apple~CloudDocs/projects-portfolio-main/5-octobre/"
)

from backend.data_preprocessing.pipeline import run_pipeline


def calculate_daily_revenue(df_retail):
    """Calculates the daily revenue."""
    df_retail["date"] = pd.to_datetime(df_retail["date"])
    daily_revenue = df_retail.groupby(df_retail["date"].dt.date)["pv_ttc"].sum()
    return daily_revenue


def calculate_monthly_revenue(df_retail):
    """Calculates the monthly revenue."""
    df_retail["date"] = pd.to_datetime(df_retail["date"])
    monthly_revenue = df_retail.groupby(df_retail["date"].dt.to_period("M"))[
        "pv_ttc"
    ].sum()
    return monthly_revenue


def calculate_yearly_revenue(df_retail):
    """Calculates the yearly revenue."""
    df_retail["date"] = pd.to_datetime(df_retail["date"])
    yearly_revenue = df_retail.groupby(df_retail["date"].dt.to_period("Y"))[
        "pv_ttc"
    ].sum()
    return yearly_revenue


def calculate_number_of_orders(df_retail):
    """Calculates the number of orders."""
    # Assuming each row in retail.csv represents an order item,
    # we can count the number of unique transaction references as the number of orders.
    number_of_orders = df_retail["ref"].nunique()
    return number_of_orders


def calculate_average_order_value(df_retail):
    """Calculates the average order value."""
    total_revenue = df_retail["pv_ttc"].sum()
    num_orders = calculate_number_of_orders(df_retail)
    if num_orders > 0:
        aov = total_revenue / num_orders
    else:
        aov = 0
    return aov


def calculate_gross_profit_margin(df_retail, df_inventory):
    """Calculates the gross profit margin."""
    # Assuming 'libellé' in df_retail corresponds to product names in df_inventory
    merged_df = pd.merge(
        df_retail, df_inventory, left_on="libellé", right_on="lib", how="left"
    )
    merged_df["cost"] = merged_df["factory_price"] * merged_df["qty"]
    merged_df["revenue"] = merged_df["retail"]
    total_revenue = merged_df["revenue"].sum()
    total_cost = merged_df["cost"].sum()
    if total_revenue > 0:
        gross_profit_margin = ((total_revenue - total_cost) / total_revenue) * 100
    else:
        gross_profit_margin = 0
    return gross_profit_margin


if __name__ == "__main__":
    cart_df, order_df, inventory_df, retail_df = run_pipeline()

    # Start Generation Here
    import os

    # Calculate revenues
    daily_revenue = calculate_daily_revenue(retail_df)
    monthly_revenue = calculate_monthly_revenue(retail_df)
    yearly_revenue = calculate_yearly_revenue(retail_df)
    gross_profit_margin = calculate_gross_profit_margin(retail_df, inventory_df)
    # Define the output directory
    output_dir = "data/analysis/sales_revenues/revenues/"
    os.makedirs(output_dir, exist_ok=True)

    # # Save the results to separate CSV files
    # daily_revenue.to_csv(os.path.join(output_dir, "daily_revenue.csv"))
    # monthly_revenue.to_csv(os.path.join(output_dir, "monthly_revenue.csv"))
    # yearly_revenue.to_csv(os.path.join(output_dir, "yearly_revenue.csv"))
    # print(gross_profit_margin)

    # print(f"[INFO] Daily, monthly, and yearly revenue data saved to {output_dir}")
