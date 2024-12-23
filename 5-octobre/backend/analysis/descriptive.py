# descriptive_kpi_analysis.py
"""
Descriptive Analysis & KPI Calculation for 5octobre (Jewelry Company)

Datasets:
  1) cart.csv      (abandoned and completed carts)
  2) inventory.csv (product inventory with cost & retail prices)
  3) order.csv     (online orders, new vs returning customers, etc.)
  4) products.csv  (master product info: name, price)
  5) retail.csv    (offline/retail transactions)
  6) Google Analytics (not a CSV, but can be used for web metrics/conversion)

Objectives:
  - Link cart -> order for real cart abandonment.
  - Compute sales & revenue metrics, new vs returning, top products.
  - Merge order -> inventory -> products for margin & advanced analysis.
  - Inventory turnover with a simplified COGS approach.
  - Offline vs. online synergy comparison (retail.csv vs. order.csv).
  - (Potential) Incorporate Google Analytics metrics for conversion if integrated.

Usage:
  python descriptive_kpi_analysis.py

Author: Your Name
"""

import os
import pandas as pd

# ------------------------------------------------------------------------------
#                               0. CONFIGURATION
# ------------------------------------------------------------------------------
BASE_DIR = "/Users/benpfeffer/Library/Mobile Documents/com~apple~CloudDocs/projects-portfolio-main/5-octobre/output"
EDA_PLOTS_DIR = os.path.join(BASE_DIR, "fig2")
os.makedirs(EDA_PLOTS_DIR, exist_ok=True)

import sys

sys.path.append(
    "/Users/benpfeffer/Library/Mobile Documents/com~apple~CloudDocs/projects-portfolio-main/5-octobre/"
)
# Import preprocessed data from pipeline

from backend.data_preprocessing.pipeline import run_pipeline


# Load cleaned data from preprocessing pipeline
def load_cleaned_data() -> (
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
):
    """
    Load cleaned DataFrames from the preprocessing pipeline.
    Returns tuple of (cart_df, order_df, inventory_df, retail_df)
    """
    try:
        cart_df, order_df, inventory_df, retail_df = run_pipeline()
        if any(df is None for df in [cart_df, order_df, inventory_df, retail_df]):
            print("[ERROR] One or more DataFrames failed to load")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        print("[INFO] Successfully loaded all cleaned DataFrames")
        return cart_df, order_df, inventory_df, retail_df

    except Exception as e:
        print(f"[ERROR] Failed to load data from pipeline: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()


# ------------------------------------------------------------------------------
#                              2. MERGING / ENRICHMENT
# ------------------------------------------------------------------------------
def merge_cart_order(cart_df: pd.DataFrame, order_df: pd.DataFrame) -> pd.DataFrame:
    """
    Link cart rows to orders for advanced cart abandonment analysis.
    We assume:
      - 'cart.csv' has columns like 'ID commande' or 'id_commande'.
      - 'order.csv' has 'Référence' or 'reference'.
      - If they match, that indicates a cart that was converted to an order.
    """
    merged = cart_df.copy()

    # Example rename to unify columns if necessary
    if "ID commande" in cart_df.columns and "Référence" in order_df.columns:
        merged.rename(columns={"ID commande": "cart_ref"}, inplace=True)
        order_temp = order_df.rename(columns={"Référence": "order_ref"})

        # Merge on reference
        merged = pd.merge(
            merged,
            order_temp[["order_ref", "id", "Total"]],  # minimal columns for linking
            left_on="cart_ref",
            right_on="order_ref",
            how="left",
            suffixes=("_cart", "_ord"),
        )
    else:
        print(
            "[WARN] Cart -> Order reference columns not found. Abandonment analysis might be partial."
        )

    return merged


def enrich_order_data(
    order_df: pd.DataFrame, inventory_df: pd.DataFrame, products_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge order data with inventory (for cost/retail) and products (for product name, standard price).
    Adjust logic if your order data references items differently (e.g., 'ean', 'id_product').

    - inventory_df has columns: 'ean', 'qty', 'factory_price', 'retail', 'retail_us'
    - products_df has columns: 'id', 'name', 'price'
    - order_df might have 'ean' or 'id_product' to link.
    """
    merged = order_df.copy()

    # Example: If order_df has an 'ean' column that matches inventory
    if "ean" in merged.columns and "ean" in inventory_df.columns:
        merged = pd.merge(
            merged,
            inventory_df[["ean", "factory_price", "retail", "retail_us"]],
            on="ean",
            how="left",
        )

    # Example: If order_df has 'id_product' that matches products 'id'
    if "id_product" in merged.columns and "id" in products_df.columns:
        merged = pd.merge(
            merged,
            products_df[["id", "name", "price"]],
            left_on="id_product",
            right_on="id",
            how="left",
            suffixes=("_inv", "_prod"),
        )
    return merged


# ------------------------------------------------------------------------------
#                              3. KPI FUNCTIONS
# ------------------------------------------------------------------------------
def compute_cart_abandonment(merged_cart: pd.DataFrame) -> None:
    """
    Compute a more robust cart abandonment rate using the merged cart+order DataFrame.
    If 'order_ref' is not null, that cart converted to an order; otherwise it's abandoned.
    """
    if merged_cart.empty:
        print(
            "[WARN] Merged cart DataFrame is empty, skipping cart abandonment calculation."
        )
        return

    total_carts = len(merged_cart)
    # If 'order_ref' is present, that means it linked to an order
    purchased_carts = merged_cart["order_ref"].notnull().sum()
    abandoned_carts = total_carts - purchased_carts

    abandonment_rate = (abandoned_carts / total_carts) * 100 if total_carts else 0.0

    print("\n--- CART ABANDONMENT (LINKED) ---")
    print(f"Total carts: {total_carts}")
    print(f"Purchased carts: {purchased_carts}")
    print(f"Abandoned carts: {abandoned_carts}")
    print(f"Abandonment Rate: {abandonment_rate:.2f}%")


def compute_sales_metrics(order_df: pd.DataFrame) -> None:
    """
    Calculate total sales, number of orders, and average order value (AOV).
    Expects 'Total' as numeric, 'id' for order ID.
    """
    if order_df.empty or "Total" not in order_df.columns:
        print(
            "[WARN] Order data is empty or missing 'Total' column, skipping sales metrics."
        )
        return

    total_sales = order_df["Total"].sum()
    num_orders = order_df["id"].nunique() if "id" in order_df.columns else len(order_df)
    aov = total_sales / num_orders if num_orders else 0.0

    print("\n--- SALES METRICS ---")
    print(f"Total Sales: {total_sales:,.2f}")
    print(f"Number of Orders: {num_orders}")
    print(f"Average Order Value (AOV): {aov:,.2f}")


def compute_customer_metrics(order_df: pd.DataFrame) -> None:
    """
    Analyze new vs. returning customers, repeat purchase rate, etc.
    Expects columns like:
      'Nouveau client' (0/1)
      'Client' (customer name or ID for repeat checks)
    """
    if order_df.empty:
        print("[WARN] Order DataFrame empty, skipping customer metrics.")
        return

    # New vs returning
    if "Nouveau client" in order_df.columns:
        new_orders = (order_df["Nouveau client"] == 1).sum()
        total_orders = len(order_df)
        returning_orders = total_orders - new_orders
        pct_new = (new_orders / total_orders * 100) if total_orders else 0.0
        pct_returning = 100 - pct_new
    else:
        print(
            "[WARN] 'Nouveau client' column not found, cannot compute new vs returning."
        )
        new_orders = returning_orders = 0
        pct_new = pct_returning = 0.0

    # Repeat purchase rate (if 'Client' is a unique identifier)
    repeat_rate = 0.0
    if "Client" in order_df.columns:
        client_counts = order_df["Client"].value_counts()
        num_clients = len(client_counts)
        repeating_clients = (client_counts > 1).sum()  # # of clients w/ >1 orders
        repeat_rate = (repeating_clients / num_clients * 100) if num_clients else 0.0

    print("\n--- CUSTOMER METRICS ---")
    print(f"New Orders: {new_orders} ({pct_new:.2f}%)")
    print(f"Returning Orders: {returning_orders} ({pct_returning:.2f}%)")
    print(f"Repeat Purchase Rate (Client-based): {repeat_rate:.2f}%")


def compute_margin_analysis(enriched_orders: pd.DataFrame) -> None:
    """
    Compute margin from order & inventory cost.
    Expects columns like:
      - 'factory_price' for cost,
      - 'Total' for sales revenue (if each row ~ 1 item sold),
        or incorporate a 'quantity' if you have line-level detail.
    """
    if enriched_orders.empty:
        print("[WARN] enriched_orders is empty, skipping margin analysis.")
        return

    cost_col = "factory_price"
    revenue_col = "Total"

    if (
        cost_col not in enriched_orders.columns
        or revenue_col not in enriched_orders.columns
    ):
        print(
            "[WARN] Missing required columns for margin analysis (factory_price, Total)."
        )
        return

    # Fill NA with 0
    enriched_orders[cost_col] = enriched_orders[cost_col].fillna(0)
    enriched_orders[revenue_col] = enriched_orders[revenue_col].fillna(0)

    # If you had a quantity column (e.g., 'qty_sold'), you'd do margin = (revenue - cost) * qty_sold
    # For this example, each row might represent the entire order. We'll assume 1:1 logic.
    enriched_orders["margin_line"] = (
        enriched_orders[revenue_col] - enriched_orders[cost_col]
    )
    total_margin = enriched_orders["margin_line"].sum()
    avg_margin = enriched_orders["margin_line"].mean()
    margin_positive_count = (enriched_orders["margin_line"] > 0).sum()

    print("\n--- MARGIN ANALYSIS ---")
    print(f"Total Margin: {total_margin:,.2f}")
    print(f"Average Margin per Order Row: {avg_margin:,.2f}")
    print(f"Rows with Positive Margin: {margin_positive_count}/{len(enriched_orders)}")


def compute_inventory_turnover(
    inventory_df: pd.DataFrame, order_df: pd.DataFrame
) -> None:
    """
    Inventory turnover = COGS / Average Inventory (simplified).
    We'll assume we have a 'qty_sold' in order_df or a separate order-line to compute cost_of_goods_sold.
    If not present, we do a naive approach or skip.

    If we only have a snapshot of inventory (inventory.csv), we approximate:
      average_inventory_value = sum(qty * factory_price)
    If we can compute COGS from order data (factory_price * qty_sold), we do so.
    """
    if inventory_df.empty or order_df.empty:
        print(
            "[WARN] Missing inventory or order data, skipping advanced inventory turnover."
        )
        return

    # Suppose 'order.csv' does not have line-level quantity.
    # We'll skip a thorough approach. If you had it, you'd merge & sum factory_price * quantity_sold.

    # As a placeholder:
    # approximate cogs from the 'Total' in orders minus margin, or from merges with inventory.
    # For demonstration, let's just do a naive 10% of total sales, or skip. Real logic requires line-level detail.

    total_sales = order_df["Total"].sum() if "Total" in order_df.columns else 0
    # Let's assume cost_of_goods_sold ~ 50% of total sales (completely placeholder).
    cogs = 0.5 * total_sales

    # Average inventory value
    if "qty" in inventory_df.columns and "factory_price" in inventory_df.columns:
        inventory_df["inventory_value"] = (
            inventory_df["qty"] * inventory_df["factory_price"]
        )
        total_inventory_value = inventory_df["inventory_value"].sum()
    else:
        total_inventory_value = 0.0

    # If we had historical data for beginning & ending inventory, we'd do (beg + end)/2.
    # Here we only have a snapshot.
    avg_inventory_value = total_inventory_value

    turnover = cogs / avg_inventory_value if avg_inventory_value else 0.0

    print("\n--- INVENTORY TURNOVER (SIMPLE) ---")
    print(f"Approx. COGS: {cogs:,.2f} (assumed 50% of total sales here)")
    print(f"Approx. Inventory Value: {total_inventory_value:,.2f}")
    print(f"Inventory Turnover Ratio: {turnover:.2f}")


def compute_offline_vs_online_sales(
    order_df: pd.DataFrame, retail_df: pd.DataFrame
) -> None:
    """
    Compare offline (retail.csv) vs. online (order.csv) total revenue.
    Assumes:
      - order_df has a numeric 'Total'
      - retail_df has a numeric 'CA TTC'
    """
    if order_df.empty or retail_df.empty:
        print(
            "[WARN] Missing order or retail data, skipping offline vs. online synergy."
        )
        return

    online_sales = order_df["Total"].sum() if "Total" in order_df.columns else 0
    offline_sales = retail_df["CA TTC"].sum() if "CA TTC" in retail_df.columns else 0

    total_combined = online_sales + offline_sales
    pct_online = (online_sales / total_combined) * 100 if total_combined else 0
    pct_offline = 100 - pct_online

    print("\n--- OFFLINE vs. ONLINE SALES ---")
    print(f"Online Sales (order.csv): {online_sales:,.2f}")
    print(f"Offline Sales (retail.csv): {offline_sales:,.2f}")
    print(f"Total Combined Sales: {total_combined:,.2f}")
    print(f"Online %: {pct_online:.2f}%, Offline %: {pct_offline:.2f}%")


# ------------------------------------------------------------------------------
#                              4. MAIN SCRIPT FLOW
# ------------------------------------------------------------------------------
def main():
    """
    1) Load all cleaned CSVs
    2) Merge cart & order for real cart abandonment
    3) Compute sales & customer metrics from order
    4) Enrich order with inventory/products -> advanced margin
    5) Inventory turnover (naive approach)
    6) Offline (retail) vs. online synergy
    """
    print("[INFO] Starting Descriptive Analysis for 5octobre...\n")

    # Load data
    cart_df, order_df, inventory_df, retail_df = load_cleaned_data()
    products_df = pd.read_csv(
        "/Users/benpfeffer/Library/Mobile Documents/com~apple~CloudDocs/projects-portfolio-main/5-octobre/data/processed/products.csv"
    )
    # Merge cart -> order
    cart_order_merged = merge_cart_order(cart_df, order_df)
    compute_cart_abandonment(cart_order_merged)

    # Core sales & customer metrics
    compute_sales_metrics(order_df)
    compute_customer_metrics(order_df)

    # Enrich order data with inventory & product info
    enriched_orders = enrich_order_data(order_df, inventory_df, products_df)
    compute_margin_analysis(enriched_orders)

    # Inventory turnover
    compute_inventory_turnover(inventory_df, order_df)

    # Compare offline & online
    compute_offline_vs_online_sales(order_df, retail_df)

    print("\n[INFO] Descriptive Analysis & KPI calculations completed successfully.\n")


if __name__ == "__main__":
    main()
