import sys


sys.path.append(
    "/Users/benpfeffer/Library/Mobile Documents/com~apple~CloudDocs/projects-portfolio-main/5-octobre"
)
from src.metrics.basic_metrics import *
from src.metrics.customer_analytics import *
from src.metrics.revenue_analytics import *
from src.metrics.time_analytics import *
from src.metrics.performance_analytics import (
    margins,
    gross_margin_rate,
    calculate_inventory_metrics,
    enrich_retail_data,
    product_profitability_analysis,
    abc_inventory_analysis,
    calculate_stock_recommendations,
    analyze_product_movement,
)
from src.data_preprocessing import preprocess_data
import os
import pandas as pd


def run_analysis(cart_df, order_df, inventory_df=None):
    """
    Run a series of analyses on the provided cart and order dataframes and return insights as a dictionary.
    Organizes metrics by category: basic, customer, revenue, time-based analytics, and performance metrics.

    Args:
        cart_df: DataFrame containing cart data
        order_df: DataFrame containing order data
        inventory_df: DataFrame containing inventory data (optional)
    """
    insights = {}

    # Basic Metrics
    insights["basic_kpis"] = basic_kpis(order_df)
    insights["average_order_value"] = compute_average_order_value(order_df)
    insights["order_value_distribution"] = order_value_distribution(order_df)
    insights["cart_abandonment_rate"] = compute_cart_abandonment_rate(cart_df, order_df)
    insights["order_state_analysis"] = order_state_analysis(order_df)

    # Customer Analytics
    insights["rfm_analysis"] = rfm_analysis(order_df)
    insights["customer_segmentation"] = customer_segmentation_by_value(order_df)
    insights["customer_lifetime_value"] = calculate_clv(order_df)
    insights["repeat_vs_one_time"] = analyze_customer_behavior(order_df)
    insights["churn_rate"] = churn_rate(order_df)

    # Revenue Analytics
    insights["revenue_concentration"] = revenue_concentration(order_df)
    insights["payment_methods"] = payment_method_analysis(order_df)
    insights["country_analysis"] = country_analysis(order_df)
    insights["refund_analysis"] = refund_cancellation_analysis(order_df)
    insights["monthly_refund_trends"] = monthly_cancellation_refund_trends(order_df)

    # Time-based Analytics
    insights["revenue_over_time"] = compute_revenue_over_time(order_df, freq="M")
    insights["revenue_growth"] = revenue_growth(order_df, freq="M")
    insights["customer_count_trend"] = analyze_customer_count(order_df)
    insights["cohort_retention"] = cohort_analysis(order_df)
    insights["daily_patterns"] = day_of_week_analysis(order_df)
    insights["hourly_patterns"] = hour_of_day_analysis(order_df)
    insights["year_over_year"] = year_over_year_metrics(order_df)

    # Performance and Inventory Analytics (if inventory data is available)
    if inventory_df is not None:
        # Enrich retail data with inventory information
        enriched_retail = enrich_retail_data(order_df, inventory_df)

        # Margin Analysis
        insights["margins"] = margins(enriched_retail)
        insights["margin_analysis"], insights["margin_stats"] = gross_margin_rate(
            enriched_retail, groupby="Ref", time_freq="M"
        )

        # Product Performance Analysis
        insights["product_metrics"], insights["product_stats"] = (
            product_profitability_analysis(
                order_df, inventory_df, period="M", min_sales=1
            )
        )

        # Inventory Analysis
        insights["inventory_metrics"] = calculate_inventory_metrics(
            order_df, inventory_df, period="W"
        )

        # ABC Analysis
        insights["abc_analysis"], insights["abc_stats"] = abc_inventory_analysis(
            order_df, inventory_df, period="M"
        )

        # Stock Recommendations
        insights["stock_recommendations"] = calculate_stock_recommendations(
            order_df, inventory_df, period="D"
        )

        # Product Movement Analysis
        insights["product_movement"], insights["movement_stats"] = (
            analyze_product_movement(
                order_df, inventory_df, analysis_period="M", stock_coverage_days=90
            )
        )

    return insights


def save_to_csv(
    insights,
    output_dir="/Users/benpfeffer/Library/Mobile Documents/com~apple~CloudDocs/projects-portfolio-main/5-octobre/data/analysis",
):
    """
    Save the insights dictionary to CSV files.

    For each key in insights:
    - If value is a DataFrame, save directly to CSV with key as filename.
    - If value is a Series, convert to DataFrame and save.
    - If value is a scalar or dict, accumulate these into a single "summary.csv".
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    summary_data = {}
    for key, value in insights.items():
        if isinstance(value, pd.DataFrame):
            # Save DataFrame directly
            filename = f"{key}.csv"
            filepath = os.path.join(output_dir, filename)
            value.to_csv(filepath, index=True)
        elif isinstance(value, pd.Series):
            # Convert Series to DataFrame and save
            filename = f"{key}.csv"
            filepath = os.path.join(output_dir, filename)
            value.to_frame(name=key).to_csv(filepath, index=True)
        elif isinstance(value, (int, float, str)):
            # Scalar values saved to summary
            summary_data[key] = value
        elif isinstance(value, dict):
            # Flatten dict into summary
            for sub_key, sub_val in value.items():
                summary_data[f"{key}_{sub_key}"] = sub_val
        else:
            # For unrecognized types, attempt conversion to string
            summary_data[key] = str(value)

    # Save summary data
    if summary_data:
        summary_df = pd.DataFrame(
            list(summary_data.items()), columns=["metric", "value"]
        )
        summary_df.to_csv(os.path.join(output_dir, "summary.csv"), index=False)


if __name__ == "__main__":
    cart_df, order_df, inventory_df, retail_df = preprocess_data()
    insights = run_analysis(cart_df, order_df, inventory_df)
    save_to_csv(insights)
