import sys

sys.path.append(
    "/Users/benpfeffer/Library/Mobile Documents/com~apple~CloudDocs/projects-portfolio-main/5-octobre"
)
from src.metrics import *
from src.data_preprocessing import preprocess_data
import os
import pandas as pd


def run_analysis(cart_df, order_df):
    """
    Run a series of analyses on the provided cart and order dataframes and return insights as a dictionary.

    Assumes that:
    - cart_df and order_df are already loaded pandas DataFrames.
    - You have defined all metric functions (basic_kpis, compute_revenue_over_time, etc.) elsewhere.
    """
    insights = {}

    # Compute various metrics (assuming these functions are defined)
    # Basic KPIs
    insights["basic_kpis"] = basic_kpis(order_df)

    # Revenue metrics
    insights["monthly_revenue"] = compute_revenue_over_time(order_df, freq="M")
    insights["average_order_value"] = compute_average_order_value(order_df)
    insights["revenue_concentration"] = revenue_concentration(order_df)

    # Cart metrics
    insights["cart_abandonment_rate"] = compute_cart_abandonment_rate(cart_df, order_df)

    # Customer metrics
    insights["monthly_unique_customers"] = analyze_customer_count(order_df)
    insights["customer_cohorts"] = new_vs_returning_customers(order_df)
    insights["customer_segmentation"] = customer_segmentation_by_value(order_df)
    insights["repeat_vs_one_time_customers"] = repeat_vs_one_time_customers(order_df)
    insights["clv"] = calculate_clv(order_df)

    # Order analysis
    insights["order_value_distribution"] = order_value_distribution(order_df)
    insights["monthly_revenue_growth"] = revenue_growth(order_df, freq="M")
    insights["day_of_week_revenue"] = day_of_week_analysis(order_df)
    insights["hour_of_day_revenue"] = hour_of_day_analysis(order_df)

    # Payment and Geographic Analysis
    insights["payment_method_analysis"] = payment_method_analysis(order_df)
    insights["country_analysis"] = country_analysis(order_df)

    # Advanced analytics
    insights["rfm_analysis"] = rfm_analysis(order_df)
    insights["cohort_retention"] = cohort_analysis(order_df)

    # Performance Analysis
    insights["refund_cancellation_analysis"] = refund_cancellation_analysis(order_df)
    insights["order_state_analysis"] = order_state_analysis(order_df)
    insights["monthly_cancellation_refund_trends"] = monthly_cancellation_refund_trends(
        order_df
    )

    # Year-over-year Analysis
    insights["year_over_year_revenue"] = year_over_year_revenue(order_df)
    insights["year_over_year_orders"] = year_over_year_orders(order_df)
    insights["year_over_year_aov"] = year_over_year_aov(order_df)

    # New metrics
    insights["repeat_purchase_interval_days"] = repeat_purchase_interval(order_df)
    insights["churn_rate"] = churn_rate(order_df)
    insights["refined_clv"] = refined_clv(order_df)

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
    cart_df, order_df = preprocess_data()
    df = run_analysis(cart_df, order_df)
    save_to_csv(df)
