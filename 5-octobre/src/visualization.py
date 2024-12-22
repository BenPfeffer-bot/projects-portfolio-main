import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style for better-looking plots
plt.style.use("ggplot")
sns.set_palette("husl")

# Load the datasets
forecast_df = pd.read_csv("data/analysis/multi_metric_forecast.csv")
summary_df = pd.read_csv("data/analysis/summary.csv")

# Convert summary_df to a dictionary for easier access
summary_dict = dict(zip(summary_df["metric"], summary_df["value"]))

# 1. Enhanced Time Series Forecast Plot
plt.figure(figsize=(15, 10))
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))

# Revenue forecast
ax1.plot(
    pd.to_datetime(forecast_df["ds"]),
    forecast_df["revenue_yhat_arima"],
    marker="o",
    linewidth=2,
    label="Revenue",
)
ax1.set_title("Revenue Forecast", fontsize=14, pad=20)
ax1.set_xlabel("Date", fontsize=12)
ax1.set_ylabel("Revenue ($)", fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.tick_params(axis="x", rotation=45)

# Customer metrics forecast
ax2.plot(
    pd.to_datetime(forecast_df["ds"]),
    forecast_df["unique_customers_yhat_arima"],
    marker="s",
    label="Unique Customers",
    linewidth=2,
)
ax2.plot(
    pd.to_datetime(forecast_df["ds"]),
    forecast_df["returning_customers_yhat_arima"],
    marker="^",
    label="Returning Customers",
    linewidth=2,
)
ax2.set_title("Customer Metrics Forecast", fontsize=14, pad=20)
ax2.set_xlabel("Date", fontsize=12)
ax2.set_ylabel("Number of Customers", fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.tick_params(axis="x", rotation=45)
ax2.legend()

plt.tight_layout()
plt.savefig("output/fig2/forecast_metrics.png", dpi=300, bbox_inches="tight")
plt.close()

# 2. Customer Behavior Analysis
plt.figure(figsize=(12, 6))
metrics = [
    "basic_kpis_avg_orders_per_customer",
    "average_order_value",
    "cart_abandonment_rate",
    "churn_rate",
]
values = [summary_dict[m] for m in metrics]
labels = [
    "Avg Orders/Customer",
    "Avg Order Value ($)",
    "Cart Abandonment Rate (%)",
    "Churn Rate (%)",
]

colors = sns.color_palette("husl", 4)
bars = plt.bar(labels, values, color=colors)
plt.title("Customer Behavior Metrics", fontsize=14, pad=20)
plt.xticks(rotation=45, ha="right")
plt.grid(True, alpha=0.3, axis="y")

# Add value labels on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2.0,
        height,
        f"{height:.1f}",
        ha="center",
        va="bottom",
    )

plt.tight_layout()
plt.savefig("output/fig2/customer_behavior.png", dpi=300, bbox_inches="tight")
plt.close()

# 3. Enhanced Order Value Distribution
plt.figure(figsize=(12, 6))
order_stats = {
    "min": summary_dict["order_value_distribution_min"],
    "25%": summary_dict["order_value_distribution_25%_quartile"],
    "median": summary_dict["order_value_distribution_median"],
    "mean": summary_dict["order_value_distribution_mean"],
    "75%": summary_dict["order_value_distribution_75%_quartile"],
    "max": summary_dict["order_value_distribution_max"],
}

plt.boxplot(
    [
        [order_stats["min"]],  # Each value needs to be in its own list
        [order_stats["25%"]],
        [order_stats["median"]],
        [order_stats["75%"]],
        [order_stats["max"]],
    ],
    labels=["Min", "25%", "Median", "75%", "Max"],
    vert=False,
    widths=0.7,
)
plt.title("Order Value Distribution", fontsize=14, pad=20)
plt.xlabel("Order Value ($)", fontsize=12)

# Add mean line
plt.axvline(
    x=order_stats["mean"],
    color="red",
    linestyle="--",
    label=f"Mean: ${order_stats['mean']:.2f}",
)
plt.legend()

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("output/fig2/order_distribution.png", dpi=300, bbox_inches="tight")
plt.close()

# 4. Revenue and Customer Segments
plt.figure(figsize=(15, 6))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Revenue Concentration
revenue_sizes = [
    summary_dict["revenue_concentration_top_10pct_revenue_concentration"],
    100 - summary_dict["revenue_concentration_top_10pct_revenue_concentration"],
]
ax1.pie(
    revenue_sizes,
    labels=["Top 10% Customers", "Other 90% Customers"],
    autopct="%1.1f%%",
    colors=sns.color_palette("husl", 2),
    startangle=90,
)
ax1.set_title("Revenue Concentration", fontsize=14, pad=20)

# Customer Segments
customer_sizes = [
    summary_dict["repeat_vs_one_time_customers_one_time_buyers_pct"],
    summary_dict["repeat_vs_one_time_customers_multi_buyers_pct"],
]
ax2.pie(
    customer_sizes,
    labels=["One-time Buyers", "Repeat Customers"],
    autopct="%1.1f%%",
    colors=sns.color_palette("husl", 2),
    startangle=90,
)
ax2.set_title("Customer Segments", fontsize=14, pad=20)

plt.tight_layout()
plt.savefig("output/fig2/segments_analysis.png", dpi=300, bbox_inches="tight")
plt.close()

# 5. Refund and Cancellation Analysis
plt.figure(figsize=(12, 6))
metrics = [
    "refund_cancellation_analysis_refund_rate",
    "refund_cancellation_analysis_cancellation_rate",
    "refund_cancellation_analysis_revenue_refund_pct",
    "refund_cancellation_analysis_revenue_cancel_pct",
]
values = [summary_dict[m] for m in metrics]
labels = [
    "Refund Rate (%)",
    "Cancellation Rate (%)",
    "Revenue Lost to Refunds (%)",
    "Revenue Lost to Cancellations (%)",
]

bars = plt.bar(labels, values, color=sns.color_palette("husl", 4))
plt.title("Refund and Cancellation Analysis", fontsize=14, pad=20)
plt.xticks(rotation=45, ha="right")
plt.grid(True, alpha=0.3, axis="y")

# Add value labels on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2.0,
        height,
        f"{height:.1f}%",
        ha="center",
        va="bottom",
    )

plt.tight_layout()
plt.savefig("output/fig2/refund_cancellation.png", dpi=300, bbox_inches="tight")
plt.close()

# 6. NEW: Customer Lifetime Value (CLV) Analysis
plt.figure(figsize=(10, 6))
clv_data = {
    "Current CLV": summary_dict["clv"],
    "Refined CLV": summary_dict["refined_clv"],
    "Avg Revenue per Customer": summary_dict["basic_kpis_total_revenue"]
    / summary_dict["basic_kpis_unique_customers"],
}

plt.bar(clv_data.keys(), clv_data.values(), color=sns.color_palette("husl", 3))
plt.title("Customer Lifetime Value Analysis", fontsize=14, pad=20)
plt.ylabel("Value ($)", fontsize=12)
plt.grid(True, alpha=0.3, axis="y")

for i, v in enumerate(clv_data.values()):
    plt.text(i, v, f"${v:,.2f}", ha="center", va="bottom")

plt.tight_layout()
plt.savefig("output/fig2/clv_analysis.png", dpi=300, bbox_inches="tight")
plt.close()

# 7. NEW: Order Patterns Over Time
plt.figure(figsize=(12, 6))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Refunds and Cancellations Forecast
ax1.plot(
    pd.to_datetime(forecast_df["ds"]),
    forecast_df["refunds_count_yhat_xgb"],
    label="Refunds",
    marker="o",
)
ax1.plot(
    pd.to_datetime(forecast_df["ds"]),
    forecast_df["cancellations_count_yhat_arima"],
    label="Cancellations",
    marker="s",
)
ax1.set_title("Refunds & Cancellations Forecast", fontsize=14)
ax1.set_xlabel("Date", fontsize=12)
ax1.set_ylabel("Count", fontsize=12)
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.tick_params(axis="x", rotation=45)

# Average Order Value Forecast
ax2.plot(
    pd.to_datetime(forecast_df["ds"]),
    forecast_df["aov_yhat_arima"],
    label="AOV",
    marker="o",
    color="green",
)
ax2.set_title("Average Order Value Forecast", fontsize=14)
ax2.set_xlabel("Date", fontsize=12)
ax2.set_ylabel("Value ($)", fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.savefig("output/fig2/order_patterns.png", dpi=300, bbox_inches="tight")
plt.close()

# 8. NEW: Customer Purchase Behavior
plt.figure(figsize=(12, 6))
purchase_metrics = {
    "Repeat Purchase Interval (Days)": summary_dict["repeat_purchase_interval_days"],
    "Multi-buyer Revenue %": summary_dict[
        "repeat_vs_one_time_customers_multi_buyer_revenue_pct"
    ],
    "Orders per Customer": summary_dict["basic_kpis_avg_orders_per_customer"],
}

colors = sns.color_palette("husl", 3)
plt.bar(purchase_metrics.keys(), purchase_metrics.values(), color=colors)
plt.title("Customer Purchase Behavior Analysis", fontsize=14, pad=20)
plt.xticks(rotation=45, ha="right")
plt.grid(True, alpha=0.3, axis="y")

for i, (k, v) in enumerate(purchase_metrics.items()):
    plt.text(i, v, f"{v:.1f}", ha="center", va="bottom")

plt.tight_layout()
plt.savefig("output/fig2/purchase_behavior.png", dpi=300, bbox_inches="tight")
plt.close()

# 9. NEW: Revenue Impact Analysis
plt.figure(figsize=(12, 6))
revenue_metrics = {
    "Total Revenue": summary_dict["basic_kpis_total_revenue"],
    "Revenue Lost to Refunds": summary_dict[
        "refund_cancellation_analysis_revenue_lost_refunds"
    ],
    "Revenue Lost to Cancellations": summary_dict[
        "refund_cancellation_analysis_revenue_lost_cancellations"
    ],
    "Top 10% Revenue": summary_dict["revenue_concentration_top_10pct_revenue"],
}

plt.bar(
    revenue_metrics.keys(),
    revenue_metrics.values(),
    color=sns.color_palette("husl", len(revenue_metrics)),
)
plt.title("Revenue Impact Analysis", fontsize=14, pad=20)
plt.ylabel("Amount ($)", fontsize=12)
plt.xticks(rotation=45, ha="right")
plt.grid(True, alpha=0.3, axis="y")

for i, (k, v) in enumerate(revenue_metrics.items()):
    plt.text(i, v, f"${v:,.0f}", ha="center", va="bottom")

plt.tight_layout()
plt.savefig("output/fig2/revenue_impact.png", dpi=300, bbox_inches="tight")
plt.close()

# 10. Customer Behavior Analysis
plt.figure(figsize=(15, 8))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Customer Type Distribution
customer_type_data = {
    "Repeat Customers": summary_dict["customer_behavior_repeat_customers_count"],
    "One-time Customers": summary_dict["customer_behavior_one_time_customers_count"],
}

ax1.pie(
    customer_type_data.values(),
    labels=customer_type_data.keys(),
    autopct="%1.1f%%",
    colors=sns.color_palette("husl", 2),
)
ax1.set_title("Customer Type Distribution", pad=20)

# Revenue Distribution
revenue_data = {
    "Repeat Revenue": summary_dict["customer_behavior_repeat_customers_total_revenue"],
    "One-time Revenue": summary_dict[
        "customer_behavior_one_time_customers_total_revenue"
    ],
}

bars = ax2.bar(
    revenue_data.keys(), revenue_data.values(), color=sns.color_palette("husl", 2)
)
ax2.set_title("Revenue Distribution by Customer Type", pad=20)
ax2.set_ylabel("Revenue ($)")
ax2.grid(True, alpha=0.3, axis="y")

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax2.text(
        bar.get_x() + bar.get_width() / 2.0,
        height,
        f"${height:,.0f}",
        ha="center",
        va="bottom",
    )

plt.tight_layout()
plt.savefig("output/fig2/customer_behavior.png", dpi=300, bbox_inches="tight")
plt.close()
