import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from analysis import run_analysis
from config import BASE_DIR, load_logger
from data_preprocessing import preprocess_data

logger = load_logger()
cart_df, order_df = preprocess_data()
insights = run_analysis(cart_df, order_df)

# Set style parameters
plt.style.use("ggplot")
sns.set_palette("husl")
PLOT_STYLE = {
    "figure.figsize": (12, 8),
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
}
plt.rcParams.update(PLOT_STYLE)

# Custom color palette
COLORS = sns.color_palette("husl", 8)


def plot_revenue_over_time(insights):
    """Enhanced monthly revenue trends with trend line and YoY comparison."""
    monthly_revenue = insights.get("monthly_revenue")
    if monthly_revenue is not None and not monthly_revenue.empty:
        fig = plt.figure(figsize=(14, 8))

        # Plot actual revenue
        plt.plot(
            monthly_revenue.index,
            monthly_revenue.values,
            label="Monthly Revenue",
            color=COLORS[0],
            linewidth=2,
        )

        # Add trend line
        z = np.polyfit(range(len(monthly_revenue)), monthly_revenue.values, 1)
        p = np.poly1d(z)
        plt.plot(
            monthly_revenue.index,
            p(range(len(monthly_revenue))),
            "--",
            color=COLORS[1],
            label="Trend Line",
        )

        # Calculate and plot moving average
        ma = monthly_revenue.rolling(window=3).mean()
        plt.plot(monthly_revenue.index, ma, color=COLORS[2], label="3-Month Moving Average")

        plt.title("Monthly Revenue Trends with Moving Average", fontsize=16, pad=20)
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Revenue (€)", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()

        # Add annotations for key points
        max_revenue = monthly_revenue.max()
        max_date = monthly_revenue.idxmax()
        plt.annotate(
            f"Peak: €{max_revenue:,.0f}",
            xy=(max_date, max_revenue),
            xytext=(10, 10),
            textcoords="offset points",
        )

        return plt
    return None


def plot_customer_segments_enhanced(insights):
    """Enhanced customer segmentation visualization with value distribution."""
    segmentation = insights.get("customer_segmentation")
    if segmentation is not None and not segmentation.empty:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

        # Customer count by segment
        segment_counts = segmentation["segment"].value_counts()
        sns.barplot(x=segment_counts.index, y=segment_counts.values, ax=ax1, hue=segment_counts.index, legend=False)
        ax1.set_title("Customer Distribution by Segment")
        ax1.set_ylabel("Number of Customers")

        # Value distribution by segment
        sns.boxplot(x="segment", y="Total", data=segmentation, ax=ax2, hue="segment", legend=False)
        ax2.set_title("Value Distribution by Segment")
        ax2.set_ylabel("Customer Value (€)")

        plt.tight_layout()
        return plt
    return None


def plot_rfm_analysis_dashboard(insights):
    """Comprehensive RFM analysis dashboard."""
    rfm = insights.get("rfm_analysis")
    if rfm is not None and not rfm.empty:
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(2, 2)

        # RFM Score Correlation Heatmap
        ax1 = fig.add_subplot(gs[0, 0])
        rfm_scores = rfm[["R_score", "F_score", "M_score"]]
        sns.heatmap(rfm_scores.corr(), annot=True, cmap="YlOrRd", ax=ax1)
        ax1.set_title("RFM Score Correlation")

        # Distribution of R, F, M scores
        ax2 = fig.add_subplot(gs[0, 1])
        rfm_melted = pd.melt(rfm_scores)
        sns.boxplot(x="variable", y="value", data=rfm_melted, ax=ax2, hue="variable", legend=False)
        ax2.set_title("Distribution of RFM Scores")

        # Scatter plot of Frequency vs Monetary
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.scatter(rfm["Frequency"], rfm["Monetary"], alpha=0.5)
        ax3.set_xlabel("Frequency")
        ax3.set_ylabel("Monetary")
        ax3.set_title("Frequency vs Monetary Value")

        # Recency distribution
        ax4 = fig.add_subplot(gs[1, 1])
        sns.histplot(rfm["Recency"], bins=30, ax=ax4)
        ax4.set_title("Recency Distribution")

        plt.tight_layout()
        return plt
    return None


def plot_customer_lifecycle(insights):
    """Visualize customer lifecycle metrics."""
    customer_data = insights.get("customer_cohorts")
    clv = insights.get("clv")

    if customer_data is not None and not customer_data.empty:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # New vs Returning customers trend
        customer_data.plot(kind="area", stacked=True, ax=ax1)
        ax1.set_title("Customer Acquisition vs Retention")
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Number of Customers")
        ax1.legend(title="Customer Type")

        # Customer Lifetime Value
        if clv:
            data = pd.Series({"Acquisition": 100, "CLV": clv})
            data.plot(kind="bar", ax=ax2, color=[COLORS[0], COLORS[1]])
            ax2.set_title("Customer Lifetime Value Analysis")
            ax2.set_ylabel("Value (€)")

        plt.tight_layout()
        return plt
    return None


def plot_payment_and_geography_dashboard(insights):
    """Combined payment and geographic analysis dashboard."""
    payment_analysis = insights.get("payment_method_analysis")
    country_analysis = insights.get("country_analysis")

    if payment_analysis is not None and country_analysis is not None:
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(2, 2)

        # Payment methods by order count
        ax1 = fig.add_subplot(gs[0, 0])
        payment_analysis["order_count"].plot(kind="bar", ax=ax1)
        ax1.set_title("Orders by Payment Method")
        ax1.set_xlabel("Payment Method")
        ax1.tick_params(axis="x", rotation=45)

        # Payment methods by average order value
        ax2 = fig.add_subplot(gs[0, 1])
        payment_analysis["avg_order_value"].plot(kind="bar", ax=ax2)
        ax2.set_title("Average Order Value by Payment Method")
        ax2.set_xlabel("Payment Method")
        ax2.tick_params(axis="x", rotation=45)

        # Top countries by revenue
        ax3 = fig.add_subplot(gs[1, :])
        top_countries = country_analysis.nlargest(10, "total_revenue")
        sns.barplot(data=top_countries, x=top_countries.index, y="total_revenue", ax=ax3, hue=top_countries.index, legend=False)
        ax3.set_title("Top 10 Countries by Revenue")
        ax3.set_xlabel("Country")
        ax3.set_ylabel("Total Revenue (€)")
        ax3.tick_params(axis="x", rotation=45)

        plt.tight_layout()
        return plt
    return None


def plot_order_patterns_dashboard(insights):
    """Comprehensive order patterns dashboard."""
    day_revenue = insights.get("day_of_week_revenue")
    hour_revenue = insights.get("hour_of_day_revenue")
    order_dist = insights.get("order_value_distribution")

    if all(x is not None for x in [day_revenue, hour_revenue, order_dist]):
        fig = plt.figure(figsize=(15, 12))
        gs = fig.add_gridspec(2, 2)

        # Daily pattern
        ax1 = fig.add_subplot(gs[0, 0])
        sns.barplot(x=day_revenue.index, y=day_revenue.values, ax=ax1, hue=day_revenue.index, legend=False)
        ax1.set_title("Revenue by Day of Week")
        ax1.tick_params(axis="x", rotation=45)

        # Hourly pattern
        ax2 = fig.add_subplot(gs[0, 1])
        sns.lineplot(x=hour_revenue.index, y=hour_revenue.values, ax=ax2)
        ax2.set_title("Revenue by Hour of Day")
        ax2.set_xlabel("Hour")

        # Order value distribution
        ax3 = fig.add_subplot(gs[1, :])
        sns.histplot(data=pd.Series(order_dist), bins=30, ax=ax3)
        ax3.set_title("Order Value Distribution")
        ax3.set_xlabel("Order Value (€)")

        plt.tight_layout()
        return plt
    return None


def plot_cohort_retention(insights):
    """Plot cohort retention analysis."""
    cohort_data = insights.get("cohort_retention")

    if cohort_data is not None:
        plt.figure(figsize=(12, 8))

        # Create heatmap
        sns.heatmap(cohort_data, annot=True, fmt=".0%", cmap="YlOrRd", vmin=0, vmax=1)

        plt.title("Customer Cohort Retention Analysis")
        plt.xlabel("Cohort Period")
        plt.ylabel("Cohort Group")

        return plt
    return None


def plot_yoy_metrics(insights):
    """Plot year-over-year metrics comparison."""
    yoy_revenue = insights.get("year_over_year_revenue")
    yoy_orders = insights.get("year_over_year_orders")
    yoy_aov = insights.get("year_over_year_aov")

    if all(x is not None for x in [yoy_revenue, yoy_orders, yoy_aov]):
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(2, 2)

        # Revenue YoY
        ax1 = fig.add_subplot(gs[0, 0])
        yoy_revenue.plot(kind="bar", ax=ax1)
        ax1.set_title("Revenue Year over Year")
        ax1.set_xlabel("Year")
        ax1.set_ylabel("Revenue (€)")
        ax1.tick_params(axis="x", rotation=45)

        # Orders YoY
        ax2 = fig.add_subplot(gs[0, 1])
        yoy_orders.plot(kind="bar", ax=ax2)
        ax2.set_title("Orders Year over Year")
        ax2.set_xlabel("Year")
        ax2.set_ylabel("Number of Orders")
        ax2.tick_params(axis="x", rotation=45)

        # AOV YoY
        ax3 = fig.add_subplot(gs[1, :])
        yoy_aov.plot(kind="bar", ax=ax3)
        ax3.set_title("Average Order Value Year over Year")
        ax3.set_xlabel("Year")
        ax3.set_ylabel("AOV (€)")
        ax3.tick_params(axis="x", rotation=45)

        plt.tight_layout()
        return plt
    return None


def plot_refund_trends(insights):
    """Plot refund and cancellation trends."""
    trends = insights.get("monthly_cancellation_refund_trends")

    if trends is not None:
        fig, ax = plt.subplots(figsize=(12, 6))

        trends.plot(ax=ax)
        plt.title("Monthly Refund and Cancellation Trends")
        plt.xlabel("Month")
        plt.ylabel("Count")
        plt.legend()
        plt.xticks(rotation=45)

        return plt
    return None


def plot_forecasts_dashboard(insights):
    """Plot comprehensive forecast dashboard including revenue, orders, and customers."""
    revenue_forecast = insights.get("revenue_forecast")
    orders_forecast = insights.get("orders_forecast")
    customers_forecast = insights.get("customers_forecast")
    multi_metric_forecast = insights.get("multi_metric_forecast")

    if any(x is not None for x in [revenue_forecast, orders_forecast, customers_forecast, multi_metric_forecast]):
        fig = plt.figure(figsize=(15, 12))
        gs = fig.add_gridspec(3, 2)

        # Revenue Forecast
        if revenue_forecast is not None:
            ax1 = fig.add_subplot(gs[0, :])
            # Plot historical data
            historical_data = insights.get("monthly_revenue")
            if historical_data is not None:
                ax1.plot(
                    historical_data.index,
                    historical_data.values,
                    label="Historical Revenue",
                    color=COLORS[0],
                    linewidth=2,
                )

            # Plot forecast
            forecast_dates = pd.to_datetime(revenue_forecast["ds"])
            ax1.plot(
                forecast_dates,
                revenue_forecast["yhat"],
                label="Revenue Forecast",
                color=COLORS[1],
                linestyle="--",
                linewidth=2,
            )

            # Add confidence intervals
            ax1.fill_between(forecast_dates, revenue_forecast["yhat_lower"], revenue_forecast["yhat_upper"], alpha=0.2, color=COLORS[1], label="95% Confidence Interval")

            ax1.set_title("Revenue Forecast")
            ax1.set_xlabel("Date")
            ax1.set_ylabel("Revenue (€)")
            ax1.legend()

        # Orders Forecast
        if orders_forecast is not None:
            ax2 = fig.add_subplot(gs[1, 0])
            forecast_dates = pd.to_datetime(orders_forecast["ds"])
            ax2.plot(
                forecast_dates,
                orders_forecast["yhat"],
                label="Orders Forecast",
                color=COLORS[2],
                linestyle="--",
            )
            ax2.fill_between(
                forecast_dates,
                orders_forecast["yhat_lower"],
                orders_forecast["yhat_upper"],
                alpha=0.2,
                color=COLORS[2],
            )
            ax2.set_title("Orders Forecast")
            ax2.set_xlabel("Date")
            ax2.set_ylabel("Number of Orders")
            ax2.legend()

        # Customers Forecast
        if customers_forecast is not None:
            ax3 = fig.add_subplot(gs[1, 1])
            forecast_dates = pd.to_datetime(customers_forecast["ds"])
            ax3.plot(
                forecast_dates,
                customers_forecast["yhat"],
                label="Customer Forecast",
                color=COLORS[3],
                linestyle="--",
            )
            ax3.fill_between(
                forecast_dates,
                customers_forecast["yhat_lower"],
                customers_forecast["yhat_upper"],
                alpha=0.2,
                color=COLORS[3],
            )
            ax3.set_title("Customer Forecast")
            ax3.set_xlabel("Date")
            ax3.set_ylabel("Number of Customers")
            ax3.legend()

        # Multi-metric comparison
        if multi_metric_forecast is not None:
            ax4 = fig.add_subplot(gs[2, :])
            metrics = ["revenue", "orders", "unique_customers"]
            for i, metric in enumerate(metrics):
                if f"{metric}_yhat" in multi_metric_forecast.columns:
                    ax4.plot(
                        pd.to_datetime(multi_metric_forecast["ds"]),
                        multi_metric_forecast[f"{metric}_yhat"],
                        label=f"{metric.capitalize()} Forecast",
                        color=COLORS[i],
                        linestyle="--",
                    )
            ax4.set_title("Multi-metric Forecast Comparison")
            ax4.set_xlabel("Date")
            ax4.set_ylabel("Normalized Values")
            ax4.legend()

        plt.tight_layout()
        return plt
    return None


def create_dashboard():
    """Create and save enhanced plots."""
    logger.info("Starting to create enhanced dashboard...")

    plots = {
        "revenue_trends": plot_revenue_over_time(insights),
        "customer_segments": plot_customer_segments_enhanced(insights),
        "rfm_analysis": plot_rfm_analysis_dashboard(insights),
        "customer_lifecycle": plot_customer_lifecycle(insights),
        "payment_geography": plot_payment_and_geography_dashboard(insights),
        "order_patterns": plot_order_patterns_dashboard(insights),
        "cohort_retention": plot_cohort_retention(insights),
        "yoy_metrics": plot_yoy_metrics(insights),
        "refund_trends": plot_refund_trends(insights),
        "forecasts": plot_forecasts_dashboard(insights),
    }

    # Create output directory
    output_dir = os.path.join(BASE_DIR, "output", "fig")
    os.makedirs(output_dir, exist_ok=True)

    # Save plots with enhanced quality
    for name, plot in plots.items():
        if plot is not None:
            try:
                plot.savefig(
                    f"{output_dir}/{name}.png",
                    bbox_inches="tight",
                    dpi=300,
                    facecolor="white",
                    edgecolor="none",
                )
                logger.info(f"Successfully saved {name} plot")
                plt.close()
            except Exception as e:
                logger.error(f"Error saving {name} plot: {e}")
        else:
            logger.warning(f"Plot {name} could not be generated")

    logger.info("Enhanced dashboard creation completed")


if __name__ == "__main__":
    create_dashboard()
