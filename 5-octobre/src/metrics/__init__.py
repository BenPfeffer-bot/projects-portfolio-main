"""
E-commerce metrics and analytics package.
"""

from .basic_metrics import (
    basic_kpis,
    compute_average_order_value,
    order_value_distribution,
    compute_cart_abandonment_rate,
    order_state_analysis,
)

from .customer_analytics import (
    rfm_analysis,
    customer_segmentation_by_value,
    calculate_clv,
    analyze_customer_behavior,
    churn_rate,
)

from .revenue_analytics import (
    revenue_concentration,
    payment_method_analysis,
    country_analysis,
    refund_cancellation_analysis,
    monthly_cancellation_refund_trends,
)

from .time_analytics import (
    compute_revenue_over_time,
    revenue_growth,
    analyze_customer_count,
    cohort_analysis,
    day_of_week_analysis,
    hour_of_day_analysis,
    year_over_year_metrics,
)

__all__ = [
    # Basic metrics
    "basic_kpis",
    "compute_average_order_value",
    "order_value_distribution",
    "compute_cart_abandonment_rate",
    "order_state_analysis",
    # Customer analytics
    "rfm_analysis",
    "customer_segmentation_by_value",
    "calculate_clv",
    "repeat_vs_one_time_customers",
    "churn_rate",
    # Revenue analytics
    "revenue_concentration",
    "payment_method_analysis",
    "country_analysis",
    "refund_cancellation_analysis",
    "monthly_cancellation_refund_trends",
    # Time analytics
    "compute_revenue_over_time",
    "revenue_growth",
    "analyze_customer_count",
    "cohort_analysis",
    "day_of_week_analysis",
    "hour_of_day_analysis",
    "year_over_year_metrics",
]
