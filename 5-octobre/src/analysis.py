import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from config import CLEANED_DATA_DIR, CART_FILENAME, ORDER_FILENAME


def load_cleaned_data():
    """
    Load the cleaned cart and order datasets.
    """
    cart_path = os.path.join(CLEANED_DATA_DIR, CART_FILENAME)
    order_path = os.path.join(CLEANED_DATA_DIR, ORDER_FILENAME)

    try:
        cart_df = pd.read_csv(cart_path)
        order_df = pd.read_csv(order_path)
        return cart_df, order_df
    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except pd.errors.EmptyDataError:
        print("One of the cleaned files is empty.")
    except Exception as e:
        print(f"Error loading cleaned data: {e}")

    return None, None


def basic_kpis(order_df, total_col="Total", client_col="Client"):
    """
    Compute basic KPIs:
    - Total orders
    - Total revenue
    - Unique customers
    - Average orders per customer
    """
    if total_col not in order_df.columns or client_col not in order_df.columns:
        print("Required columns for basic KPIs not found.")
        return {}

    total_orders = len(order_df)
    total_revenue = order_df[total_col].sum()
    unique_customers = order_df[client_col].nunique()
    avg_orders_per_customer = total_orders / unique_customers if unique_customers > 0 else 0

    return {"total_orders": total_orders, "total_revenue": total_revenue, "unique_customers": unique_customers, "avg_orders_per_customer": avg_orders_per_customer}


def compute_revenue_over_time(order_df, freq="M", date_col="Date", total_col="Total"):
    """Compute total revenue aggregated by a specified time frequency."""
    if date_col not in order_df.columns or total_col not in order_df.columns:
        print("Required columns for revenue over time not found.")
        return pd.Series(dtype=float)
    try:
        order_df[date_col] = pd.to_datetime(order_df[date_col], errors="coerce")
        revenue_series = order_df.set_index(date_col).resample(freq)[total_col].sum()
        return revenue_series
    except Exception as e:
        print(f"Error computing revenue over time: {e}")
        return pd.Series(dtype=float)


def compute_average_order_value(order_df, total_col="Total"):
    """Compute the Average Order Value (AOV)."""
    if total_col not in order_df.columns:
        print(f"Column {total_col} not found in orders data.")
        return np.nan
    try:
        return order_df[total_col].mean()
    except Exception as e:
        print(f"Error computing average order value: {e}")
        return np.nan


def compute_cart_abandonment_rate(cart_df, order_df, cart_id_col="ID commande", order_ref_col="Référence"):
    """Compute cart abandonment rate."""
    if cart_id_col not in cart_df.columns:
        print(f"Column {cart_id_col} not found in cart data.")
        return np.nan
    if order_ref_col not in order_df.columns:
        print(f"Column {order_ref_col} not found in order data.")
        return np.nan

    try:
        total_carts = cart_df[cart_id_col].nunique()
        completed_orders = cart_df[cart_id_col].isin(order_df[order_ref_col].unique()).sum()
        if total_carts == 0:
            return 0.0
        return (1 - (completed_orders / total_carts)) * 100
    except Exception as e:
        print(f"Error computing cart abandonment rate: {e}")
        return np.nan


def analyze_customer_count(order_df, date_col="Date", client_col="Client", freq="M"):
    """Analyze unique customers over time."""
    if date_col not in order_df.columns or client_col not in order_df.columns:
        print("Required columns for customer analysis not found.")
        return pd.Series(dtype=float)

    try:
        order_df[date_col] = pd.to_datetime(order_df[date_col], errors="coerce")
        return order_df.set_index(date_col).groupby(pd.Grouper(freq=freq))[client_col].nunique()
    except Exception as e:
        print(f"Error analyzing customer count: {e}")
        return pd.Series(dtype=float)


def order_value_distribution(order_df, total_col="Total"):
    """Provide descriptive statistics for the order values."""
    if total_col not in order_df.columns:
        print(f"{total_col} not found in orders data.")
        return {}
    desc = order_df[total_col].describe()
    return {
        "min": desc["min"],
        "max": desc["max"],
        "median": desc["50%"],
        "mean": desc["mean"],
        "std": desc["std"],
        "25%_quartile": desc["25%"],
        "75%_quartile": desc["75%"],
    }


def revenue_growth(order_df, freq="M", date_col="Date", total_col="Total"):
    """Compute month-over-month (or chosen freq) revenue growth percentage."""
    revenue_series = compute_revenue_over_time(order_df, freq=freq, date_col=date_col, total_col=total_col)
    if revenue_series.empty:
        return pd.Series(dtype=float)
    growth = revenue_series.pct_change() * 100
    return growth


def new_vs_returning_customers(order_df, date_col="Date", client_col="Client", freq="M"):
    """Analyze new vs. returning customers over time."""
    if date_col not in order_df.columns or client_col not in order_df.columns:
        print("Required columns for new/returning customer analysis not found.")
        return pd.DataFrame()

    try:
        order_df[date_col] = pd.to_datetime(order_df[date_col], errors="coerce")
        order_df = order_df.sort_values(date_col)
        first_purchase = order_df.groupby(client_col)[date_col].min().reset_index().rename(columns={date_col: "FirstPurchaseDate"})
        order_df = order_df.merge(first_purchase, on=client_col, how="left")

        order_df["Cohort"] = order_df["FirstPurchaseDate"].dt.to_period(freq)
        order_df["OrderPeriod"] = order_df[date_col].dt.to_period(freq)

        new_customers = order_df[order_df["Cohort"] == order_df["OrderPeriod"]].groupby("OrderPeriod")[client_col].nunique()
        total_customers = order_df.groupby("OrderPeriod")[client_col].nunique()
        returning_customers = total_customers - new_customers

        df = pd.DataFrame({"new_customers": new_customers, "returning_customers": returning_customers}).fillna(0)
        return df
    except Exception as e:
        print(f"Error analyzing new vs returning customers: {e}")
        return pd.DataFrame()


def payment_method_analysis(order_df, total_col="Total", payment_col="Paiement"):
    """Analyze revenue and AOV by payment method."""
    if total_col not in order_df.columns or payment_col not in order_df.columns:
        print("Required columns for payment method analysis not found.")
        return pd.DataFrame()
    try:
        payment_stats = order_df.groupby(payment_col)[total_col].agg(["sum", "mean", "count"]).rename(columns={"sum": "total_revenue", "mean": "avg_order_value", "count": "order_count"})
        return payment_stats.sort_values("total_revenue", ascending=False)
    except Exception as e:
        print(f"Error analyzing payment methods: {e}")
        return pd.DataFrame()


def country_analysis(order_df, total_col="Total", country_col="Livraison", client_col="Client"):
    """Analyze top countries by revenue, AOV, and unique customers."""
    if total_col not in order_df.columns or country_col not in order_df.columns or client_col not in order_df.columns:
        print("Required columns for country analysis not found.")
        return pd.DataFrame()
    try:
        country_stats = order_df.groupby(country_col).agg(
            total_revenue=(total_col, "sum"), avg_order_value=(total_col, "mean"), unique_customers=(client_col, "nunique"), order_count=(total_col, "count")
        )
        return country_stats.sort_values("total_revenue", ascending=False)
    except Exception as e:
        print(f"Error analyzing countries: {e}")
        return pd.DataFrame()


def customer_segmentation_by_value(order_df, client_col="Client", total_col="Total"):
    """
    Segment customers into tiers based on total spend:
    High-value: top 20%
    Mid-value: next 30%
    Low-value: bottom 50%
    """
    if client_col not in order_df.columns or total_col not in order_df.columns:
        print("Required columns for customer segmentation not found.")
        return pd.DataFrame()

    try:
        customer_spend = order_df.groupby(client_col)[total_col].sum().sort_values(ascending=False)
        total = customer_spend.sum()
        hv_threshold = total * 0.2
        mv_threshold = total * 0.5

        seg_df = pd.DataFrame(customer_spend).reset_index()
        seg_df["cumsum_val"] = seg_df[total_col].cumsum()

        def segment(row):
            if row["cumsum_val"] <= hv_threshold:
                return "High-value"
            elif row["cumsum_val"] <= mv_threshold:
                return "Mid-value"
            else:
                return "Low-value"

        seg_df["segment"] = seg_df.apply(segment, axis=1)
        return seg_df
    except Exception as e:
        print(f"Error in customer segmentation: {e}")
        return pd.DataFrame()


def rfm_analysis(order_df, client_col="Client", date_col="Date", total_col="Total", analysis_date=None):
    """
    Conduct RFM analysis (Recency, Frequency, Monetary).
    """
    if client_col not in order_df.columns or date_col not in order_df.columns or total_col not in order_df.columns:
        print("Required columns for RFM analysis not found.")
        return pd.DataFrame()

    try:
        order_df[date_col] = pd.to_datetime(order_df[date_col])
        if analysis_date is None:
            analysis_date = order_df[date_col].max() + pd.Timedelta(days=1)

        rfm = (
            order_df.groupby(client_col)
            .agg({date_col: lambda x: (analysis_date - x.max()).days, total_col: "sum", client_col: "count"})
            .rename(columns={date_col: "Recency", total_col: "Monetary", client_col: "Frequency"})
        )

        def score_rfm(x, col):
            quantiles = x[col].quantile([0.25, 0.5, 0.75])

            def score(val):
                if val <= quantiles[0.25]:
                    return 4
                elif val <= quantiles[0.5]:
                    return 3
                elif val <= quantiles[0.75]:
                    return 2
                else:
                    return 1

            return x[col].apply(score)

        # Recency: lower is better
        rfm["R_score"] = score_rfm(rfm, "Recency")

        def score_rev(x, col):
            quantiles = x[col].quantile([0.25, 0.5, 0.75])

            def score(val):
                if val <= quantiles[0.25]:
                    return 1
                elif val <= quantiles[0.5]:
                    return 2
                elif val <= quantiles[0.75]:
                    return 3
                else:
                    return 4

            return x[col].apply(score)

        # Frequency and Monetary: higher is better
        rfm["F_score"] = score_rev(rfm, "Frequency")
        rfm["M_score"] = score_rev(rfm, "Monetary")

        rfm["RFM_score"] = rfm["R_score"].astype(str) + rfm["F_score"].astype(str) + rfm["M_score"].astype(str)

        return rfm
    except Exception as e:
        print(f"Error in RFM analysis: {e}")
        return pd.DataFrame()


def refund_cancellation_analysis(order_df, state_col="État", total_col="Total"):
    """
    Analyze refund and cancellation rates and revenue impact.
    """
    if state_col not in order_df.columns or total_col not in order_df.columns:
        print("Required columns for refund/cancellation analysis not found.")
        return {}

    try:
        total_orders = len(order_df)
        total_revenue = order_df[total_col].sum()

        refunded = order_df[order_df[state_col].str.contains("Remboursé", na=False)]
        canceled = order_df[order_df[state_col].str.contains("Annulée", na=False)]
        delivered = order_df[order_df[state_col].str.contains("Livré", na=False)]

        refund_rate = (len(refunded) / total_orders * 100) if total_orders > 0 else 0
        cancellation_rate = (len(canceled) / total_orders * 100) if total_orders > 0 else 0

        revenue_lost_refunds = refunded[total_col].sum()
        revenue_lost_cancellations = canceled[total_col].sum()

        # Percentage of revenue that was refunded or canceled
        revenue_refund_pct = (revenue_lost_refunds / total_revenue * 100) if total_revenue > 0 else 0
        revenue_cancel_pct = (revenue_lost_cancellations / total_revenue * 100) if total_revenue > 0 else 0

        return {
            "refund_rate": refund_rate,
            "cancellation_rate": cancellation_rate,
            "revenue_lost_refunds": revenue_lost_refunds,
            "revenue_refund_pct": revenue_refund_pct,
            "revenue_lost_cancellations": revenue_lost_cancellations,
            "revenue_cancel_pct": revenue_cancel_pct,
        }
    except Exception as e:
        print(f"Error in refund/cancellation analysis: {e}")
        return {}


def order_state_analysis(order_df, state_col="État", total_col="Total"):
    """
    Break down orders by their state (État).
    - Count and percentage of total orders per state.
    - Total and percentage of total revenue per state.
    """
    if state_col not in order_df.columns or total_col not in order_df.columns:
        print("Required columns for state analysis not found.")
        return pd.DataFrame()

    try:
        total_orders = len(order_df)
        total_revenue = order_df[total_col].sum()

        state_groups = order_df.groupby(state_col)[total_col].agg(["sum", "count"]).rename(columns={"sum": "total_revenue", "count": "order_count"})
        state_groups["order_pct"] = (state_groups["order_count"] / total_orders) * 100 if total_orders > 0 else 0
        state_groups["revenue_pct"] = (state_groups["total_revenue"] / total_revenue * 100) if total_revenue > 0 else 0
        return state_groups.sort_values("total_revenue", ascending=False)
    except Exception as e:
        print(f"Error in order state analysis: {e}")
        return pd.DataFrame()


def monthly_cancellation_refund_trends(order_df, state_col="État", date_col="Date"):
    """
    Analyze monthly trends in cancellations and refunds.
    """
    if state_col not in order_df.columns or date_col not in order_df.columns:
        print("Required columns for monthly cancellation/refund trends not found.")
        return pd.DataFrame()

    try:
        order_df[date_col] = pd.to_datetime(order_df[date_col])
        order_df["Month"] = order_df[date_col].dt.to_period("M").astype(str)

        monthly_data = order_df.groupby("Month")[state_col].apply(lambda x: x.value_counts()).unstack().fillna(0)

        # Calculate monthly percentages
        monthly_totals = monthly_data.sum(axis=1)
        monthly_pct = monthly_data.div(monthly_totals, axis=0) * 100

        # Focus on "Annulée" and "Remboursé"
        interesting_states = ["Annulée", "Remboursé", "Remboursement partiel"]
        columns_to_keep = [c for c in interesting_states if c in monthly_pct.columns]
        return monthly_pct[columns_to_keep].fillna(0)
    except Exception as e:
        print(f"Error in monthly cancellation/refund trends: {e}")
        return pd.DataFrame()


def revenue_concentration(order_df, total_col="Total"):
    """
    Check revenue concentration:
    - What % of total revenue comes from the top 10% of orders?
    """
    if total_col not in order_df.columns:
        print(f"Column {total_col} not found.")
        return {}

    try:
        sorted_orders = order_df[total_col].sort_values(ascending=False)
        total_revenue = sorted_orders.sum()
        top_10pct_count = int(len(sorted_orders) * 0.1)
        top_10pct_revenue = sorted_orders.iloc[:top_10pct_count].sum() if top_10pct_count > 0 else 0
        concentration_pct = (top_10pct_revenue / total_revenue * 100) if total_revenue > 0 else 0
        return {
            "top_10pct_revenue_concentration": concentration_pct,
            "top_10pct_revenue": top_10pct_revenue,
        }
    except Exception as e:
        print(f"Error in revenue concentration analysis: {e}")
        return {}


def repeat_vs_one_time_customers(order_df, client_col="Client"):
    """
    Identify how many customers purchase only once vs. multiple times.
    Also compute what % of revenue comes from repeat customers.
    """
    if client_col not in order_df.columns:
        print("Required column for repeat vs. one-time analysis not found.")
        return {}

    try:
        customer_counts = order_df.groupby(client_col).size()
        one_time_buyers = (customer_counts == 1).sum()
        multi_buyers = (customer_counts > 1).sum()
        total_customers = len(customer_counts)

        # Revenue from multi-buyers vs one-time buyers
        # Assume a 'Total' column exists
        if "Total" in order_df.columns:
            total_revenue = order_df["Total"].sum()
            multi_buyer_list = customer_counts[customer_counts > 1].index
            multi_buyer_revenue = order_df[order_df[client_col].isin(multi_buyer_list)]["Total"].sum()
            multi_buyer_revenue_pct = (multi_buyer_revenue / total_revenue * 100) if total_revenue > 0 else 0
        else:
            multi_buyer_revenue_pct = np.nan

        return {
            "one_time_buyers": one_time_buyers,
            "multi_buyers": multi_buyers,
            "one_time_buyers_pct": (one_time_buyers / total_customers * 100) if total_customers > 0 else 0,
            "multi_buyers_pct": (multi_buyers / total_customers * 100) if total_customers > 0 else 0,
            "multi_buyer_revenue_pct": multi_buyer_revenue_pct,
        }
    except Exception as e:
        print(f"Error analyzing repeat vs one-time customers: {e}")
        return {}


def cohort_analysis(order_df, client_col="Client", date_col="Date", freq="M"):
    """
    Analyze retention by grouping customers into monthly cohorts based on their first purchase.
    Cohort analysis helps understand retention over time.
    """
    if client_col not in order_df.columns or date_col not in order_df.columns:
        print("Required columns for cohort analysis not found.")
        return pd.DataFrame()

    try:
        order_df[date_col] = pd.to_datetime(order_df[date_col])
        order_df["CohortMonth"] = order_df.groupby(client_col)[date_col].transform("min").dt.to_period(freq)
        order_df["OrderMonth"] = order_df[date_col].dt.to_period(freq)

        cohort_data = order_df.groupby(["CohortMonth", "OrderMonth"])[client_col].nunique().reset_index()
        cohort_data["CohortIndex"] = (cohort_data["OrderMonth"] - cohort_data["CohortMonth"]).apply(lambda x: x.n)

        cohort_pivot = cohort_data.pivot_table(index="CohortMonth", columns="CohortIndex", values=client_col)
        cohort_size = cohort_pivot.iloc[:, 0]
        retention = cohort_pivot.divide(cohort_size, axis=0) * 100
        return retention
    except Exception as e:
        print(f"Error in cohort analysis: {e}")
        return pd.DataFrame()


def day_of_week_analysis(order_df, date_col="Date", total_col="Total"):
    """Analyze revenue by day of the week."""
    if date_col not in order_df.columns or total_col not in order_df.columns:
        print("Required columns for day_of_week analysis not found.")
        return pd.Series(dtype=float)
    try:
        order_df[date_col] = pd.to_datetime(order_df[date_col])
        order_df["DayOfWeek"] = order_df[date_col].dt.day_name()
        return order_df.groupby("DayOfWeek")[total_col].sum().sort_values(ascending=False)
    except Exception as e:
        print(f"Error in day_of_week analysis: {e}")
        return pd.Series(dtype=float)


def hour_of_day_analysis(order_df, date_col="Date", total_col="Total"):
    """Analyze revenue by hour of the day."""
    if date_col not in order_df.columns or total_col not in order_df.columns:
        print("Required columns for hour_of_day analysis not found.")
        return pd.Series(dtype=float)
    try:
        order_df[date_col] = pd.to_datetime(order_df[date_col])
        order_df["Hour"] = order_df[date_col].dt.hour
        return order_df.groupby("Hour")[total_col].sum().sort_values(ascending=False)
    except Exception as e:
        print(f"Error in hour_of_day analysis: {e}")
        return pd.Series(dtype=float)


def year_over_year_revenue(order_df, date_col="Date", total_col="Total"):
    """
    Compute total revenue for each year and calculate year-over-year growth.

    Returns a DataFrame with columns:
    - 'year': The year
    - 'total_revenue': The total revenue for that year
    - 'yoy_growth': The percentage change compared to the previous year
    """
    if date_col not in order_df.columns or total_col not in order_df.columns:
        print("Required columns for year-over-year revenue analysis not found.")
        return pd.DataFrame()

    try:
        order_df[date_col] = pd.to_datetime(order_df[date_col], errors="coerce")
        # Group by year
        yearly_revenue = order_df.groupby(pd.Grouper(key=date_col, freq="Y")).agg({total_col: "sum"}).reset_index()
        yearly_revenue["year"] = yearly_revenue[date_col].dt.year
        yearly_revenue = yearly_revenue.rename(columns={total_col: "total_revenue"})
        yearly_revenue.drop(columns=[date_col], inplace=True)

        # Compute YoY growth
        yearly_revenue["yoy_growth"] = yearly_revenue["total_revenue"].pct_change() * 100
        return yearly_revenue
    except Exception as e:
        print(f"Error in year_over_year_revenue: {e}")
        return pd.DataFrame()


def year_over_year_orders(order_df, date_col="Date"):
    """
    Compute the total number of orders for each year and calculate year-over-year growth.

    Returns a DataFrame with columns:
    - 'year': The year
    - 'total_orders': The total number of orders that year
    - 'yoy_growth': The percentage change compared to the previous year
    """
    if date_col not in order_df.columns:
        print("Required date column for year-over-year orders analysis not found.")
        return pd.DataFrame()

    try:
        order_df[date_col] = pd.to_datetime(order_df[date_col], errors="coerce")
        yearly_orders = order_df.set_index(date_col).resample("Y").size().reset_index(name="total_orders")
        yearly_orders["year"] = yearly_orders[date_col].dt.year
        yearly_orders.drop(columns=[date_col], inplace=True)

        yearly_orders["yoy_growth"] = yearly_orders["total_orders"].pct_change() * 100
        return yearly_orders
    except Exception as e:
        print(f"Error in year_over_year_orders: {e}")
        return pd.DataFrame()


def year_over_year_aov(order_df, date_col="Date", total_col="Total"):
    """
    Compute the average order value (AOV) for each year and calculate year-over-year growth.

    Returns a DataFrame with columns:
    - 'year': The year
    - 'yearly_aov': The average order value for that year
    - 'yoy_growth': The percentage change in AOV compared to the previous year
    """
    if date_col not in order_df.columns or total_col not in order_df.columns:
        print("Required columns for year-over-year AOV analysis not found.")
        return pd.DataFrame()

    try:
        order_df[date_col] = pd.to_datetime(order_df[date_col], errors="coerce")
        # Compute yearly AOV = total revenue per year / total orders per year
        yearly_data = order_df.set_index(date_col).resample("Y").agg({total_col: "sum", "Référence": "count"}).reset_index()
        yearly_data["year"] = yearly_data[date_col].dt.year
        yearly_data.rename(columns={total_col: "total_revenue", "Référence": "total_orders"}, inplace=True)
        yearly_data["yearly_aov"] = yearly_data["total_revenue"] / yearly_data["total_orders"]

        # Compute YoY growth for AOV
        yearly_data["yoy_growth"] = yearly_data["yearly_aov"].pct_change() * 100
        return yearly_data[["year", "yearly_aov", "yoy_growth"]]
    except Exception as e:
        print(f"Error in year_over_year_aov: {e}")
        return pd.DataFrame()


###############################
# NEWLY ADDED METRICS
###############################


def calculate_clv(order_df, total_col="Total", client_col="Client", date_col="Date"):
    """
    Calculate Customer Lifetime Value (CLV).
    A simplified model:
    CLV = Average Order Value * Purchase Frequency * Average Customer Lifetime in months (assume 12)
    """
    if total_col not in order_df.columns or client_col not in order_df.columns or date_col not in order_df.columns:
        print("Required columns for CLV calculation not found.")
        return np.nan
    try:
        # Ensure date conversion
        order_df[date_col] = pd.to_datetime(order_df[date_col])

        # Average order value
        avg_order_value = order_df[total_col].mean()

        # Purchase frequency (average number of orders per customer)
        purchase_frequency = order_df.groupby(client_col).size().mean()

        # Assuming an average customer lifetime of 12 months
        customer_lifetime = 12

        clv = avg_order_value * purchase_frequency * customer_lifetime
        return clv
    except Exception as e:
        print(f"Error calculating CLV: {e}")
        return np.nan


def run_analysis():
    """
    Run a series of analyses on the cleaned data and return insights.
    """
    cart_df, order_df = load_cleaned_data()
    if cart_df is None or order_df is None:
        print("Could not load cleaned data. Analysis cannot proceed.")
        return {}

    insights = {}

    # Existing Metrics
    insights["basic_kpis"] = basic_kpis(order_df)
    insights["monthly_revenue"] = compute_revenue_over_time(order_df, freq="M")
    insights["average_order_value"] = compute_average_order_value(order_df)
    insights["cart_abandonment_rate"] = compute_cart_abandonment_rate(cart_df, order_df)
    insights["monthly_unique_customers"] = analyze_customer_count(order_df)
    insights["order_value_distribution"] = order_value_distribution(order_df)
    insights["monthly_revenue_growth"] = revenue_growth(order_df, freq="M")
    insights["customer_cohorts"] = new_vs_returning_customers(order_df)
    insights["payment_method_analysis"] = payment_method_analysis(order_df)
    insights["country_analysis"] = country_analysis(order_df)
    insights["customer_segmentation"] = customer_segmentation_by_value(order_df)
    insights["rfm_analysis"] = rfm_analysis(order_df)
    insights["refund_cancellation_analysis"] = refund_cancellation_analysis(order_df)
    insights["order_state_analysis"] = order_state_analysis(order_df)
    insights["monthly_cancellation_refund_trends"] = monthly_cancellation_refund_trends(order_df)
    insights["cohort_retention"] = cohort_analysis(order_df)
    insights["day_of_week_revenue"] = day_of_week_analysis(order_df)
    insights["hour_of_day_revenue"] = hour_of_day_analysis(order_df)
    insights["revenue_concentration"] = revenue_concentration(order_df)
    insights["clv"] = calculate_clv(order_df)
    insights["repeat_vs_one_time_customers"] = repeat_vs_one_time_customers(order_df)

    # New Year-over-Year Metrics
    insights["year_over_year_revenue"] = year_over_year_revenue(order_df)
    insights["year_over_year_orders"] = year_over_year_orders(order_df)
    insights["year_over_year_aov"] = year_over_year_aov(order_df)

    return insights


if __name__ == "__main__":
    results = run_analysis()
    if results:
        print("Enhanced Analysis Results with More Advanced Metrics:")
        for key, value in results.items():
            print(f"{key}:")
            print(value, "\n")
