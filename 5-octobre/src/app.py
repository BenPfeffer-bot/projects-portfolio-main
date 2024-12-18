import streamlit as st
import plotly.express as px
import pandas as pd
from analysis import run_analysis

st.set_page_config(page_title="E-commerce Analytics Dashboard", page_icon=":chart_with_upwards_trend:", layout="wide")

# Run analysis once and store the results
insights = run_analysis()

# Extract data
monthly_revenue = insights.get("monthly_revenue", pd.Series(dtype=float))
year_over_year_revenue = insights.get("year_over_year_revenue", pd.DataFrame())
payment_method_df = insights.get("payment_method_analysis", pd.DataFrame())
country_df = insights.get("country_analysis", pd.DataFrame())
order_state_df = insights.get("order_state_analysis", pd.DataFrame())
revenue_concentration = insights.get("revenue_concentration", {})
basic_kpis = insights.get("basic_kpis", {})
cart_abandonment_rate = insights.get("cart_abandonment_rate", 0.0)
clv = insights.get("clv", 0.0)
monthly_revenue_growth = insights.get("monthly_revenue_growth", pd.Series(dtype=float))
rfm_df = insights.get("rfm_analysis", pd.DataFrame())
customer_cohorts = insights.get("customer_cohorts", pd.DataFrame())
refund_cancellation = insights.get("refund_cancellation_analysis", {})
monthly_refund_trends = insights.get("monthly_cancellation_refund_trends", pd.DataFrame())
cohort_retention = insights.get("cohort_retention", pd.DataFrame())
customer_segmentation = insights.get("customer_segmentation", pd.DataFrame())

# Convert Period objects to strings in customer_cohorts if present
if not customer_cohorts.empty:
    customer_cohorts = customer_cohorts.copy()
    if customer_cohorts.index.dtype == "period[M]":
        customer_cohorts.index = customer_cohorts.index.astype(str)
    if "OrderPeriod" in customer_cohorts.columns and customer_cohorts["OrderPeriod"].dtype == "period[M]":
        customer_cohorts["OrderPeriod"] = customer_cohorts["OrderPeriod"].astype(str)

# Convert Period objects to strings in cohort_retention if present
if not cohort_retention.empty:
    cohort_retention = cohort_retention.copy()
    if cohort_retention.index.dtype == "period[M]":
        cohort_retention.index = cohort_retention.index.astype(str)
    if isinstance(cohort_retention.columns, pd.PeriodIndex):
        cohort_retention.columns = cohort_retention.columns.astype(str)

# Compute additional KPI: AOV YoY growth (if year_over_year_aov is available)
year_over_year_aov = insights.get("year_over_year_aov", pd.DataFrame())
if not year_over_year_aov.empty:
    aov_yoy_growth = year_over_year_aov["yoy_growth"].iloc[-1]
else:
    aov_yoy_growth = None

############
# SIDEBAR
############
st.sidebar.title("Filters & Settings")
date_range = st.sidebar.date_input("Select Date Range", [])
selected_country = st.sidebar.selectbox("Select Country", options=["All"] + (list(country_df.index) if not country_df.empty else []))
chart_type = st.sidebar.radio("Chart Type", ["Line", "Bar"], index=0)

st.sidebar.markdown("Use these filters (once implemented) to refine your data view.")

# Add a help/info section in the sidebar
with st.sidebar.expander("Help / Documentation"):
    st.write("""
    **Instructions:**
    - Use the filters above to narrow down the analysis by date or country.
    - Hover over charts to see detailed tooltips.
    - Switch chart types in certain tabs for a different perspective.
    - KPI definitions:
      - **CLV**: Estimated Customer Lifetime Value.
      - **Cart Abandonment Rate**: Percentage of carts not completed as orders.
      - **AOV YoY Growth**: Year-over-year growth in Average Order Value.
    """)

############
# CSS for styling
############
st.markdown(
    """
<style>
h1 {
    font-family: "Helvetica", sans-serif;
    font-weight: 700;
    margin-bottom: 0.5em;
    color: #2E4053;
}
.kpi-card {
    background-color: #F7F9F9;
    border-radius: 8px;
    padding: 20px;
    text-align: center;
    border: 1px solid #EAECEE;
    margin-bottom: 10px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}
.kpi-title {
    font-size: 1rem;
    color: #5D6D7E;
    margin-bottom: 0.5em;
}
.kpi-value {
    font-size: 1.5rem;
    color: #1B4F72;
    font-weight: bold;
}
.stTabs [role="tab"] {
    font-size:1rem;
    font-weight:600;
    color:#2E4053;
}
</style>
""",
    unsafe_allow_html=True,
)

st.title(":bar_chart: E-commerce Analytics Dashboard")

############
# KPI SECTION
############
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        f"""
    <div class="kpi-card" title="Total number of orders placed.">
        <div class="kpi-title">🛍️ Total Orders</div>
        <div class="kpi-value">{basic_kpis.get('total_orders', 'N/A')}</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

with col2:
    rev = f"{basic_kpis.get('total_revenue', 0):.2f}"
    st.markdown(
        f"""
    <div class="kpi-card" title="Sum of all order values.">
        <div class="kpi-title">💰 Total Revenue</div>
        <div class="kpi-value">€{rev}</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        f"""
    <div class="kpi-card" title="Estimated total revenue from a customer over their lifetime.">
        <div class="kpi-title">👤 CLV</div>
        <div class="kpi-value">€{clv:.2f}</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

with col4:
    if aov_yoy_growth is not None:
        yoy_val = f"{aov_yoy_growth:.2f}%"
    else:
        yoy_val = "N/A"
    st.markdown(
        f"""
    <div class="kpi-card" title="Year-over-year growth in the average order value.">
        <div class="kpi-title">📈 AOV YoY Growth</div>
        <div class="kpi-value">{yoy_val}</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

col5, col6, col7, col8 = st.columns(4)
with col5:
    uniq_cust = basic_kpis.get("unique_customers", "N/A")
    st.markdown(
        f"""
    <div class="kpi-card" title="Number of distinct customers who made at least one purchase.">
        <div class="kpi-title">👥 Unique Customers</div>
        <div class="kpi-value">{uniq_cust}</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

with col6:
    aoc = f"{basic_kpis.get('avg_orders_per_customer', 0):.2f}"
    st.markdown(
        f"""
    <div class="kpi-card" title="Average number of orders per customer.">
        <div class="kpi-title">🧮 Avg Orders/Customer</div>
        <div class="kpi-value">{aoc}</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

with col7:
    st.markdown(
        f"""
    <div class="kpi-card" title="Percentage of carts created but not converted into orders.">
        <div class="kpi-title">🚪 Cart Abandonment</div>
        <div class="kpi-value">{cart_abandonment_rate:.2f}%</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

top_10_conc = revenue_concentration.get("top_10pct_revenue_concentration", 0)
with col8:
    st.markdown(
        f"""
    <div class="kpi-card" title="Percentage of revenue contributed by top 10% highest-value orders.">
        <div class="kpi-title">🔝 Top 10% Rev%</div>
        <div class="kpi-value">{top_10_conc:.2f}%</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

######################
# CHART PREPARATION
######################
monthly_revenue_df = monthly_revenue.reset_index()
monthly_revenue_df.columns = ["Date", "Revenue"]

# Add hover templates
hover_template_line = "Date: %{x}<br>Value: €%{y:.2f}<extra></extra>"

fig_monthly_revenue = px.line(
    monthly_revenue_df,
    x="Date",
    y="Revenue",
    title="Monthly Revenue Over Time",
    color_discrete_sequence=["#1B4F72"],
)
fig_monthly_revenue.update_traces(hovertemplate=hover_template_line)
fig_monthly_revenue.update_layout(hovermode="x unified")

if not year_over_year_revenue.empty:
    fig_yoy_revenue = px.bar(
        year_over_year_revenue,
        x="year",
        y="total_revenue",
        title="Year-over-Year Revenue",
        color_discrete_sequence=["#1B4F72"],
        hover_data=["yoy_growth"],
    )
    fig_yoy_revenue.update_traces(hovertemplate="Year: %{x}<br>Revenue: €%{y:.2f}<br>YoY Growth: %{customdata[0]:.2f}%<extra></extra>")
else:
    fig_yoy_revenue = px.bar(title="No YoY Revenue Data Available", color_discrete_sequence=["#1B4F72"])

if not payment_method_df.empty:
    fig_payment = px.bar(
        payment_method_df.reset_index(),
        x="Paiement",
        y="total_revenue",
        hover_data=["avg_order_value", "order_count"],
        title="Revenue by Payment Method",
        color_discrete_sequence=["#1B4F72"],
    )
    fig_payment.update_traces(hovertemplate="Payment Method: %{x}<br>Total Revenue: €%{y:.2f}<br>Avg Order Value: €%{customdata[0]:.2f}<br>Order Count: %{customdata[1]}<extra></extra>")
else:
    fig_payment = px.bar(title="No Payment Method Data Available", color_discrete_sequence=["#1B4F72"])

if not country_df.empty:
    fig_country = px.bar(
        country_df.reset_index().head(10),
        x="Livraison",
        y="total_revenue",
        hover_data=["avg_order_value", "unique_customers", "order_count"],
        title="Top 10 Countries by Revenue",
        color_discrete_sequence=["#1B4F72"],
    )
    fig_country.update_traces(
        hovertemplate="Country: %{x}<br>Total Revenue: €%{y:.2f}<br>Avg Order Value: €%{customdata[0]:.2f}<br>Unique Customers: %{customdata[1]}<br>Order Count: %{customdata[2]}<extra></extra>"
    )
else:
    fig_country = px.bar(title="No Country Data Available", color_discrete_sequence=["#1B4F72"])


######################
# TABS
######################
tabs = st.tabs(["Time-Based Metrics", "Geography & Payment", "Order States", "Revenue Concentration", "Customer & RFM Analysis", "Refunds & Cancellations", "Cohort Retention"])

with tabs[0]:
    st.subheader("Time-Based Metrics")
    # Let the user switch between line and bar chart for monthly revenue
    if chart_type == "Bar":
        fig_monthly_revenue_alt = px.bar(monthly_revenue_df, x="Date", y="Revenue", title="Monthly Revenue Over Time", color_discrete_sequence=["#1B4F72"])
        fig_monthly_revenue_alt.update_traces(hovertemplate=hover_template_line)
        st.plotly_chart(fig_monthly_revenue_alt, use_container_width=True)
    else:
        st.plotly_chart(fig_monthly_revenue, use_container_width=True)

    st.plotly_chart(fig_yoy_revenue, use_container_width=True)

    if not monthly_revenue_growth.empty:
        st.subheader("Monthly Revenue Growth (%)")
        growth_df = monthly_revenue_growth.reset_index()
        growth_df.columns = ["Date", "Growth"]
        fig_growth = px.line(growth_df, x="Date", y="Growth", title="Monthly Revenue Growth", color_discrete_sequence=["#1B4F72"])
        fig_growth.update_traces(hovertemplate="Date: %{x}<br>Growth: %{y:.2f}%<extra></extra>")
        st.plotly_chart(fig_growth, use_container_width=True)

with tabs[1]:
    st.subheader("Geography & Payment Methods")
    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(fig_country, use_container_width=True)
    with c4:
        st.plotly_chart(fig_payment, use_container_width=True)

with tabs[2]:
    st.subheader("Order State Analysis")
    if not order_state_df.empty:
        selected_state = st.selectbox("Select State:", order_state_df.index)
        row = order_state_df.loc[selected_state]
        state_metrics_df = pd.DataFrame(
            {"Metric": ["order_count", "total_revenue", "order_pct", "revenue_pct"], "Value": [row["order_count"], row["total_revenue"], row["order_pct"], row["revenue_pct"]]}
        )
        fig_state = px.bar(state_metrics_df, x="Metric", y="Value", title=f"Metrics for State: {selected_state}", color_discrete_sequence=["#1B4F72"])
        fig_state.update_traces(hovertemplate="%{x}: %{y:.2f}<extra></extra>")
        st.plotly_chart(fig_state, use_container_width=True)
    else:
        st.write("No State Analysis Data Available")

with tabs[3]:
    st.subheader("Revenue Concentration")
    st.write(f"Top 10% of orders account for **{top_10_conc:.2f}%** of total revenue.")

with tabs[4]:
    st.subheader("Customer & RFM Analysis")
    if not rfm_df.empty:
        st.markdown("### RFM Scores Distribution")
        c5, c6 = st.columns(2)
        fig_r_score = px.histogram(rfm_df, x="Recency", nbins=30, title="Recency Distribution", color_discrete_sequence=["#1B4F72"])
        fig_f_score = px.histogram(rfm_df, x="Frequency", nbins=30, title="Frequency Distribution", color_discrete_sequence=["#1B4F72"])
        fig_r_score.update_traces(hovertemplate="Recency: %{x}<br>Count: %{y}<extra></extra>")
        fig_f_score.update_traces(hovertemplate="Frequency: %{x}<br>Count: %{y}<extra></extra>")
        c5.plotly_chart(fig_r_score, use_container_width=True)
        c6.plotly_chart(fig_f_score, use_container_width=True)

        st.markdown("### RFM Segments")
        if "segment" in rfm_df.columns:
            seg_counts = rfm_df["segment"].value_counts().reset_index()
            seg_counts.columns = ["Segment", "Count"]
            fig_segments = px.bar(seg_counts, x="Segment", y="Count", title="RFM Segments Count", color_discrete_sequence=["#1B4F72"])
            fig_segments.update_traces(hovertemplate="Segment: %{x}<br>Count: %{y}<extra></extra>")
            st.plotly_chart(fig_segments, use_container_width=True)

    if not customer_cohorts.empty:
        st.markdown("### New vs Returning Customers Over Time")
        fig_cohorts = px.line(customer_cohorts.reset_index(), x="OrderPeriod", y="new_customers", title="New Customers Over Time", color_discrete_sequence=["#1B4F72"])
        fig_cohorts.update_traces(hovertemplate="Period: %{x}<br>New Customers: %{y}<extra></extra>")
        st.plotly_chart(fig_cohorts, use_container_width=True)

with tabs[5]:
    st.subheader("Refunds & Cancellations")
    if refund_cancellation:
        # Create KPI cards with animated values
        col1, col2, col3, col4 = st.columns(4)

        # Custom CSS for transparent animated KPI cards
        st.markdown(
            """
        <style>
        .refund-kpi {
            background: linear-gradient(45deg, rgba(27,79,114,0.1), rgba(52,152,219,0.1));
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            animation: gradient 3s ease infinite;
            backdrop-filter: blur(10px);
            transition: all 0.3s ease;
        }
        .refund-kpi:hover {
            transform: translateY(-5px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        @keyframes gradient {
            0% {background-position: 0% 50%;}
            50% {background-position: 100% 50%;}
            100% {background-position: 0% 50%;}
        }
        .metric-value {
            font-size: 24px;
            font-weight: bold;
            background: linear-gradient(45deg, #1B4F72, #3498DB);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            animation: number-animation 2s ease-out;
        }
        @keyframes number-animation {
            from {opacity: 0; transform: translateY(20px);}
            to {opacity: 1; transform: translateY(0);}
        }
        </style>
        """,
            unsafe_allow_html=True,
        )

        # Display KPI cards with animations
        metrics = list(refund_cancellation.items())
        for i, (k, v) in enumerate(metrics):
            with [col1, col2, col3][i % 3]:
                st.markdown(
                    f"""
                <div class="refund-kpi">
                    <div>{k.capitalize().replace('_',' ')}</div>
                    <div class="metric-value">{v:.2f}%</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

    if not monthly_refund_trends.empty:
        st.markdown("### Monthly Refund & Cancellation Trends")
        fig_refund_trends = px.line(
            monthly_refund_trends.reset_index(), x="Month", y=monthly_refund_trends.columns, title="Monthly Cancellation/Refund Rates", color_discrete_sequence=["#1B4F72", "#3498DB", "#2980B9"]
        )

        # Enhanced hover template with transparent background
        fig_refund_trends.update_traces(
            hovertemplate="<b>%{y:.2f}%</b><br>" + "Month: %{x}<br>" + "<extra></extra>", hoverlabel=dict(bgcolor="rgba(255,255,255,0.8)", font_size=12, font_family="Arial")
        )

        # Enhance chart appearance
        fig_refund_trends.update_layout(hovermode="x unified", plot_bgcolor="rgba(255,255,255,0.9)", paper_bgcolor="rgba(0,0,0,0)", xaxis_title="", yaxis_title="Rate (%)", legend_title_text="")

        st.plotly_chart(fig_refund_trends, use_container_width=True)

with tabs[6]:
    st.subheader("Cohort Retention")
    if not cohort_retention.empty:
        st.markdown("### Customer Retention Cohorts")
        fig_cohort = px.imshow(cohort_retention, aspect="auto", title="Cohort Retention (%)", color_continuous_scale="Blues")
        fig_cohort.update_traces(hovertemplate="Cohort: %{y}<br>Period: %{x}<br>Retention: %{z:.2f}%<extra></extra>")
        st.plotly_chart(fig_cohort, use_container_width=True)

st.markdown("---")
st.markdown("**Tip:** Use the sidebar filters (once implemented) to refine the analysis. Hover over charts and KPIs for more details.**")
st.markdown("<small>Dashboard by Your Company – Further Enhanced Layout, Visuals & Analytics</small>", unsafe_allow_html=True)
