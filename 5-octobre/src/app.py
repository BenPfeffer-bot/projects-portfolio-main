import streamlit as st
import plotly.express as px
import pandas as pd
from analysis import run_analysis
from streamlit_lottie import st_lottie
import json

st.set_page_config(page_title="E-commerce Analytics", page_icon=":chart_with_upwards_trend:", layout="wide")


# Load Lottie Animations
def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)


# Example lottie animation files (ensure these JSON files exist in your directory)
lottie_dashboard = load_lottiefile("lottie_dashboard.json")  # A dashboard-themed animation
lottie_growth = load_lottiefile("lottie_growth.json")  # A growth-themed animation
lottie_success = load_lottiefile("lottie_success.json")  # A success or celebration animation

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

year_over_year_aov = insights.get("year_over_year_aov", pd.DataFrame())
aov_yoy_growth = year_over_year_aov["yoy_growth"].iloc[-1] if not year_over_year_aov.empty else None

########################
# Sidebar: Filters & Theme
########################
st.sidebar.title("Settings")
date_range = st.sidebar.date_input("Date Range", [])
selected_country = st.sidebar.selectbox("Country", options=["All"] + (list(country_df.index) if not country_df.empty else []))

# A toggle for dark mode simulation
dark_mode = st.sidebar.checkbox("Dark Mode", value=False)

chart_type = st.sidebar.radio("Chart Type", ["Line", "Bar"], index=0)

st.sidebar.markdown("Use these filters to refine your data view once implemented.")

with st.sidebar.expander("Help / Documentation"):
    st.write("""
    **How to Use:**
    - Use filters to focus on a time range or a country.
    - Hover over charts for detailed tooltips.
    - Toggle Dark Mode for a different look.
    """)

########################
# Custom CSS & Fonts
########################
# Google Font & custom animations
st.markdown(
    """
<link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600&display=swap" rel="stylesheet">
<style>
body {
    font-family: 'Montserrat', sans-serif;
    transition: background-color 0.5s ease;
}

header, .stTabs [role="tab"] {
    font-weight:600;
}

.dark-mode {
    background-color: #1E1E1E !important;
    color: #FAFAFA !important;
}

.light-mode {
    background-color: #FFFFFF !important;
    color: #333333 !important;
}

.top-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 10px 20px;
    position: sticky;
    top:0;
    background: linear-gradient(90deg, #1B4F72, #3498DB);
    z-index:9999;
}
.top-bar h1 {
    color: white;
    margin:0;
    font-size:1.5rem;
    letter-spacing:1px;
}
.top-bar img {
    height: 40px;
}
.kpi-container {
    display:flex; 
    gap:20px;
    flex-wrap:wrap;
    margin-bottom:20px;
}
.kpi-card {
    background: #F7F9F9;
    border-radius: 10px;
    padding: 20px;
    text-align: center;
    border: 1px solid #EAECEE;
    flex:1;
    min-width:200px;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
    position:relative;
    overflow:hidden;
    cursor:default;
}
.kpi-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 5px 15px rgba(0,0,0,0.1);
}
.kpi-card:after {
    content:'';
    position:absolute;
    top:-50%;
    left:-50%;
    width:200%;
    height:200%;
    background: radial-gradient(circle, rgba(27,79,114,0.15), transparent 60%);
    animation: rotate 10s linear infinite;
    z-index:-1;
}
@keyframes rotate {
    0% {transform:rotate(0deg);}
    100% {transform:rotate(360deg);}
}
.kpi-title {
    font-size: 0.9rem;
    color: #5D6D7E;
    margin-bottom: 0.5em;
    text-transform:uppercase;
    letter-spacing:1px;
}
.kpi-value {
    font-size: 1.6rem;
    color: #1B4F72;
    font-weight: bold;
}
.chart-container {
    background: #FFFFFF;
    border-radius:10px;
    padding:20px;
    margin-bottom:20px;
    box-shadow:0 1px 3px rgba(0,0,0,0.1);
}
.dark-mode .chart-container {
    background:#2C2C2C;
}
.tab-label {
    font-weight:600;
}
</style>
""",
    unsafe_allow_html=True,
)

if dark_mode:
    st.markdown("<style>body{background-color:#2C2C2C;color:#FAFAFA;}</style>", unsafe_allow_html=True)
else:
    st.markdown("<style>body{background-color:#FFFFFF;color:#333333;}</style>", unsafe_allow_html=True)

########################
# Top Bar with Animation
########################
st.markdown(
    """
<div class="top-bar">
    <h1>📊 E-Commerce Analytics Dashboard</h1>
</div>
""",
    unsafe_allow_html=True,
)

# Lottie animation at the top for a welcome banner
st_lottie(lottie_dashboard, height=200, key="dashboard_animation")

########################
# KPI Section with Hover & Animation
########################
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(
        f"""
    <div class="kpi-card">
        <div class="kpi-title">Total Orders</div>
        <div class="kpi-value">{basic_kpis.get('total_orders', 'N/A')}</div>
    </div>
    """,
        unsafe_allow_html=True,
    )
with col2:
    rev = f"{basic_kpis.get('total_revenue', 0):.2f}"
    st.markdown(
        f"""
    <div class="kpi-card">
        <div class="kpi-title">Total Revenue</div>
        <div class="kpi-value">€{rev}</div>
    </div>
    """,
        unsafe_allow_html=True,
    )
with col3:
    st.markdown(
        f"""
    <div class="kpi-card">
        <div class="kpi-title">CLV</div>
        <div class="kpi-value">€{clv:.2f}</div>
    </div>
    """,
        unsafe_allow_html=True,
    )
with col4:
    yoy_val = f"{aov_yoy_growth:.2f}%" if aov_yoy_growth is not None else "N/A"
    st.markdown(
        f"""
    <div class="kpi-card">
        <div class="kpi-title">AOV YoY Growth</div>
        <div class="kpi-value">{yoy_val}</div>
    </div>
    """,
        unsafe_allow_html=True,
    )


######################
# Chart Preparation & Animations
######################
monthly_revenue_df = monthly_revenue.reset_index()
monthly_revenue_df.columns = ["Date", "Revenue"]
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
    fig_yoy_revenue = px.bar(year_over_year_revenue, x="year", y="total_revenue", title="Year-over-Year Revenue", color_discrete_sequence=["#1B4F72"], hover_data=["yoy_growth"])
    fig_yoy_revenue.update_traces(hovertemplate="Year: %{x}<br>Revenue: €%{y:.2f}<br>Growth: %{customdata[0]:.2f}%<extra></extra>")
else:
    fig_yoy_revenue = px.bar(title="No YoY Revenue Data", color_discrete_sequence=["#1B4F72"])

if not payment_method_df.empty:
    fig_payment = px.bar(
        payment_method_df.reset_index(), x="Paiement", y="total_revenue", hover_data=["avg_order_value", "order_count"], title="Revenue by Payment Method", color_discrete_sequence=["#1B4F72"]
    )
else:
    fig_payment = px.bar(title="No Payment Method Data", color_discrete_sequence=["#1B4F72"])

if not country_df.empty:
    fig_country = px.bar(
        country_df.reset_index().head(10),
        x="Livraison",
        y="total_revenue",
        hover_data=["avg_order_value", "unique_customers", "order_count"],
        title="Top 10 Countries by Revenue",
        color_discrete_sequence=["#1B4F72"],
    )
else:
    fig_country = px.bar(title="No Country Data Available", color_discrete_sequence=["#1B4F72"])

######################
# Tabs with Animated Icons & Lottie
######################
tabs = st.tabs(["📅 Time-Based Metrics", "🌍 Geography & Payment", "🔧 Order States", "💎 Revenue Concentration", "👥 Customer & RFM", "💳 Refunds & Cancellations", "🔄 Cohort Retention"])

with tabs[0]:
    st.subheader("Time-Based Metrics")
    # Chart type toggle from sidebar
    if chart_type == "Bar":
        fig_monthly_alt = px.bar(monthly_revenue_df, x="Date", y="Revenue", title="Monthly Revenue Over Time", color_discrete_sequence=["#1B4F72"])
        fig_monthly_alt.update_traces(hovertemplate=hover_template_line)
        st.plotly_chart(fig_monthly_alt, use_container_width=True)
    else:
        st.plotly_chart(fig_monthly_revenue, use_container_width=True)

    st.plotly_chart(fig_yoy_revenue, use_container_width=True)

    if not monthly_revenue_growth.empty:
        st_lottie(lottie_growth, height=150, key="growth_animation")
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
    st.subheader("Order States Analysis")
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
        st.write("No State Data Available")

with tabs[3]:
    st.subheader("Revenue Concentration")
    st.write(f"Top 10% of orders account for **{revenue_concentration.get('top_10pct_revenue_concentration',0):.2f}%** of total revenue.")
    st_lottie(lottie_success, height=100, key="success_animation")

with tabs[4]:
    st.subheader("Customer & RFM Analysis")
    if not rfm_df.empty:
        st.markdown("### RFM Scores Distribution")
        c5, c6 = st.columns(2)
        fig_r_score = px.histogram(rfm_df, x="Recency", nbins=30, title="Recency Distribution", color_discrete_sequence=["#1B4F72"])
        fig_f_score = px.histogram(rfm_df, x="Frequency", nbins=30, title="Frequency Distribution", color_discrete_sequence=["#1B4F72"])
        c5.plotly_chart(fig_r_score, use_container_width=True)
        c6.plotly_chart(fig_f_score, use_container_width=True)

        st.markdown("### RFM Segments")
        if "segment" in rfm_df.columns:
            seg_counts = rfm_df["segment"].value_counts().reset_index()
            seg_counts.columns = ["Segment", "Count"]
            fig_segments = px.bar(seg_counts, x="Segment", y="Count", title="RFM Segments Count", color_discrete_sequence=["#1B4F72"])
            st.plotly_chart(fig_segments, use_container_width=True)

    if not customer_cohorts.empty:
        st.markdown("### New vs. Returning Customers Over Time")
        fig_cohorts = px.line(customer_cohorts.reset_index(), x="OrderPeriod", y="new_customers", title="New Customers Over Time", color_discrete_sequence=["#1B4F72"])
        st.plotly_chart(fig_cohorts, use_container_width=True)

with tabs[5]:
    st.subheader("Refunds & Cancellations")
    if refund_cancellation:
        st.write("**Refund & Cancellation Analysis:**")
        for k, v in refund_cancellation.items():
            st.write(f"- **{k.capitalize().replace('_',' ')}:** {v}")

    if not monthly_refund_trends.empty:
        st.markdown("### Monthly Refund & Cancellation Trends")
        fig_refund_trends = px.line(
            monthly_refund_trends.reset_index(), x="Month", y=monthly_refund_trends.columns, title="Monthly Cancellation/Refund Rates", color_discrete_sequence=["#1B4F72", "#2980B9", "#3498DB"]
        )
        fig_refund_trends.update_traces(hovertemplate="Month: %{x}<br>Rate: %{y:.2f}%<extra></extra>")
        st.plotly_chart(fig_refund_trends, use_container_width=True)

with tabs[6]:
    st.subheader("Cohort Retention")
    if not cohort_retention.empty:
        st.markdown("### Customer Retention Cohorts")
        fig_cohort = px.imshow(cohort_retention, aspect="auto", title="Cohort Retention (%)", color_continuous_scale="Blues")
        fig_cohort.update_traces(hovertemplate="Cohort: %{y}<br>Period: %{x}<br>Retention: %{z:.2f}%<extra></extra>")
        st.plotly_chart(fig_cohort, use_container_width=True)


st.markdown("---")
st.markdown("**Tip:** Use the sidebar filters to refine the analysis. Hover over charts and KPIs for more info.")
