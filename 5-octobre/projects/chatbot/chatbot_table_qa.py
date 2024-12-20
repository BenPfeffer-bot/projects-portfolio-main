# projects/chatbot/chatbot_table_qa.py
import os
import sys
import pandas as pd

sys.path.append("/Users/benpfeffer/Library/Mobile Documents/com~apple~CloudDocs/projects-portfolio-main/5-octobre")
from src.config import load_logger
from src.data_preprocessing import preprocess_data
from src.analysis import run_analysis
from table_qa import TableQuestionAnswering

logger = load_logger()

# Preprocess and load data
cart_df, order_df = preprocess_data()

# Run analysis to get insights
insights = run_analysis(cart_df, order_df)

# Initialize the TAPAS Q&A model
tqa = TableQuestionAnswering()

# Select a DataFrame from insights that might be interesting to query with TAPAS.
# For demonstration, let's pick "country_analysis" which has columns like total_revenue, avg_order_value, etc.
country_analysis_df = insights.get("country_analysis")
payment_method_df = insights.get("payment_method_analysis")


def is_metric_query(query: str) -> bool:
    """
    Very simple heuristic to detect if the user is asking about a known metric.
    For instance, keywords like 'revenue', 'orders', 'customers' might indicate a metric query.
    """
    metric_keywords = ["revenue", "orders", "customers", "average order value", "aov", "clv", "kpi", "growth", "abandonment"]
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in metric_keywords)


def answer_metric_query(query: str, insights_dict: dict) -> str:
    """
    Try to provide an answer to a metric-related query using the insights dictionary.
    This is a simplistic approach. In a real scenario, you might use NLP to parse exactly what the user wants.
    """
    # Example handling:
    # If user asks "What is the total revenue?", we can look into insights["basic_kpis"]["total_revenue"]
    q_lower = query.lower()
    # Check some common metrics
    if "total revenue" in q_lower:
        rev = insights_dict.get("basic_kpis", {}).get("total_revenue", None)
        if rev is not None:
            return f"The total revenue is {rev:.2f}."
    if "average order value" in q_lower or "aov" in q_lower:
        aov = insights_dict.get("average_order_value", None)
        if aov is not None:
            return f"The average order value (AOV) is {aov:.2f}."
    if "total orders" in q_lower:
        total_orders = insights_dict.get("basic_kpis", {}).get("total_orders", None)
        if total_orders is not None:
            return f"The total number of orders is {total_orders}."
    if "unique customers" in q_lower:
        unique_cust = insights_dict.get("basic_kpis", {}).get("unique_customers", None)
        if unique_cust is not None:
            return f"The number of unique customers is {unique_cust}."

    # Add more conditions for other metrics as needed
    if "clv" in q_lower:
        clv = insights_dict.get("clv", None)
        if clv is not None:
            return f"The estimated Customer Lifetime Value (CLV) is {clv:.2f}."

    # If we couldn't find a direct match:
    return "I don't have a direct metric for that, can you be more specific?"


def is_table_question(query: str) -> bool:
    """
    Detect if the question might be answered by querying a table via TAPAS.
    For simplicity, if query mentions 'country', 'payment method', or 'table', we route to TAPAS.
    """
    keywords = ["country", "countries", "payment method", "methods", "table"]
    q_lower = query.lower()
    return any(k in q_lower for k in keywords)


def handle_table_query(query: str) -> str:
    """
    Handle a table-related query using the TAPAS model.
    Decide which table to use based on the query content.
    """
    q_lower = query.lower()
    if "country" in q_lower:
        if country_analysis_df is not None and not country_analysis_df.empty:
            return tqa.answer_question(country_analysis_df, query)
        else:
            return "I don't have country analysis data available."
    elif "payment" in q_lower:
        if payment_method_df is not None and not payment_method_df.empty:
            return tqa.answer_question(payment_method_df, query)
        else:
            return "I don't have payment method analysis data available."
    else:
        # If no specific table identified, fallback:
        return "I’m not sure which table to query. Can you specify if you mean countries or payment methods?"


def handle_query(query: str) -> str:
    """
    Handle user queries.
    - If it's a metric query, try to answer from insights.
    - If it's a table query, use TAPAS.
    - Otherwise, fallback.
    """
    if is_metric_query(query):
        return answer_metric_query(query, insights)
    elif is_table_question(query):
        return handle_table_query(query)
    else:
        return "I'm not sure. Can you ask about revenue, orders, customers, or refer to the country/payment tables?"


if __name__ == "__main__":
    print("Enhanced Table & Metric QA Chatbot ready. Type your question. Type 'exit' to quit.")
    while True:
        user_query = input("Q: ")
        if user_query.lower() in ["exit", "quit"]:
            break
        response = handle_query(user_query)
        print("A:", response)
