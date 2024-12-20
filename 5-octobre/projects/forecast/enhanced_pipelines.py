import torch
import os
import sys
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional, Dict, Any
import atexit
import multiprocessing as mp

# Add your project path
sys.path.append("/Users/benpfeffer/Library/Mobile Documents/com~apple~CloudDocs/projects-portfolio-main/5-octobre")

from src.config import BASE_DIR, load_logger
from src.data_preprocessing import preprocess_data
from src.metrics import (
    basic_kpis,
    compute_revenue_over_time,
    analyze_customer_count,
    customer_segmentation_by_value,
    rfm_analysis,
    cohort_analysis,
    # Add other existing metrics as needed
)

logger = load_logger()


def cleanup_resources():
    """Cleanup function to handle process termination"""
    try:
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"Error during cleanup: {e}")


# Register the cleanup function
atexit.register(cleanup_resources)

###############################################################################
# LOAD DATA
###############################################################################

# Load main cleaned data
cart_df, order_df = preprocess_data()

# Load products data
products_file = os.path.join(BASE_DIR, "5octobre_products.csv")
products_df = pd.read_csv(products_file)

# Load line-item order data (required for product-level metrics).
# If you do not have this, you need to create or obtain it.
# This file should link each order reference to products, quantity, and line-level price.
order_items_file = os.path.join(BASE_DIR, "data", "cleaned", "order_items.csv")
if os.path.exists(order_items_file):
    order_items_df = pd.read_csv(order_items_file)
    order_items_df["Date"] = pd.to_datetime(order_items_df["Date"], errors="coerce")
else:
    logger.warning("No order_items_df found. Product-level metrics limited.")
    order_items_df = pd.DataFrame()


###############################################################################
# ADDITIONAL METRICS IMPLEMENTATION
###############################################################################


def top_selling_products(order_items_df, product_col="product_name", qty_col="quantity", price_col="line_price", top_n=10):
    """
    Identify top selling products by volume and by revenue.
    Returns two DataFrames:
    - top_volume: products sorted by units sold
    - top_revenue: products sorted by total revenue generated
    """
    if order_items_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    product_sales = (
        order_items_df.groupby(product_col)
        .agg({qty_col: "sum", price_col: "sum"})
        .rename(columns={qty_col: "total_units_sold", price_col: "total_revenue"})
        .sort_values("total_revenue", ascending=False)
    )

    top_volume = product_sales.sort_values("total_units_sold", ascending=False).head(top_n)
    top_revenue = product_sales.sort_values("total_revenue", ascending=False).head(top_n)
    return top_volume, top_revenue


def product_revenue_share(order_items_df, price_col="line_price", product_col="product_name"):
    """
    Determine the percentage of total revenue contributed by each product.
    Useful for identifying revenue concentration at the product level.
    """
    if order_items_df.empty:
        return pd.DataFrame()

    total_revenue = order_items_df[price_col].sum()
    product_revenue = order_items_df.groupby(product_col)[price_col].sum().sort_values(ascending=False)
    product_revenue_share = (product_revenue / total_revenue) * 100
    result = pd.DataFrame({"revenue": product_revenue, "revenue_share_pct": product_revenue_share})
    return result


def product_price_variation(order_items_df, products_df, product_col="product_name", price_col="line_price", qty_col="quantity"):
    """
    Compare actual selling prices to base price and detect discounts.
    Returns a DataFrame with product-level average discount rates.
    """
    if order_items_df.empty or products_df.empty:
        return pd.DataFrame()

    merged = order_items_df.merge(products_df, left_on=product_col, right_on="name", how="left")
    merged["unit_sold_price"] = merged[price_col] / merged[qty_col]
    merged = merged[merged["price"] > 0]  # filter out products with invalid base price
    merged["discount_rate"] = (1 - (merged["unit_sold_price"] / merged["price"])) * 100
    discount_stats = merged.groupby(product_col)["discount_rate"].mean().sort_values(ascending=False)
    return discount_stats.to_frame("avg_discount_rate_pct")


def product_repeat_purchase_rate(order_items_df, client_col="Client", product_col="product_name"):
    """
    Identify products that customers frequently rebuy.
    This requires order_items_df to have a customer identifier as well.
    If not present, this metric cannot be computed.
    """
    if order_items_df.empty or client_col not in order_items_df.columns:
        return pd.DataFrame()
    # Count how many distinct customers bought each product multiple times
    product_cust_counts = order_items_df.groupby([client_col, product_col]).size().reset_index(name="purchase_count")
    # Consider a product 'repeat purchased' if a customer bought it more than once
    repeat_purchases = product_cust_counts[product_cust_counts["purchase_count"] > 1]
    product_repeat_counts = repeat_purchases.groupby(product_col)[client_col].nunique()
    product_total_counts = product_cust_counts.groupby(product_col)[client_col].nunique()
    repeat_rate = (product_repeat_counts / product_total_counts) * 100
    return repeat_rate.to_frame("repeat_purchase_rate_pct")


###############################################################################
# CUSTOMER & CHURN METRICS
###############################################################################


def exact_churn_rate(order_df, date_col="Date", client_col="Client"):
    """
    Compute monthly exact churn by tracking customers month-over-month.
    """
    order_df[date_col] = pd.to_datetime(order_df[date_col])
    order_df["Month"] = order_df[date_col].dt.to_period("M").dt.to_timestamp()
    monthly_customers = order_df.groupby("Month")[client_col].apply(set)

    months = monthly_customers.index.sort_values()
    churn_data = []
    for i in range(1, len(months)):
        prev_customers = monthly_customers.loc[months[i - 1]]
        current_customers = monthly_customers.loc[months[i]]
        if len(prev_customers) > 0:
            lost = len(prev_customers - current_customers)
            churn_rate = lost / len(prev_customers) * 100
        else:
            churn_rate = np.nan
        churn_data.append({"Month": months[i], "churn_rate": churn_rate})
    return pd.DataFrame(churn_data)


def net_revenue_retention(order_df, date_col="Date", client_col="Client", total_col="Total"):
    """
    Compute Net Revenue Retention (NRR) month-over-month for cohorts.
    NRR considers expansions, contractions, and churn.

    Steps:
    - Identify recurring revenue from the same customer set month-over-month.
    - NRR = (Revenue from retained, expanded customers / Revenue from initial cohort) * 100
    """
    order_df[date_col] = pd.to_datetime(order_df[date_col])
    order_df["Month"] = order_df[date_col].dt.to_period("M").dt.to_timestamp()

    # Map each customer to their first month of purchase
    first_purchase = order_df.groupby(client_col)["Month"].min().rename("FirstMonth")
    order_df = order_df.merge(first_purchase, on=client_col, how="left")

    # Cohort: customers grouped by their FirstMonth
    # We'll compute revenue each month for that cohort and then NRR comparing month i to cohort start month
    cohort_data = order_df.groupby(["FirstMonth", "Month"])[total_col].sum().reset_index()

    nrr_list = []
    # For each cohort start month
    for start_month in cohort_data["FirstMonth"].unique():
        cohort_revenue = cohort_data[cohort_data["FirstMonth"] == start_month]
        initial_revenue = cohort_revenue[cohort_revenue["Month"] == start_month][total_col].sum()
        if initial_revenue == 0:
            continue
        for m in cohort_revenue["Month"].unique():
            if m >= start_month:
                current_revenue = cohort_revenue[cohort_revenue["Month"] == m][total_col].sum()
                nrr = (current_revenue / initial_revenue) * 100
                nrr_list.append({"cohort_start": start_month, "Month": m, "NRR_pct": nrr})

    return pd.DataFrame(nrr_list)


###############################################################################
# FORECASTING ENHANCEMENTS
###############################################################################

from prophet import Prophet
from pmdarima import auto_arima
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit


def prophet_forecast(series: pd.Series, periods=6):
    """
    Forecast a time series with Prophet.
    """
    df = series.reset_index()
    df.columns = ["ds", "y"]
    model = Prophet(yearly_seasonality=True)
    model.fit(df)
    future = model.make_future_dataframe(periods=periods, freq="ME")
    forecast = model.predict(future)
    return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]


def arima_forecast(series: pd.Series, periods=6):
    """
    Forecast a time series using auto-ARIMA.
    """
    model = auto_arima(series, seasonal=False, trace=False)
    forecast = model.predict(n_periods=periods, return_conf_int=True)
    fc_values = forecast[0]
    fc_conf = forecast[1]
    idx = pd.date_range(series.index[-1] + pd.offsets.MonthEnd(), periods=periods, freq="M")
    result = pd.DataFrame({"ds": idx, "yhat": fc_values, "yhat_lower": fc_conf[:, 0], "yhat_upper": fc_conf[:, 1]})
    return result


def xgb_forecast(series: pd.Series, periods=6, lags=12):
    """
    Forecast with XGBoost by creating lag features.
    This is a simple example. In practice, add external regressors, seasonal indicators, etc.
    """
    df = series.to_frame("y").copy()
    # Create lag features
    for i in range(1, lags + 1):
        df[f"lag_{i}"] = df["y"].shift(i)
    df = df.dropna()

    X = df.drop("y", axis=1)
    y = df["y"]
    tscv = TimeSeriesSplit(n_splits=3)
    model = XGBRegressor(n_estimators=100, max_depth=3)
    # No param tuning shown here, but can be done
    model.fit(X, y)

    # Forecasting next periods:
    last_vals = df.iloc[-lags:].copy()
    future_preds = []
    current_features = last_vals.drop("y", axis=1).tail(1)

    for i in range(periods):
        # Predict next step
        pred = model.predict(current_features)[0]
        future_preds.append(pred)
        # Shift lags:
        new_row = current_features.iloc[0].copy()
        # Move all lag_i to lag_(i+1)
        for j in range(lags, 1, -1):
            new_row[f"lag_{j}"] = new_row[f"lag_{j-1}"]
        new_row["lag_1"] = pred
        current_features = new_row.to_frame().T

    idx = pd.date_range(series.index[-1] + pd.offsets.MonthEnd(), periods=periods, freq="M")
    result = pd.DataFrame({"ds": idx, "yhat": future_preds})
    # No intervals by default in this simplistic example
    result["yhat_lower"] = result["yhat"] - result["yhat"] * 0.1
    result["yhat_upper"] = result["yhat"] + result["yhat"] * 0.1
    return result


###############################################################################
# CHATBOT ENHANCEMENTS
###############################################################################
# This snippet shows how you might set up a retrieval-based QA chatbot with a pre-trained LLM.
# For full details, see previous code. Adapt paths and model as necessary.
#
# Enhancements:
# - More robust entity extraction: use spaCy (if available)
# - Advanced date normalization using dateparser
# - More powerful LLM with better reasoning: using a model from Hugging Face Hub

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline


def setup_chatbot(analysis_text_path: str):
    """
    Setup a chatbot that can answer questions about the metrics and forecasts.
    """
    if not os.path.exists(analysis_text_path):
        logger.warning("Analysis summary file not found. Chatbot will have limited capabilities.")
        return None

    try:
        # Only set start method if it hasn't been set
        if not mp.get_start_method(allow_none=True):
            mp.set_start_method("spawn", force=True)

        with open(analysis_text_path, "r") as f:
            text = f.read()

        chunks = [text[i : i + 500] for i in range(0, len(text), 500)]

        # Updated embeddings initialization with warning suppression
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})

        # Create vector store
        vectorstore = Chroma.from_texts(chunks, embeddings)

        # Load a smaller model to avoid memory issues
        model_id = "google/flan-t5-base"  # Changed from flan-t5-xxl
        hf_tokenizer = AutoTokenizer.from_pretrained(model_id)
        hf_model = AutoModelForSeq2SeqLM.from_pretrained(model_id, device_map="auto", max_length=512, torch_dtype="auto")

        # Configure pipeline with appropriate parameters
        llm_pipeline = pipeline(
            "text2text-generation",
            model=hf_model,
            tokenizer=hf_tokenizer,
            max_length=512,
            max_new_tokens=200,
            temperature=0.7,
            device="cpu",  # Explicitly set device
        )

        llm = HuggingFacePipeline(pipeline=llm_pipeline)
        retriever = vectorstore.as_retriever(
            search_kwargs={"k": 3}  # Limit number of chunks retrieved
        )
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=False)

        return qa_chain
    except Exception as e:
        logger.error(f"Error setting up chatbot: {e}")
        return None
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


###############################################################################
# MAIN EXECUTION
###############################################################################

if __name__ == "__main__":
    try:
        # Compute some metrics
        logger.info("Computing extended metrics...")

        # Basic KPIs
        kpis = basic_kpis(order_df)
        logger.info(f"KPIs: {kpis}")

        # Monthly revenue
        monthly_revenue = compute_revenue_over_time(order_df, freq="ME")

        # Product-level metrics (requires order_items_df)
        top_vol, top_rev = top_selling_products(order_items_df)
        product_shares = product_revenue_share(order_items_df)
        discount_stats = product_price_variation(order_items_df, products_df)

        # Churn and NRR
        churn_df = exact_churn_rate(order_df)
        nrr_df = net_revenue_retention(order_df)

        # Customer Segmentation
        seg_df = customer_segmentation_by_value(order_df)

        # RFM
        rfm_df = rfm_analysis(order_df)

        # Cohort retention
        cohort_ret = cohort_analysis(order_df)

        # Forecast overall revenue using Prophet
        if not monthly_revenue.empty:
            rev_forecast = prophet_forecast(monthly_revenue, periods=6)
            logger.info("Revenue forecast generated.")

        # Forecast top product revenue with ARIMA or XGBoost if we have monthly product data
        # Suppose we have monthly product revenue from order_items:
        if not order_items_df.empty:
            monthly_product_revenue = (
                order_items_df.assign(Month=lambda df: df["Date"].dt.to_period("M").dt.to_timestamp()).groupby(["Month", "product_name"])["line_price"].sum().unstack(fill_value=0)
            )
            # Choose a top product
            if not monthly_product_revenue.empty:
                top_product = monthly_product_revenue.sum().sort_values(ascending=False).index[0]
                prod_series = monthly_product_revenue[top_product]
                prod_fc_arima = arima_forecast(prod_series, periods=6)
                prod_fc_xgb = xgb_forecast(prod_series, periods=6)
                logger.info(f"ARIMA and XGB forecasts generated for top product: {top_product}")

        # Setup chatbot (assuming analysis_summary.md is available)
        analysis_summary_path = os.path.join(BASE_DIR, "data", "analysis", "analysis_summary.md")
        chatbot = setup_chatbot(analysis_summary_path)
        if chatbot:
            try:
                query = "What was the monthly revenue trend last quarter?"
                answer = chatbot.invoke({"query": query})
                logger.info(f"Chatbot answer: {answer}")
            except Exception as e:
                logger.error(f"Error getting chatbot response: {e}")
            finally:
                # Clean up resources
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        else:
            logger.warning("Chatbot not configured.")

        # Save results if needed
        output_dir = os.path.join(BASE_DIR, "data", "analysis")
        os.makedirs(output_dir, exist_ok=True)
        if not top_vol.empty:
            top_vol.to_csv(os.path.join(output_dir, "top_products_by_volume.csv"))
        if not top_rev.empty:
            top_rev.to_csv(os.path.join(output_dir, "top_products_by_revenue.csv"))
        if not product_shares.empty:
            product_shares.to_csv(os.path.join(output_dir, "product_revenue_share.csv"))
        if isinstance(discount_stats, pd.DataFrame) and not discount_stats.empty:
            discount_stats.to_csv(os.path.join(output_dir, "product_discount_stats.csv"))
        if rev_forecast is not None:
            rev_forecast.to_csv(os.path.join(output_dir, "revenue_forecast.csv"), index=False)
        if churn_df is not None and not churn_df.empty:
            churn_df.to_csv(os.path.join(output_dir, "exact_churn_rate.csv"), index=False)
        if nrr_df is not None and not nrr_df.empty:
            nrr_df.to_csv(os.path.join(output_dir, "net_revenue_retention.csv"), index=False)
        if seg_df is not None and not seg_df.empty:
            seg_df.to_csv(os.path.join(output_dir, "customer_segmentation.csv"), index=False)
        if rfm_df is not None and not rfm_df.empty:
            rfm_df.to_csv(os.path.join(output_dir, "rfm_analysis.csv"))
        if cohort_ret is not None and not cohort_ret.empty:
            cohort_ret.to_csv(os.path.join(output_dir, "cohort_retention.csv"))

        # Product forecasting results
        if "prod_fc_arima" in locals():
            prod_fc_arima.to_csv(os.path.join(output_dir, f"{top_product}_arima_forecast.csv"), index=False)
        if "prod_fc_xgb" in locals():
            prod_fc_xgb.to_csv(os.path.join(output_dir, f"{top_product}_xgb_forecast.csv"), index=False)

        logger.info("All analyses and forecasts completed.")
    except Exception as e:
        logger.error(f"Main execution error: {e}")
    finally:
        cleanup_resources()
