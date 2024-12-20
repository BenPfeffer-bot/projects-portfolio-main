import os
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import logging
import sys

sys.path.append("/Users/benpfeffer/Library/Mobile Documents/com~apple~CloudDocs/projects-portfolio-main/5-octobre")
from src.data_preprocessing import preprocess_data
from src.metrics import compute_revenue_over_time
from src.config import BASE_DIR, load_logger

# Additional imports for advanced forecasting
from pmdarima import auto_arima
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

logger = load_logger()


def load_product_data(products_file_path):
    """
    Load product data from CSV containing product names, base prices, and possibly categories.
    """
    logger.info(f"Loading product data from {products_file_path}")
    try:
        products_df = pd.read_csv(products_file_path)
        # Ensure columns: 'name', 'price'
        # If categories or other attributes exist, they can be used for segmentation.
        logger.debug(f"Products DataFrame shape: {products_df.shape}")
        return products_df
    except FileNotFoundError as e:
        logger.error(f"Product file not found: {e}")
        return pd.DataFrame()


def load_order_items_data(order_items_path):
    """
    Load order items data linking orders to products and their sold price and quantity.
    This file or table should contain columns like:
    - 'order_id' or 'Référence'
    - 'product_name' or product_id
    - 'quantity'
    - 'line_price' (the total price for that line)

    If not currently available, this function returns empty or a placeholder.
    """
    # This is a placeholder. Adjust path and columns to your actual data structure.
    if not os.path.exists(order_items_path):
        logger.warning("Order items data not found. Product-level analysis won't be possible.")
        return pd.DataFrame()

    order_items_df = pd.read_csv(order_items_path)
    # Ensure proper datetime conversion if there's a 'Date' or 'order_date' column
    if "Date" in order_items_df.columns:
        order_items_df["Date"] = pd.to_datetime(order_items_df["Date"], errors="coerce")
    return order_items_df


def compute_monthly_product_sales(order_items_df, date_col="Date", product_col="product_name", qty_col="quantity", price_col="line_price"):
    """
    Compute monthly units sold and monthly revenue for each product.
    Returns two DataFrames:
    - monthly_units: indexed by Month, columns=product, values=units sold
    - monthly_revenue: indexed by Month, columns=product, values=revenue
    """
    if order_items_df.empty:
        logger.warning("Order items data is empty. Cannot compute monthly product sales.")
        return pd.DataFrame(), pd.DataFrame()

    # Resample by month
    order_items_df[date_col] = pd.to_datetime(order_items_df[date_col])
    order_items_df["Month"] = order_items_df[date_col].dt.to_period("M").dt.to_timestamp()

    monthly_units = order_items_df.groupby(["Month", product_col])[qty_col].sum().unstack(fill_value=0)
    monthly_revenue = order_items_df.groupby(["Month", product_col])[price_col].sum().unstack(fill_value=0)

    return monthly_units, monthly_revenue


def compute_product_price_metrics(order_items_df, products_df, product_col="product_name", price_col="line_price", qty_col="quantity"):
    """
    Compute price-related metrics:
    - Average selling price per product over time
    - Discount rates = (Base Price - Actual Selling Price) / Base Price
    Assuming 'price' in products_df is the base price.

    Returns:
    - monthly_avg_price: Average sold price per unit (line_price/quantity) by month and product
    - monthly_discount_rate: Mean discount rate by month and product (if base price available)
    """
    if order_items_df.empty or products_df.empty:
        logger.warning("Either order_items_df or products_df is empty. Cannot compute product price metrics.")
        return pd.DataFrame(), pd.DataFrame()

    # Merge base price info into order_items_df
    # Ensure that 'name' in products_df matches 'product_name' in order_items_df
    merged = order_items_df.merge(products_df, left_on=product_col, right_on="name", how="left", suffixes=("", "_base"))
    if "price" not in merged.columns:
        logger.warning("Base price column not found in products data. Cannot compute discount rates.")
        return pd.DataFrame(), pd.DataFrame()

    # Compute actual sold price per unit
    merged["unit_sold_price"] = merged[price_col] / merged[qty_col]

    # Discount = 1 - (unit_sold_price / base_price)
    # If base_price = 0 (unlikely), handle gracefully
    merged["discount_rate"] = np.where(merged["price"] > 0, 1 - (merged["unit_sold_price"] / merged["price"]), np.nan)

    # Group by month and product
    merged["Month"] = merged["Date"].dt.to_period("M").dt.to_timestamp()
    monthly_avg_price = merged.groupby(["Month", product_col])["unit_sold_price"].mean().unstack(fill_value=np.nan)
    monthly_discount_rate = merged.groupby(["Month", product_col])["discount_rate"].mean().unstack(fill_value=np.nan)

    return monthly_avg_price, monthly_discount_rate


def forecast_product_metric(metric_series, product_name, periods=6):
    """
    Given a monthly series for a particular product (e.g. monthly revenue),
    forecast the next 6 periods using Prophet as an example.

    metric_series: pd.Series indexed by datetime (Month), values = metric (e.g. revenue)
    product_name: str, the product name for logging
    """
    df = metric_series.dropna().reset_index()
    df.columns = ["ds", "y"]
    df["ds"] = pd.to_datetime(df["ds"])
    if len(df) < 12:
        logger.info(f"Not enough data to forecast {product_name}. Returning empty.")
        return pd.DataFrame()

    # Simple Prophet model (could tune hyperparams as above)
    model = Prophet(yearly_seasonality=True, daily_seasonality=False, weekly_seasonality=False)
    model.fit(df)
    future = model.make_future_dataframe(periods=periods, freq="M")
    forecast = model.predict(future)
    forecast["product"] = product_name
    return forecast[["ds", "yhat", "yhat_lower", "yhat_upper", "product"]]


def compute_monthly_orders(order_df, date_col="Date"):
    order_df[date_col] = pd.to_datetime(order_df[date_col])
    monthly_orders = order_df.set_index(date_col).resample("M")["Référence"].count()
    return monthly_orders


def compute_monthly_unique_customers(order_df, date_col="Date", client_col="Client"):
    order_df[date_col] = pd.to_datetime(order_df[date_col])
    monthly_customers = order_df.set_index(date_col).groupby(pd.Grouper(freq="M"))[client_col].nunique()
    return monthly_customers


def compute_monthly_aov(monthly_revenue, monthly_orders):
    return monthly_revenue / monthly_orders.replace(0, np.nan)


def compute_monthly_new_returning_customers(order_df, date_col="Date", client_col="Client"):
    """
    Compute monthly new and returning customers.
    """
    order_df[date_col] = pd.to_datetime(order_df[date_col])
    order_df = order_df.sort_values(date_col)

    first_purchase = order_df.groupby(client_col)[date_col].min().rename("FirstPurchaseDate")
    order_df = order_df.merge(first_purchase, on=client_col, how="left")

    order_df["OrderMonth"] = order_df[date_col].dt.to_period("M")
    order_df["FirstPurchaseMonth"] = order_df["FirstPurchaseDate"].dt.to_period("M")

    monthly_unique = order_df.groupby("OrderMonth")[client_col].nunique()
    monthly_new = order_df[order_df["OrderMonth"] == order_df["FirstPurchaseMonth"]].groupby("OrderMonth")[client_col].nunique()
    monthly_new = monthly_new.reindex(monthly_unique.index, fill_value=0)
    monthly_returning = monthly_unique - monthly_new

    monthly_new.index = monthly_new.index.to_timestamp()
    monthly_returning.index = monthly_returning.index.to_timestamp()
    return monthly_new, monthly_returning


def compute_sophisticated_churn_rate(order_df, date_col="Date", client_col="Client"):
    """
    More accurate churn calculation:
    - Extract sets of customers each month.
    - For month M -> M+1:
        churn = (# of customers in M who do NOT appear in M+1) / (# of customers in M)
    """
    order_df[date_col] = pd.to_datetime(order_df[date_col])
    order_df = order_df.sort_values(date_col)

    order_df["OrderMonth"] = order_df[date_col].dt.to_period("M")
    month_groups = order_df.groupby("OrderMonth")[client_col].apply(set)

    months = month_groups.index.to_timestamp()
    churn_rates = pd.Series(index=months, dtype=float)

    for i in range(1, len(months)):
        prev_month = months[i - 1]
        this_month = months[i]
        prev_set = month_groups.iloc[i - 1]
        curr_set = month_groups.iloc[i]
        if len(prev_set) > 0:
            churn = len(prev_set - curr_set) / len(prev_set)
        else:
            churn = np.nan
        churn_rates.loc[this_month] = churn

    return churn_rates


def compute_monthly_refunds_cancellations(order_df, date_col="Date", state_col="État", total_col="Total"):
    order_df[date_col] = pd.to_datetime(order_df[date_col])

    refunded_mask = order_df[state_col].str.contains("Remboursé", na=False) | order_df[state_col].str.contains("Remboursement partiel", na=False)
    cancelled_mask = order_df[state_col].str.contains("Annulée", na=False)

    monthly_refunds_count = order_df[refunded_mask].set_index(date_col).resample("M")["Référence"].count()
    monthly_refunds_amount = order_df[refunded_mask].set_index(date_col).resample("M")[total_col].sum()

    monthly_cancellations_count = order_df[cancelled_mask].set_index(date_col).resample("M")["Référence"].count()
    monthly_cancellations_amount = order_df[cancelled_mask].set_index(date_col).resample("M")[total_col].sum()

    return (monthly_refunds_count, monthly_refunds_amount, monthly_cancellations_count, monthly_cancellations_amount)


def compute_monthly_top_payment_method(order_df, date_col="Date", payment_col="Paiement", total_col="Total"):
    order_df[date_col] = pd.to_datetime(order_df[date_col])
    monthly_payment = order_df.groupby([pd.Grouper(key=date_col, freq="M"), payment_col])[total_col].sum().reset_index()
    top_payment = monthly_payment.groupby(date_col).apply(lambda g: g.loc[g[total_col].idxmax(), payment_col])
    return top_payment


def compute_monthly_top_country(order_df, date_col="Date", country_col="Livraison", total_col="Total"):
    order_df[date_col] = pd.to_datetime(order_df[date_col])
    monthly_country = order_df.groupby([pd.Grouper(key=date_col, freq="M"), country_col])[total_col].sum().reset_index()
    top_country = monthly_country.groupby(date_col).apply(lambda g: g.loc[g[total_col].idxmax(), country_col])
    return top_country


def tune_prophet_hyperparams(df):
    """
    Simple hyperparameter tuning for Prophet using a small search over changepoint_prior_scale and seasonality_mode.
    This is a demonstration; real tuning may require more extensive search or cross-validation.
    """
    param_grid = {"changepoint_prior_scale": [0.01, 0.1, 0.5], "seasonality_mode": ["additive", "multiplicative"]}

    best_model = None
    best_mae = float("inf")

    for cps in param_grid["changepoint_prior_scale"]:
        for smode in param_grid["seasonality_mode"]:
            m = Prophet(daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=True, changepoint_prior_scale=cps, seasonality_mode=smode)
            m.fit(df)
            # Use cross_validation if dataset is long enough
            # Otherwise, use in-sample error as a proxy (not ideal)
            try:
                cv_results = cross_validation(m, initial="365 days", period="180 days", horizon="180 days")
                perf = performance_metrics(cv_results)
                mae = perf["mae"].mean()
            except Exception:
                # fallback: in-sample error (rough approximation)
                forecast = m.predict(df)
                mae = np.mean(np.abs(df["y"].values - forecast["yhat"].values))

            if mae < best_mae:
                best_mae = mae
                best_model = m

    return best_model


def prepare_metric_for_prophet(series, periods=6):
    df = series.dropna().reset_index()
    df.columns = ["ds", "y"]
    df["ds"] = pd.to_datetime(df["ds"])
    if len(df) < 24:
        # not enough data, just fit a default model
        model = Prophet(yearly_seasonality=True)
        model.fit(df)
    else:
        model = tune_prophet_hyperparams(df)

    future = model.make_future_dataframe(periods=periods, freq="M")
    forecast = model.predict(future)
    return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]], model


def forecast_with_arima(series, periods=6):
    """
    ARIMA forecasting using pmdarima's auto_arima for a demonstration.
    """
    series = series.dropna()
    if len(series) < 24:
        return None, None

    y = series.values
    # Fit ARIMA model
    arima_model = auto_arima(y, seasonal=False, trace=False, error_action="ignore")
    # Forecast
    fc = arima_model.predict(n_periods=periods)
    # Construct a DataFrame for consistency
    future_index = pd.date_range(start=series.index[-1] + pd.offsets.MonthEnd(), periods=periods, freq="M")
    fc_df = pd.DataFrame({"ds": future_index, "yhat_arima": fc})
    return fc_df, arima_model


def forecast_with_xgboost(series, periods=6):
    """
    Simple XGBoost-based forecasting:
    We create lag features and train on historical data. This is a simplistic demonstration.
    """
    series = series.dropna()
    if len(series) < 24:
        return None, None

    df = pd.DataFrame({"y": series})
    # Create lag features
    lags = 12
    for i in range(1, lags + 1):
        df[f"lag_{i}"] = df["y"].shift(i)
    df = df.dropna()

    X = df.drop("y", axis=1)
    Y = df["y"]
    # Train-test split for validation
    tscv = TimeSeriesSplit(n_splits=3)
    best_mae = float("inf")
    best_model = None

    # Simple parameter, no tuning
    model = XGBRegressor(n_estimators=100, learning_rate=0.1)
    # Evaluate
    for train_idx, test_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_val = Y.iloc[train_idx], Y.iloc[test_idx]
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        mae = mean_absolute_error(y_val, preds)
        if mae < best_mae:
            best_mae = mae
            best_model = model

    # Retrain best_model on full data
    best_model.fit(X, Y)
    # Forecast future by appending future predictions
    # For a deterministic forecast, we use last available data
    last_known = df.iloc[-1]
    future_values = []
    curr_values = last_known.copy()

    for _ in range(periods):
        # Create a single row for prediction
        lags_vals = curr_values[[f"lag_{i}" for i in range(1, lags + 1)]]
        # Shift lags by one month (y predicted becomes lag_1 next iteration)
        pred = best_model.predict([lags_vals])[0]
        future_values.append(pred)
        # Update curr_values: shift lags
        for i in range(lags, 1, -1):
            curr_values[f"lag_{i}"] = curr_values[f"lag_{i-1}"]
        curr_values["lag_1"] = pred

    future_index = pd.date_range(start=series.index[-1] + pd.offsets.MonthEnd(), periods=periods, freq="M")
    fc_xgb = pd.DataFrame({"ds": future_index, "yhat_xgb": future_values})
    return fc_xgb, best_model


def select_best_forecast_method(series, periods=6):
    """
    Try Prophet (with tuning), ARIMA, and XGBoost and pick the best based on a simple backtest (if possible).
    For simplicity, we'll just do a quick backtest with each model and choose best by MAE.
    """
    # Prepare a backtest: last 6 months as test
    if len(series.dropna()) < 30:
        # Not enough data for full evaluation, just return Prophet as default
        fc_prophet, _ = prepare_metric_for_prophet(series, periods=periods)
        return fc_prophet, "Prophet"

    train = series.iloc[:-6]
    test = series.iloc[-6:]

    # Evaluate Prophet
    fc_p, mp = prepare_metric_for_prophet(train, periods=6)
    fc_p = fc_p.set_index("ds")
    prophet_preds = fc_p["yhat"].iloc[-6:]
    prophet_mae = mean_absolute_error(test.values, prophet_preds.values)

    # Evaluate ARIMA
    fc_a, ma = forecast_with_arima(train, periods=6)
    if fc_a is not None:
        fc_a = fc_a.set_index("ds")
        arima_preds = fc_a["yhat_arima"]
        arima_mae = mean_absolute_error(test.values, arima_preds.values)
    else:
        arima_mae = float("inf")

    # Evaluate XGBoost
    fc_x, mx = forecast_with_xgboost(train, periods=6)
    if fc_x is not None:
        fc_x = fc_x.set_index("ds")
        xgb_preds = fc_x["yhat_xgb"]
        xgb_mae = mean_absolute_error(test.values, xgb_preds.values)
    else:
        xgb_mae = float("inf")

    # Choose best
    maes = {"Prophet": prophet_mae, "ARIMA": arima_mae, "XGBoost": xgb_mae}
    best_method = min(maes, key=maes.get)

    # Now refit best model on full series
    if best_method == "Prophet":
        fc_final, _ = prepare_metric_for_prophet(series, periods=periods)
    elif best_method == "ARIMA":
        fc_final, _ = forecast_with_arima(series, periods=periods)
    else:
        fc_final, _ = forecast_with_xgboost(series, periods=periods)

    return fc_final, best_method


def run_multi_metric_analysis_and_forecasting(cart_df, order_df, output_dir=None, periods=6):
    if output_dir is None:
        output_dir = os.path.join(BASE_DIR, "data", "analysis")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger.info("Computing monthly metrics...")

    monthly_revenue = compute_revenue_over_time(order_df, freq="M", date_col="Date", total_col="Total")
    monthly_orders = compute_monthly_orders(order_df)
    monthly_customers = compute_monthly_unique_customers(order_df)
    monthly_aov = compute_monthly_aov(monthly_revenue, monthly_orders)
    monthly_new, monthly_returning = compute_monthly_new_returning_customers(order_df)
    # More accurate churn calculation
    monthly_churn_rate = compute_sophisticated_churn_rate(order_df)
    (monthly_refunds_count, monthly_refunds_amount, monthly_cancellations_count, monthly_cancellations_amount) = compute_monthly_refunds_cancellations(
        order_df, date_col="Date", state_col="État", total_col="Total"
    )
    monthly_top_payment = compute_monthly_top_payment_method(order_df, date_col="Date", payment_col="Paiement", total_col="Total")
    monthly_top_country = compute_monthly_top_country(order_df, date_col="Date", country_col="Livraison", total_col="Total")

    metrics_df = pd.DataFrame(
        {
            "monthly_revenue": monthly_revenue,
            "monthly_orders": monthly_orders,
            "monthly_unique_customers": monthly_customers,
            "monthly_aov": monthly_aov,
            "monthly_new_customers": monthly_new,
            "monthly_returning_customers": monthly_returning,
            "monthly_churn_rate": monthly_churn_rate,
            "monthly_refunds_count": monthly_refunds_count,
            "monthly_refunds_amount": monthly_refunds_amount,
            "monthly_cancellations_count": monthly_cancellations_count,
            "monthly_cancellations_amount": monthly_cancellations_amount,
            "monthly_top_payment_method": monthly_top_payment,
            "monthly_top_country": monthly_top_country,
        }
    )

    metrics_df.index.name = "Date"
    metrics_df.sort_index(inplace=True)
    metrics_df.to_csv(os.path.join(output_dir, "monthly_metrics.csv"))
    logger.info(f"Saved monthly metrics to {os.path.join(output_dir, 'monthly_metrics.csv')}")

    metrics_to_forecast = {
        "revenue": monthly_revenue,
        "orders": monthly_orders,
        "unique_customers": monthly_customers,
        "aov": monthly_aov,
        "returning_customers": monthly_returning,
        "refunds_count": monthly_refunds_count,
        "refunds_amount": monthly_refunds_amount,
        "cancellations_count": monthly_cancellations_count,
        "cancellations_amount": monthly_cancellations_amount,
    }

    forecast_results = None
    for metric_name, series in metrics_to_forecast.items():
        clean_series = series.dropna()
        if len(clean_series) > 18:
            # Attempt multiple forecasting methods and choose best
            fc_df, best_method = select_best_forecast_method(clean_series, periods=periods)
            if fc_df is not None:
                fc_df = fc_df.set_index("ds")
                fc_df.rename(columns={c: f"{metric_name}_{c}" for c in fc_df.columns if c.startswith("yhat")}, inplace=True)
                # Add the chosen method name as a comment or separate column?
                # For simplicity, we skip that. Just store final forecast.
                if forecast_results is None:
                    forecast_results = fc_df
                else:
                    forecast_results = forecast_results.join(fc_df, how="outer")
        else:
            logger.warning(f"Not enough data to properly forecast {metric_name}")

    if forecast_results is not None:
        forecast_results.reset_index(inplace=True)
        forecast_results.rename(columns={"index": "ds"}, inplace=True)
        forecast_results.to_csv(os.path.join(output_dir, "multi_metric_forecast.csv"), index=False)
        logger.info(f"Saved multi-metric forecast to {os.path.join(output_dir, 'multi_metric_forecast.csv')}")
    else:
        logger.warning("No forecasts were produced due to insufficient data.")

    return metrics_df


if __name__ == "__main__":
    logger.info("Starting comprehensive multi-metric analysis and forecasting with advanced methods...")
    cart_df, order_df = preprocess_data()
    if order_df is not None and not order_df.empty:
        metrics_df = run_multi_metric_analysis_and_forecasting(cart_df, order_df, periods=6)
        print("Monthly metrics:")
        print(metrics_df.tail())
    else:
        print("No order data available for analysis and forecasting.")
