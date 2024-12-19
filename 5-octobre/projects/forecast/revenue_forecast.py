import pandas as pd
from prophet import Prophet
import logging
import sys

sys.path.append(
    "/Users/benpfeffer/Library/Mobile Documents/com~apple~CloudDocs/projects-portfolio-main/5-octobre"
)
from src.data_preprocessing import preprocess_data
from src.metrics import compute_revenue_over_time

logger = logging.getLogger(__name__)


def forecast_monthly_revenue(order_df, periods=6, date_col="Date", total_col="Total"):
    """
    Forecast monthly revenue using Prophet.

    Parameters:
    - order_df: Pandas DataFrame of order data with at least 'Date' and 'Total' columns.
    - periods: Integer, number of months to forecast into the future.
    - date_col: Column name of the date column in order_df.
    - total_col: Column name of the total revenue column in order_df.

    Returns:
    A tuple (forecast_df, model):
    - forecast_df: DataFrame containing the forecast, with columns 'ds', 'yhat', 'yhat_lower', 'yhat_upper'.
    - model: The trained Prophet model, for further inspection or advanced usage.
    """
    logger.info("Starting monthly revenue forecasting...")

    # Compute monthly revenue
    monthly_revenue = compute_revenue_over_time(
        order_df, freq="M", date_col=date_col, total_col=total_col
    )
    if monthly_revenue.empty:
        logger.error("Monthly revenue series is empty, cannot forecast.")
        return pd.DataFrame(), None

    # Prepare data for Prophet
    df = monthly_revenue.reset_index()
    df.columns = ["ds", "y"]  # Prophet requires ds and y
    # Ensure ds is datetime; compute_revenue_over_time should already return a DatetimeIndex, but double check
    if not pd.api.types.is_datetime64_ns_dtype(df["ds"]):
        df["ds"] = pd.to_datetime(df["ds"])

    # Initialize Prophet model
    model = Prophet(
        daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=True
    )

    logger.info("Fitting the Prophet model on historical monthly revenue...")
    model.fit(df)

    # Create future dataframe
    future = model.make_future_dataframe(periods=periods, freq="M")

    logger.info(f"Forecasting the next {periods} months of revenue...")
    forecast = model.predict(future)

    logger.info("Monthly revenue forecasting complete.")
    return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]], model


# Example usage within the run_analysis or as a separate step:
if __name__ == "__main__":
    # Assume cart_df, order_df are loaded and run_analysis or another loading mechanism is available
    cart_df, order_df = preprocess_data()
    if order_df is not None:
        forecast_df, prophet_model = forecast_monthly_revenue(order_df, periods=6)
        if not forecast_df.empty:
            print("Forecasted Monthly Revenue:")
            print(forecast_df.tail(6))
        else:
            print("Forecast not available.")
