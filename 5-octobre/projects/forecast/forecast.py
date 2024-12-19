import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
import logging
import sys

# Adjust path as needed
sys.path.append("/Users/benpfeffer/Library/Mobile Documents/com~apple~CloudDocs/projects-portfolio-main/5-octobre")

from src.data_preprocessing import preprocess_data
from src.metrics import compute_revenue_over_time

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def prepare_time_series_features(series, lags=12, forecast_horizon=1, external_df=None):
    """
    Prepare feature matrix and target vector for time series forecasting.

    Reasoning:
    - We create lag features to capture past values of the series which often have predictive power.
    - Include month/quarter features to model seasonality.
    - Rolling means can help the model understand smoother trends.
    - External regressors can capture exogenous influences (e.g., marketing spend).

    Parameters:
    -----------
    series : pd.Series
        Time series data indexed by date at a monthly frequency (e.g., monthly revenue or orders).
    lags : int
        Number of lagged values to include as features.
    forecast_horizon : int
        How many steps ahead to forecast (1 means next month).
    external_df : pd.DataFrame or None
        Optional external regressors, indexed by the same date frequency as `series`.

    Returns:
    --------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target values (shifted by forecast_horizon).
    dates : pd.DatetimeIndex
        The index (dates) for the rows included in X, y.
    """
    df = pd.DataFrame(series, copy=True)
    df.columns = ["y"]

    # Create lag features - previous N months' values to capture patterns and dependencies
    # e.g. lag_1 is previous month, lag_2 is 2 months ago, etc.
    for i in range(1, lags + 1):
        df[f"lag_{i}"] = df["y"].shift(i)

    # Add seasonal/time features to model cyclical patterns
    # month: captures monthly seasonality (1-12)
    df["month"] = df.index.month
    # quarter: captures quarterly seasonality (1-4)
    df["quarter"] = df.index.quarter

    # Rolling stats for smoothing trends and reducing noise
    # rolling_3_mean: 3-month moving average, shifted by 1 to avoid data leakage
    df["rolling_3_mean"] = df["y"].shift(1).rolling(3).mean()
    # rolling_6_mean: 6-month moving average for longer-term trends
    df["rolling_6_mean"] = df["y"].shift(1).rolling(6).mean()

    # Join external regressors if provided
    if external_df is not None:
        df = df.join(external_df, how="left")

    # Drop rows with NaNs (due to lags and rolling)
    df = df.dropna()

    # Shift target by forecast horizon
    df["y_target"] = df["y"].shift(-forecast_horizon)
    df = df.dropna()

    y = df["y_target"]
    # Select features: all lag_* and rolling_* plus month/quarter and external regressors if any
    base_features = [c for c in df.columns if c.startswith("lag_") or c.startswith("rolling_")]
    base_features += ["month", "quarter"]

    if external_df is not None:
        extra_cols = [c for c in external_df.columns if c in df.columns]
        base_features.extend(extra_cols)

    X = df[base_features]
    dates = df.index

    return X, y, dates


def simple_baseline_forecast(series, forecast_horizon=1):
    """
    A simple baseline forecast (e.g., naive forecast) for comparison.

    Reasoning:
    - Always predicting the last known value or some simple rule can provide a benchmark.
    - If our complex model cannot beat a naive forecast, we need to revisit the approach.

    Parameters:
    -----------
    series : pd.Series
        Historical time series data.
    forecast_horizon : int
        Steps ahead (1 means next period).

    Returns:
    --------
    float
        The naive forecast value (here we just use the last known actual).
    """
    return series.iloc[-1]


def residual_bootstrap_intervals(residuals, num_samples=1000, alpha=0.05):
    """
    Generate prediction intervals from residual bootstrapping.

    Reasoning:
    - We assume model residuals are representative of forecast error distribution.
    - By sampling from residuals, we can create an empirical distribution of forecasts and derive intervals.
    - This is a simplistic approach and not always statistically rigorous, but can give a rough idea.

    Parameters:
    -----------
    residuals : np.array
        Residuals from the forecast model on a validation set.
    num_samples : int
        Number of bootstrap samples.
    alpha : float
        Significance level (0.05 means 95% CI).

    Returns:
    --------
    tuple (lower_bound, upper_bound)
        Approximate prediction interval bounds based on the bootstrap distribution.
    """
    # Draw samples from residuals and compute quantiles
    res_samples = np.random.choice(residuals, size=num_samples, replace=True)
    lower = np.percentile(res_samples, alpha / 2 * 100)
    upper = np.percentile(res_samples, (1 - alpha / 2) * 100)
    return lower, upper


def ml_forecast(
    series,
    lags=12,
    forecast_horizon=1,
    future_periods=60,
    external_df=None,
    metric_name="revenue",
):
    """
    Forecast a given time series metric using ML (XGB) plus enhancements.

    Reasoning Steps:
    - We use GridSearchCV for hyperparam tuning on a TimeSeriesSplit for robust validation.
    - Evaluate model on a hold-out test set.
    - Combine ML forecast with a simple baseline for an ensemble: final = 0.8*ml_forecast + 0.2*baseline.
    - Approximate intervals using residual bootstrap.

    Parameters:
    -----------
    series : pd.Series
        Monthly aggregated data of the metric we want to forecast.
    lags : int
        Number of lag features.
    forecast_horizon : int
        Steps ahead to forecast.
    future_periods : int
        How many future periods to forecast.
    external_df : pd.DataFrame or None
        Additional exogenous features.
    metric_name : str
        Name of the metric being forecasted (for logging and reference).

    Returns:
    --------
    pd.DataFrame
        DataFrame with columns: ds (dates), yhat (forecast), yhat_lower, yhat_upper for intervals.
    """
    # Log start of forecasting process
    logger.info(f"Starting ML forecast for {metric_name}...")

    # Step 1: Feature preparation
    # Create lagged features and prepare data for modeling
    X, y, dates = prepare_time_series_features(series, lags=lags, forecast_horizon=forecast_horizon, external_df=external_df)

    # Check if we have enough data points for meaningful forecasting
    if len(X) < 24:  # Minimum 2 years of monthly data
        logger.warning(f"Not enough data to forecast {metric_name}. Returning empty.")
        return pd.DataFrame()

    # Step 2: Model Configuration and Training
    # Set up time series cross-validation
    tscv = TimeSeriesSplit(n_splits=3)

    # Define hyperparameter search space
    param_grid = {
        "n_estimators": [100, 200],  # Number of trees
        "max_depth": [3, 5],  # Tree depth
        "learning_rate": [0.05, 0.1],  # Learning rate for boosting
        "subsample": [0.8, 1.0],  # Fraction of samples for trees
    }

    # Initialize and train model with grid search
    model = XGBRegressor(random_state=42)
    gsearch = GridSearchCV(model, param_grid, cv=tscv, scoring="neg_mean_absolute_error", n_jobs=-1)
    gsearch.fit(X, y)
    best_model = gsearch.best_estimator_
    logger.info(f"Best params for {metric_name}: {gsearch.best_params_}")

    # Step 3: Hold-out Testing
    # Split data into train/test sets
    test_size = 6  # 6 months test set
    train_size = len(X) - test_size
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

    # Evaluate on hold-out test set
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    logger.info(f"Hold-out Test MAE for {metric_name}: {mae:.2f}")

    # Calculate residuals for prediction intervals
    residuals = y_test.values - y_pred

    # Step 4: Generate Future Forecasts
    current_series = series.copy()
    current_date = series.index[-1]

    forecast_values = []
    forecast_dates = []

    # Iteratively forecast future periods
    for step in range(future_periods):
        # Generate features for next prediction
        X_future, _, _ = prepare_time_series_features(
            current_series,
            lags=lags,
            forecast_horizon=forecast_horizon,
            external_df=external_df,
        )
        X_future_point = X_future.iloc[[-1]]  # Get latest point for prediction

        # Get ML model prediction
        ml_forecast_val = best_model.predict(X_future_point)[0]

        # Get baseline prediction for ensemble
        baseline_val = simple_baseline_forecast(current_series, forecast_horizon=forecast_horizon)

        # Create ensemble forecast (weighted average)
        final_pred = 0.8 * ml_forecast_val + 0.2 * baseline_val

        # Calculate prediction intervals using residual bootstrap
        lower_res, upper_res = residual_bootstrap_intervals(residuals, alpha=0.05)
        yhat_lower = final_pred + lower_res
        yhat_upper = final_pred + upper_res

        # Update series with new prediction for next iteration
        next_date = current_date + pd.offsets.MonthEnd(forecast_horizon)
        current_series = pd.concat([current_series, pd.Series([final_pred], index=[next_date])])
        current_date = next_date

        # Store results
        forecast_values.append(final_pred)
        forecast_dates.append(next_date)

    # Step 5: Format Results
    forecast_df = pd.DataFrame(
        {
            "ds": forecast_dates,
            "yhat": forecast_values,
            "yhat_lower": [val + lower_res for val in forecast_values],
            "yhat_upper": [val + upper_res for val in forecast_values],
        }
    )

    return forecast_df


def plot_forecast(forecast_df, historical_series=None, title="Forecast"):
    # Plot the forecast
    plt.figure(figsize=(12, 6))
    plt.plot(forecast_df["ds"], forecast_df["yhat"], label="Forecast", color="#2C5F9E", linewidth=2, linestyle="--")
    plt.fill_between(forecast_df["ds"], forecast_df["yhat_lower"], forecast_df["yhat_upper"], alpha=0.2)
    plt.plot(historical_series.index, historical_series.values, label="Historical", color="gray")
    plt.title(title)
    plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1))
    plt.show()


if __name__ == "__main__":
    # Example usage:
    cart_df, order_df = preprocess_data()
    if order_df is not None:
        # Forecast monthly revenue
        monthly_revenue = compute_revenue_over_time(order_df, freq="M")
        revenue_forecast = ml_forecast(monthly_revenue, metric_name="revenue")
        print("Revenue Forecast:")
        print(revenue_forecast)

        # Forecast monthly orders: first compute monthly order count
        order_count_series = order_df.set_index("Date").resample("M")["Référence"].count()
        orders_forecast = ml_forecast(order_count_series, metric_name="orders")
        print("Orders Forecast:")
        print(orders_forecast)

        plot_forecast(revenue_forecast, historical_series=monthly_revenue, title="Revenue Forecast")
        plot_forecast(orders_forecast, historical_series=order_count_series, title="Orders Forecast")

        # Example: if we had external data, we could load it:
        # external_df = pd.read_csv("external_regressors.csv", parse_dates=["Date"], index_col="Date")
        # Make sure external_df frequency matches monthly and aligns with series index
        # revenue_forecast_with_external = ml_forecast(monthly_revenue, external_df=external_df, metric_name="revenue_with_external")
        # print(revenue_forecast_with_external)
