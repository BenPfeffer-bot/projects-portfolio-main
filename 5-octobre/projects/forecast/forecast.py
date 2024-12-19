import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import logging
import sys

sys.path.append(
    "/Users/benpfeffer/Library/Mobile Documents/com~apple~CloudDocs/projects-portfolio-main/5-octobre"
)

from src.data_preprocessing import preprocess_data
from src.metrics import compute_revenue_over_time

logger = logging.getLogger(__name__)


def prepare_time_series_features(series, lags=12, forecast_horizon=1, external_df=None):
    """
    Prepare a feature matrix from a time series with optional external regressors.

    Parameters:
    - series: pd.Series indexed by date with monthly aggregated data (e.g., revenue)
    - lags: how many lags to use as features
    - forecast_horizon: forecast steps ahead (1 for next month)
    - external_df: Optional DataFrame with external regressors, indexed by the same frequency

    Returns:
    X, y, dates
    """
    df = pd.DataFrame(series)
    df.columns = ["y"]

    # Create lag features
    for i in range(1, lags + 1):
        df[f"lag_{i}"] = df["y"].shift(i)

    # Month as feature (captures seasonality)
    df["month"] = df.index.month
    # Optional: Add quarter
    df["quarter"] = df.index.quarter

    # Rolling stats
    df["rolling_3_mean"] = df["y"].shift(1).rolling(3).mean()
    df["rolling_6_mean"] = df["y"].shift(1).rolling(6).mean()

    # Integrate external regressors if provided
    # external_df should have the same index (monthly)
    if external_df is not None:
        df = df.join(external_df, how="left")

    df = df.dropna()

    # Target is horizon steps ahead
    df["y_target"] = df["y"].shift(-forecast_horizon)
    df = df.dropna()

    y = df["y_target"]
    feature_cols = [
        c
        for c in df.columns
        if c.startswith("lag_") or c.startswith("rolling_") or c in ["month", "quarter"]
    ]

    # Include external regressor columns if added
    if external_df is not None:
        extra_cols = [c for c in external_df.columns if c in df.columns]
        feature_cols.extend(extra_cols)

    X = df[feature_cols]
    dates = df.index
    return X, y, dates


def ml_forecast_monthly_revenue(
    order_df,
    date_col="Date",
    total_col="Total",
    lags=12,
    forecast_horizon=1,
    future_periods=6,
    external_df=None,
):
    """
    Forecast monthly revenue using ML with enhancements:
    - GridSearch for hyperparameter tuning
    - TimeSeries cross-validation
    - External regressors (optional)

    Steps:
    1. Compute monthly revenue
    2. Prepare features and target
    3. TimeSeriesSplit for CV and GridSearchCV for hyperparam tuning
    4. Train best model
    5. Evaluate and forecast future
    """
    monthly_revenue = compute_revenue_over_time(
        order_df, freq="M", date_col=date_col, total_col=total_col
    )
    if monthly_revenue.empty:
        logger.error("Monthly revenue series is empty, cannot forecast.")
        return pd.DataFrame()

    # Prepare features
    X, y, dates = prepare_time_series_features(
        monthly_revenue,
        lags=lags,
        forecast_horizon=forecast_horizon,
        external_df=external_df,
    )

    if len(X) < 24:
        logger.warning("Not enough data to perform a meaningful forecast.")
        return pd.DataFrame()

    # Time series split for validation
    tscv = TimeSeriesSplit(n_splits=3)

    # Parameter grid for XGB tuning
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [3, 5],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.8, 1.0],
    }

    model = XGBRegressor(random_state=42)
    gsearch = GridSearchCV(
        model, param_grid, cv=tscv, scoring="neg_mean_absolute_error", n_jobs=-1
    )
    gsearch.fit(X, y)

    logger.info(
        f"Best parameters: {gsearch.best_params_}, Best score: {-gsearch.best_score_:.2f}"
    )
    best_model = gsearch.best_estimator_

    # Evaluate on a last hold-out period if desired:
    # For final evaluation, we can keep the last few months as final test
    test_size = 6
    train_size = len(X) - test_size
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    logger.info(f"Hold-out Test MAE: {mae:.2f}")

    # Forecast future periods
    # We'll iteratively predict one step ahead and append predictions to lag features
    current_series = monthly_revenue.copy()
    current_date = monthly_revenue.index[-1]

    for step in range(future_periods):
        # Recompute features for the current extended series
        X_future, _, _ = prepare_time_series_features(
            current_series,
            lags=lags,
            forecast_horizon=forecast_horizon,
            external_df=external_df,
        )
        X_future_point = X_future.iloc[[-1]]  # last available row for next pred
        y_future_pred = best_model.predict(X_future_point)[0]

        # Add the predicted point to current_series for next iteration
        next_date = current_date + pd.offsets.MonthEnd(forecast_horizon)
        current_series = pd.concat(
            [current_series, pd.Series([y_future_pred], index=[next_date])]
        )
        current_date = next_date

    # The last `future_periods` points of current_series are our forecasts
    forecast_points = current_series.iloc[-future_periods:]
    forecast_df = pd.DataFrame(
        {"ds": forecast_points.index, "yhat": forecast_points.values}
    )

    return forecast_df


# Example usage:
if __name__ == "__main__":
    cart_df, order_df = preprocess_data()
    # Suppose we have an external_df with monthly marketing spend aligned to the same dates:
    # external_df = pd.read_csv('marketing_spend.csv', parse_dates=['Date'], index_col='Date')
    # external_df should have monthly frequency and a column like 'marketing_spend'.

    external_df = None  # replace with actual external DataFrame if available

    if order_df is not None:
        future_forecast = ml_forecast_monthly_revenue(
            order_df,
            lags=12,
            forecast_horizon=1,
            future_periods=6,
            external_df=external_df,
        )
        print("Enhanced ML-based Future Forecast:")
        print(future_forecast)
