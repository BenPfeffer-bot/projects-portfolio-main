import pandas as pd
from prophet import Prophet
import sys
import matplotlib.pyplot as plt

sys.path.append(
    "/Users/benpfeffer/Library/Mobile Documents/com~apple~CloudDocs/projects-portfolio-main/5-octobre"
)
from src.data_preprocessing import preprocess_data


def forecast_monthly_customers(
    order_df, periods=6, date_col="Date", client_col="Client"
):
    # Compute monthly unique customers
    order_df[date_col] = pd.to_datetime(order_df[date_col])
    monthly_customers = (
        order_df.set_index(date_col).groupby(pd.Grouper(freq="M"))[client_col].nunique()
    )

    df = monthly_customers.reset_index()
    df.columns = ["ds", "y"]

    if df.empty:
        return pd.DataFrame(), None

    model = Prophet()
    model.fit(df)

    future = model.make_future_dataframe(periods=periods, freq="M")
    forecast = model.predict(future)
    return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]], model


def plot_forecast(forecast, model):
    fig = model.plot(forecast)
    plt.title("Customers Forecast")
    plt.xlabel("Date")
    plt.ylabel("Number of Customers")
    plt.show()


if __name__ == "__main__":
    order_df, cart_df = preprocess_data()
    print(forecast_monthly_customers(order_df))
    forecast, model = forecast_monthly_customers(order_df)
    plot_forecast(forecast, model)
