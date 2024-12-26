import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def plot_clv(clv_values: pd.Series):
    """
    Plots the Customer Lifetime Value (CLV).

    Args:
        clv_values (pd.Series): Series of CLV values.
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(clv_values, bins=20, kde=True)
    plt.title("Customer Lifetime Value Distribution")
    plt.xlabel("CLV")
    plt.ylabel("Frequency")
    plt.show()


def plot_cac(cac_values: pd.Series):
    """
    Plots the Customer Acquisition Cost (CAC).

    Args:
        cac_values (pd.Series): Series of CAC values.
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(cac_values, bins=20, kde=True)
    plt.title("Customer Acquisition Cost Distribution")
    plt.xlabel("CAC")
    plt.ylabel("Frequency")
    plt.show()


def plot_rfm_distribution(rfm_df: pd.DataFrame):
    """
    Plots the RFM-Based Segment Distribution.

    Args:
        rfm_df (pd.DataFrame): DataFrame with RFM segments.
    """
    plt.figure(figsize=(10, 6))
    sns.countplot(
        data=rfm_df, x="Segment", order=["Low", "Medium", "High", "Very High"]
    )
    plt.title("RFM Segment Distribution")
    plt.xlabel("Segment")
    plt.ylabel("Count")
    plt.show()


def plot_repeat_purchase_rate(rpr_values: pd.Series):
    """
    Plots the Repeat Purchase Rate (RPR).

    Args:
        rpr_values (pd.Series): Series of RPR values.
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(rpr_values, bins=20, kde=True)
    plt.title("Repeat Purchase Rate Distribution")
    plt.xlabel("RPR (%)")
    plt.ylabel("Frequency")
    plt.show()


def plot_on_time_delivery_rate(otd_values: pd.Series):
    """
    Plots the On-Time Delivery Rate.

    Args:
        otd_values (pd.Series): Series of On-Time Delivery Rate values.
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(otd_values, bins=20, kde=True)
    plt.title("On-Time Delivery Rate Distribution")
    plt.xlabel("On-Time Delivery Rate (%)")
    plt.ylabel("Frequency")
    plt.show()
