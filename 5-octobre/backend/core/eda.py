"""
Exploratory Data Analysis (EDA) Script

This script:
1. Loads the cleaned CSV files for cart, order, inventory, and retail.
2. Performs basic descriptive analysis (shapes, columns, missing values, summary stats).
3. Generates simple visualizations (histograms, correlations).
4. Saves plots (if desired) to an output folder.

Instructions:
    python eda_analysis.py

Make sure to adjust file paths in CLEANED_DATA_DIR and FILENAMES to match your environment.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 0. Configuration
# ------------------------------------------------------------------------------
BASE_DIR = "/Users/benpfeffer/Library/Mobile Documents/com~apple~CloudDocs/projects-portfolio-main/5-octobre/output"
EDA_PLOTS_DIR = os.path.join(BASE_DIR, "fig")
os.makedirs(EDA_PLOTS_DIR, exist_ok=True)

import sys

sys.path.append(
    "/Users/benpfeffer/Library/Mobile Documents/com~apple~CloudDocs/projects-portfolio-main/5-octobre/"
)
# Import preprocessed data from pipeline

from backend.data_preprocessing.pipeline import run_pipeline


# Load cleaned data from preprocessing pipeline
def load_cleaned_data() -> (
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
):
    """
    Load cleaned DataFrames from the preprocessing pipeline.
    Returns tuple of (cart_df, order_df, inventory_df, retail_df)
    """
    try:
        cart_df, order_df, inventory_df, retail_df = run_pipeline()
        if any(df is None for df in [cart_df, order_df, inventory_df, retail_df]):
            print("[ERROR] One or more DataFrames failed to load")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        print("[INFO] Successfully loaded all cleaned DataFrames")
        return cart_df, order_df, inventory_df, retail_df

    except Exception as e:
        print(f"[ERROR] Failed to load data from pipeline: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()


def basic_info(df: pd.DataFrame, df_name: str) -> None:
    """
    Print basic information about a DataFrame: shape, columns, missing values,
    and descriptive stats.
    """
    print(f"\n=== {df_name.upper()} DataFrame Info ===")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # Check for missing values
    missing_count = df.isna().sum()
    print(f"\nMissing Values:\n{missing_count[missing_count > 0]}")

    # Display dtypes
    print("\nData Types:")
    print(df.dtypes)

    # Basic numerical summary
    print("\nBasic Descriptive Statistics (numeric columns):")
    print(df.describe(include=[float, int]))


def plot_histograms(df: pd.DataFrame, numeric_cols: list, df_name: str) -> None:
    """
    Generate histograms for a list of numeric columns in the given DataFrame.
    """
    for col in numeric_cols:
        if col not in df.columns:
            continue
        plt.figure(figsize=(6, 4))
        sns.histplot(df[col], kde=True, bins=30)
        plt.title(f"{df_name}: Distribution of {col}")
        plt.xlabel(col)
        plt.tight_layout()

        # Save the figure to EDA_PLOTS_DIR
        plot_path = os.path.join(EDA_PLOTS_DIR, f"{df_name}_hist_{col}.png")
        plt.savefig(plot_path)
        plt.close()  # Close the plot to free memory
        print(f"[INFO] Saved histogram for {col} -> {plot_path}")


def plot_correlation_heatmap(df: pd.DataFrame, df_name: str) -> None:
    """
    Generate and save a correlation heatmap for the numeric columns of the DataFrame.
    """
    if df.select_dtypes(include=[float, int]).empty:
        print(f"[WARN] No numeric columns found in {df_name} to compute correlation.")
        return

    corr = df.select_dtypes(include=[float, int]).corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", square=True)
    plt.title(f"{df_name}: Correlation Heatmap")
    plt.tight_layout()

    plot_path = os.path.join(EDA_PLOTS_DIR, f"{df_name}_corr_heatmap.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"[INFO] Saved correlation heatmap -> {plot_path}")


# 2. Main EDA Flow
# ------------------------------------------------------------------------------
def main():
    """
    Main function that orchestrates the EDA steps:
    1. Load cleaned data (cart, order, inventory, retail).
    2. Print basic info for each DataFrame.
    3. Plot histograms and correlation heatmaps (where numeric columns exist).
    """
    # Load data
    cart_df, order_df, inventory_df, retail_df = load_cleaned_data()

    # Basic info
    basic_info(cart_df, "cart")
    basic_info(order_df, "order")
    basic_info(inventory_df, "inventory")
    basic_info(retail_df, "retail")

    # Plot histograms for certain numeric columns
    # Adjust the list of numeric columns based on your dataset.
    cart_numeric = ["total"]  # Example
    order_numeric = ["total"]  # Example
    inventory_numeric = ["qty", "factory_price", "retail", "retail_us"]
    retail_numeric = ["pv_ttc", "ca_ttc"]

    plot_histograms(cart_df, cart_numeric, "cart")
    plot_histograms(order_df, order_numeric, "order")
    plot_histograms(inventory_df, inventory_numeric, "inventory")
    plot_histograms(retail_df, retail_numeric, "retail")

    # Plot correlation heatmaps
    plot_correlation_heatmap(cart_df, "cart")
    plot_correlation_heatmap(order_df, "order")
    plot_correlation_heatmap(inventory_df, "inventory")
    plot_correlation_heatmap(retail_df, "retail")

    print(
        "\n[INFO] EDA analysis complete. Check console output and eda_plots directory for results."
    )


if __name__ == "__main__":
    main()
