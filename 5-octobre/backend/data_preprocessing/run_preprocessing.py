from pipeline import run_pipeline

if __name__ == "__main__":
    """
    Usage:
        python run_preprocessing.py
    """
    cart_df, order_df, inventory_df, retail_df = run_pipeline()
    if any(df is None for df in [cart_df, order_df, inventory_df, retail_df]):
        print("[ERROR] Preprocessing pipeline encountered an error.")
    else:
        print("[INFO] Preprocessing pipeline completed successfully!")
