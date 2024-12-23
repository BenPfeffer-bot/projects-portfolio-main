import os

# Base data directory
BASE_DATA_DIR = "/Users/benpfeffer/Library/Mobile Documents/com~apple~CloudDocs/projects-portfolio-main/5-octobre/"

# Subdirectories
RAW_DATA_DIR = os.path.join(BASE_DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(BASE_DATA_DIR, "processed")
CLEANED_DATA_DIR = os.path.join(BASE_DATA_DIR, "cleaned")

# Filenames
CART_FILENAME = "cart.csv"
ORDER_FILENAME = "order.csv"
INVENTORY_FILENAME = "inventory.csv"
RETAIL_FILENAME = "retail.csv"
PRODUCTS_FILENAME = "products.csv"
