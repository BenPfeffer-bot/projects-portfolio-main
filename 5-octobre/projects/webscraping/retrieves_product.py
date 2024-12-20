from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import time
import pandas as pd


def setup_driver():
    # Setup Chrome options
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")  # Run in headless mode
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    return webdriver.Chrome(options=options)


def get_products_from_page(driver, url):
    """Extract products (name and price) from a single page."""
    driver.get(url)
    time.sleep(3)  # Let content load, adjust if needed
    soup = BeautifulSoup(driver.page_source, "html.parser")

    product_items = soup.find_all("article", class_="product-miniature")
    products = []

    for item in product_items:
        try:
            # Extract product name
            name_element = item.find("div", class_="cinqOct_productTitle")
            if name_element and name_element.find("a"):
                name = name_element.find("a").get_text(strip=True)
            else:
                name = "N/A"

            # Extract product price
            price_element = item.find("span", class_="price")
            if price_element:
                price_raw = price_element.get_text(strip=True)
                # Remove the currency symbol and any thousands separator
                price_cleaned = price_raw.replace("€", "").replace(",", "").strip()
                try:
                    price = float(price_cleaned)
                except ValueError:
                    price = "N/A"
            else:
                price = "N/A"

            products.append({"name": name, "price": price})

        except Exception as e:
            print(f"Error processing product: {e}")
            continue

    return products


def has_next_page(soup):
    """Check if there is a Next link in the pagination."""
    next_link = soup.select_one(".cinqOct_pageListContainer .next")
    return next_link is not None


def scrape_all_products():
    base_url = "https://www.5octobre.com/en/12-jewelry"
    results_per_page = 19
    page = 1

    driver = setup_driver()
    all_products = []

    try:
        while True:
            page_url = f"{base_url}?resultsPerPage={results_per_page}&page={page}"
            print(f"Scraping page {page}: {page_url}")
            driver.get(page_url)
            time.sleep(3)
            soup = BeautifulSoup(driver.page_source, "html.parser")

            products = get_products_from_page(driver, page_url)
            if not products:
                # If no products found on this page, break out
                break

            all_products.extend(products)
            print(f"Found {len(products)} products on page {page} (Total so far: {len(all_products)})")

            # Check if there's a next page
            if not has_next_page(soup):
                break

            page += 1

        # Save to CSV
        df = pd.DataFrame(all_products)
        df.to_csv("5octobre_products.csv", index=False, encoding="utf-8")
        print(f"Scraped {len(all_products)} products and saved to 5octobre_products.csv")

    except Exception as e:
        print(f"Error scraping: {e}")
    finally:
        driver.quit()


if __name__ == "__main__":
    scrape_all_products()
