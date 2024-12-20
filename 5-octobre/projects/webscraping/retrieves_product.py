from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import time
import pandas as pd
import re


def setup_driver():
    # ds: Setup Chrome options for headless mode
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")  # Run in headless mode
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    # ds: Returning a headless Chrome webdriver instance
    return webdriver.Chrome(options=options)


def extract_product_id_from_url(url):
    """
    ds: Extract product ID from the product URL.
    The URL format: https://www.5octobre.com/en/accueil/<id>-<rest of product name and code>
    For example:
    'https://www.5octobre.com/en/accueil/880-flora-morga-earrings-3666023064947' -> '880'
    """
    match = re.search(r"/accueil/(\d+)-", url)
    if match:
        return match.group(1)
    return ""


def get_products_from_page(driver, url):
    """
    ds: Extract products (name, price, and product id) from a single page.
    Navigates to the given URL, parses the page, and returns a list of dictionaries
    with product name, price, and id.
    """
    driver.get(url)
    time.sleep(3)  # ds: Let content load, adjust if needed
    soup = BeautifulSoup(driver.page_source, "html.parser")

    product_items = soup.find_all("article", class_="product-miniature")
    products = []

    for item in product_items:
        try:
            # ds: Extract product name and URL
            name_element = item.find("div", class_="cinqOct_productTitle")
            if name_element and name_element.find("a"):
                name_tag = name_element.find("a")
                name = name_tag.get_text(strip=True)
                product_url = name_tag["href"]
            else:
                name = "N/A"
                product_url = ""

            # ds: Extract product price
            price_element = item.find("span", class_="price")
            if price_element:
                price_raw = price_element.get_text(strip=True)
                # ds: Remove the currency symbol and any thousands separator
                price_cleaned = price_raw.replace("€", "").replace(",", "").strip()
                try:
                    price = float(price_cleaned)
                except ValueError:
                    price = "N/A"
            else:
                price = "N/A"

            # ds: Extract product ID from URL
            product_id = extract_product_id_from_url(product_url)

            products.append({"id": product_id, "name": name, "price": price})

        except Exception as e:
            print(f"Error processing product: {e}")
            continue

    return products


def has_next_page(soup):
    """
    ds: Check if there is a Next link in the pagination.
    If no 'next' link is found, it implies this is the last page.
    """
    next_link = soup.select_one(".cinqOct_pageListContainer .next")
    return next_link is not None


def scrape_all_products():
    """
    ds: Scrape all product pages starting from the base URL.
    Paginate through all results, gather product info, and save to CSV.
    The script extracts product ID from the product URL and includes it in the final CSV.
    """
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
                # ds: If no products found on this page, break out
                break

            all_products.extend(products)
            print(f"Found {len(products)} products on page {page} (Total so far: {len(all_products)})")

            # ds: Check if there's a next page
            if not has_next_page(soup):
                break

            page += 1

        # ds: Save all products to CSV
        df = pd.DataFrame(all_products)
        df.to_csv("5octobre_products.csv", index=False, encoding="utf-8")
        print(f"Scraped {len(all_products)} products and saved to 5octobre_products.csv with 'id' column.")

    except Exception as e:
        print(f"Error scraping: {e}")
    finally:
        driver.quit()


if __name__ == "__main__":
    scrape_all_products()
