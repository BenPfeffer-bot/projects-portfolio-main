import requests
from bs4 import BeautifulSoup


def get_html(url):
    response = requests.get(url)
    return response.text


def get_soup(html):
    return BeautifulSoup(html, "html.parser")


def get_data(url):
    html = get_html(url)
    soup = get_soup(html)
    return soup


def save_html(html, filename="output.html"):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(html)


get_data("https://www.5octobre.com/en/12-jewelry")
save_html(get_html("https://www.5octobre.com/en/12-jewelry"))
