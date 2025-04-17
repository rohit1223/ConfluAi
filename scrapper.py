import os
import re
import time
import unicodedata

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

from config import VALID_LINK_SUBSTRING, VALID_LINK_EXTENSION, EXCLUDE_EXTENSIONS

# Directory to store scraped documents (ensure this exists or will be created)
DOCS_DIR = "./confluence_docs"

HEADERS = {"User-Agent": "Mozilla/5.0"}
visited = set()


def load_base_urls(file_path):
    """Load base URLs from a file, one URL per line."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Base URLs file not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        urls = [line.strip() for line in f if line.strip()]
    return urls


def is_valid_link(link):
    return (
        link and
        VALID_LINK_SUBSTRING in link and
        link.endswith(VALID_LINK_EXTENSION) and
        not any(ext in link for ext in EXCLUDE_EXTENSIONS)
    )


def clean_text(text: str) -> str:
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    # Remove wiki macros or content inside curly braces
    text = re.sub(r"\{[^}]+\}", "", text)
    # Remove Markdown-style formatting (bold, underscores, dashes)
    text = re.sub(r"(\*{2,}|\_{2,}|-{2,})", "", text)
    # Remove boilerplate text like "page created by..." or "last edited by..."
    # (case-insensitive)
    text = re.sub(r"(?i)page created by.*|last edited by.*", "", text)
    # Normalize extra whitespace (multiple spaces into one)
    text = re.sub(r"\s{2,}", " ", text)
    # Collapse multiple newlines into a single newline
    text = re.sub(r'\n+', '\n', text)
    # Normalize the text to decompose combined characters (e.g., diacritics)
    normalized_text = unicodedata.normalize('NFKD', text)
    # Encode to ASCII and ignore non-ASCII characters (e.g., unwanted
    # artifacts like "Â")
    ascii_text = normalized_text.encode('ascii', 'ignore').decode('ascii')
    return ascii_text.strip()


def slugify(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9]+', '_', text)
    return text.strip('_')


def scrape(url):
    if url in visited:
        return
    print(f"[+] Scraping: {url}")
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        if response.status_code != 200:
            print(f"[-] Skipped {url} with status {response.status_code}")
            return
        soup = BeautifulSoup(response.text, 'html.parser')

        title = soup.title.string.strip() if soup.title else "no_title"
        headers_list = [tag.get_text(strip=True)
                        for tag in soup.find_all(['h1', 'h2', 'h3'])]
        content_paragraphs = [
            clean_text(
                p.get_text()) for p in soup.find_all('p') if p.get_text(
                strip=True)]

        full_text = f"Title: {title}\n\n"
        if headers_list:
            full_text += "Headers:\n" + "\n".join(headers_list) + "\n\n"
        if content_paragraphs:
            full_text += "Content:\n" + "\n".join(content_paragraphs)

        safe_title = slugify(title)
        if not safe_title:
            safe_title = slugify(url)
        filename = os.path.join(DOCS_DIR, f"{safe_title}.txt")
        os.makedirs(DOCS_DIR, exist_ok=True)
        with open(filename, "w", encoding="utf-8") as f:
            f.write(full_text)
        print(f"[✓] Saved: {filename}")

        visited.add(url)

        for a_tag in soup.find_all('a', href=True):
            full_url = urljoin(url, a_tag['href'])
            if is_valid_link(full_url) and full_url not in visited:
                time.sleep(3)
                scrape(full_url)

    except Exception as e:
        print(f"[!] Failed to scrape {url}: {e}")


# Load base URLs from the file
BASE_URLS = load_base_urls("base_urls.txt")

# Start scraping from each base URL
for base_url in BASE_URLS:
    scrape(base_url)

print("[✓] Done. Scraped pages have been saved in", DOCS_DIR)
