import os
import re
import time
import unicodedata
import html  

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

from config import VALID_LINK_SUBSTRING, VALID_LINK_EXTENSION, EXCLUDE_EXTENSIONS

# Directory to store scraped documents (ensure this exists or will be created)
DOCS_DIR = "./confluence_docs"
visited = set()
session = requests.Session()
session.headers.update({"User-Agent": "Mozilla/5.0"})


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
    """
    Optimized text cleaning for improved embedding quality:
    - Unescape HTML entities
    - Remove code fences, inline code, HTML tags, wiki markup
    - Strip markdown links, URLs, emails, citations
    - Normalize punctuation and unicode
    - Lowercase and normalize whitespace
    """
    # Unescape HTML entities
    text = html.unescape(text)

    # Remove code fences (```...```) and inline code (`...`)
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    text = re.sub(r"`[^`]+`", "", text)

    # Remove HTML tags and wiki markup
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\{\{[^}]+\}\}", "", text)

    # Remove markdown headings, blockquotes, and list markers
    text = re.sub(r"(?m)^[#>+\-*]\s*", "", text)

    # Strip markdown links but keep link text
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)

    # Remove URLs and email addresses
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"\S+@\S+\.\S+", "", text)

    # Remove inline citations like [1], (1)
    text = re.sub(r"\[\d+\]|\(\d+\)", "", text)

    # Remove boilerplate lines
    text = re.sub(r"(?i)(page created by|last edited by)[^\n]*", "", text)

    # Normalize punctuation: fancy quotes to ASCII, dashes
    replacements = {
        "“": "\"", "”": "\"", "‘": "'", "’": "'",
        "—": "-", "–": "-"
    }
    for orig, repl in replacements.items():
        text = text.replace(orig, repl)

    # Normalize unicode characters
    text = unicodedata.normalize("NFKC", text)

    # Remove control characters
    text = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C")

    # Lowercase for consistency
    text = text.lower()

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # final ASCII filter: drop any non‑ASCII
    ascii_text = text.encode('ascii', 'ignore').decode('ascii')
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
        response = session.get(url, timeout=10)
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
