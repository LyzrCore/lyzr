import logging
from typing import List, Set

from bs4 import BeautifulSoup, Tag
from llama_index.schema import Document
from playwright.sync_api import sync_playwright

logger = logging.getLogger(__name__)

CONTENT_TAGS = [
    "p",
    "div",
    "span",
    "a",
    "td",
    "tr",
    "li",
    "article",
    "section",
    "pre",
    "code",
    "blockquote",
    "em",
    "strong",
    "b",
    "i",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "title",
]


def scrape(html: str) -> str:
    soup: BeautifulSoup = BeautifulSoup(html, "html.parser")

    content: List[Tag] = soup.find_all(CONTENT_TAGS)

    text_set: Set[str] = set()

    for p in content:
        for text in p.stripped_strings:
            text_set.add(text)

    return " ".join(text_set)


def load_content_using_playwright(url: str) -> str:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(url)
        html = page.content()
        content = scrape(html)
        browser.close()
    return content


class LyzrWebPageReader:
    @staticmethod
    def load_data(url: str) -> List[Document]:
        content = load_content_using_playwright(url)
        document = Document(text=content, metadata={"url": url})
        return [document]
