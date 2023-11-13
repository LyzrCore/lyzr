import asyncio
import nest_asyncio
import logging
from typing import List, Set

from bs4 import BeautifulSoup, Tag
from llama_index.schema import Document
from playwright.async_api import async_playwright

logger = logging.getLogger(__name__)

nest_asyncio.apply()

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


async def async_load_content_using_playwright(url: str) -> str:
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.goto(url)
        html = await page.content()
        await browser.close()
        return html


def load_content_using_playwright(url: str) -> str:
    return asyncio.get_event_loop().run_until_complete(
        async_load_content_using_playwright(url)
    )


class LyzrWebPageReader:
    @staticmethod
    def load_data(url: str) -> List[Document]:
        html = load_content_using_playwright(url)
        content = scrape(html)
        document = Document(text=content, metadata={"url": url})
        return [document]
