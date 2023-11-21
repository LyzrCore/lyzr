import sys 
import asyncio
import logging
import warnings
import nest_asyncio
from typing import List, Set
from bs4 import BeautifulSoup, Tag
from typing import List
from llama_index.schema import Document 

IS_IPYKERNEL = "ipykernel_launcher" in sys.argv[0]

if IS_IPYKERNEL:
    nest_asyncio.apply()

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


async def async_load_content_using_playwright(url: str) -> str:
    
    try:
        from playwright.async_api import async_playwright
        
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()
            await page.goto(url)
            html = await page.content()
            await browser.close()
            return html

    except ImportError:
            raise ImportError(
                "`playwright` package not found, please install it with "
                "`pip install playwright && playwright install`"
            )

def load_content_using_playwright(url: str) -> str:
    return asyncio.get_event_loop().run_until_complete(
        async_load_content_using_playwright(url)
    )

class LyzrWebPageReader:
    
    def __init__(self) -> None:
        pass

    @staticmethod
    def load_data(url: str) -> List[Document]:
        if IS_IPYKERNEL:
            warning_msg = "Running in Google Colab or a Jupyter notebook. Consider using nest_asyncio.apply() to avoid event loop conflicts."
            warnings.warn(warning_msg, RuntimeWarning)
        
        html = load_content_using_playwright(url)
        content = scrape(html)
        document = Document(text=content, metadata={"url": url})
        return [document]
