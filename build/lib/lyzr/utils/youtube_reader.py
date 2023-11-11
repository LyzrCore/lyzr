from typing import List

from llama_index.readers.base import BaseReader
from llama_index.readers.youtube_transcript import YoutubeTranscriptReader
from llama_index.schema import Document


class LyzrYoutubeReader(BaseReader):
    def __init__(self) -> None:
        None

    def load_data(self, urls: List[str]) -> List[Document]:
        loader = YoutubeTranscriptReader()
        documents = loader.load_data(ytlinks=urls)
        return documents
