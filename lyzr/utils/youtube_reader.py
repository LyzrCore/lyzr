from typing import List

from llama_index.readers.base import BaseReader
from llama_index.readers.youtube_transcript import YoutubeTranscriptReader
from llama_index.schema import Document


class LyzrYoutubeReader(BaseReader):
    def __init__(self) -> None:
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
        except ImportError:
            raise ImportError(
                "`youtube_transcript_api` package not found, \
                    please run `pip install youtube-transcript-api`"
            )

    def load_data(self, urls: List[str]) -> List[Document]:
        loader = YoutubeTranscriptReader()
        documents = loader.load_data(ytlinks=urls)
        return documents
