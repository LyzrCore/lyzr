from typing import List

from llmsherpa.readers import LayoutPDFReader
from llama_index.readers.base import BaseReader
from llama_index.schema import Document


class LyzrPDFReader(BaseReader):
    def __init__(self) -> None:
        try:
            from llmsherpa.readers import LayoutPDFReader
        except ImportError:
            raise ImportError(
                "`llmsherpa` package not found, please install it with "
                "`pip install llmsherpa`"
            )

    def load_data(self, file_path: str, extra_info: dict = None) -> List[Document]:
        llmsherpa_api_url = "https://readers.llmsherpa.com/api/document/developer/parseDocument?renderFormat=all"
        loader = LayoutPDFReader(llmsherpa_api_url)

        doc = loader.read_pdf(str(file_path))
        metadata = {"source": str(file_path)} 
        documents = []
        for chunk in doc.chunks():
            document = Document(text=chunk.to_context_text(), metadata=metadata)
            documents.append(document)

        return documents
