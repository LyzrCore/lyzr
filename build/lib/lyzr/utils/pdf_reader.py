from typing import List

from langchain.document_loaders import PDFMinerLoader
from llama_index.readers.base import BaseReader
from llama_index.schema import Document


class LyzrPDFReader(BaseReader):
    def __init__(self) -> None:
        try:
            from pdfminer.high_level import extract_text 
        except ImportError:
            raise ImportError(
                "`pdfminer` package not found, please install it with "
                "`pip install pdfminer.six`"
            )

    def load_data(self, file_path: str, extra_info: dict = None) -> List[Document]:
        loader = PDFMinerLoader(str(file_path))
        langchain_documents = loader.load()  

        documents = []
        for langchain_document in langchain_documents:
            doc = Document.from_langchain_format(langchain_document)

            if extra_info is not None:
                doc.metadata.update(extra_info)

            documents.append(doc)

        return documents
