from typing import List

from langchain.document_loaders import Docx2txtLoader
from llama_index.readers.base import BaseReader
from llama_index.schema import Document


class LyzrDocxReader(BaseReader):
    def __init__(self) -> None:
        try:
            import docx2txt
        except ImportError:
            raise ImportError(
                "`docx2txt` package not found, please run `pip install docx2txt`"
            )

    def load_data(self, file_path: str, extra_info: dict = None) -> List[Document]:
        loader = Docx2txtLoader(str(file_path))
        langchain_documents = loader.load()

        documents = []
        for langchain_document in langchain_documents:
            doc = Document.from_langchain_format(langchain_document)

            if extra_info is not None:
                doc.metadata.update(extra_info)

            documents.append(doc)

        return documents
