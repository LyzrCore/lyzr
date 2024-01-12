from typing import Optional, Sequence

from llama_index import Document, ServiceContext, VectorStoreIndex, StorageContext
from llama_index.node_parser import SimpleNodeParser


def import_vector_store_class(vector_store_class_name: str):
    module = __import__("llama_index.vector_stores", fromlist=[vector_store_class_name])
    class_ = getattr(module, vector_store_class_name)
    return class_


class LyzrVectorStoreIndex:
    @staticmethod
    def from_defaults(
        vector_store_type: str = "LanceDBVectorStore",
        documents: Optional[Sequence[Document]] = None,
        service_context: Optional[ServiceContext] = None,
        **kwargs
    ) -> VectorStoreIndex:
        if documents is None and vector_store_type == "SimpleVectorStore":
            raise ValueError("documents must be provided for SimpleVectorStore")

        vector_store_class = import_vector_store_class(vector_store_type)

        if documents is None:
            vector_store = vector_store_class(**kwargs)
            index = VectorStoreIndex.from_vector_store(
                vector_store=vector_store, service_context=service_context
            )
        else:
            if vector_store_type == "LanceDBVectorStore":
                kwargs["uri"] = "./.lancedb" if "uri" not in kwargs else kwargs["uri"]
                kwargs["table_name"] = (
                    "vectors" if "table_name" not in kwargs else kwargs["table_name"]
                )
            vector_store = vector_store_class(**kwargs)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)

            index = VectorStoreIndex.from_documents(
                documents=documents,
                storage_context=storage_context,
                service_context=service_context,
                show_progress=True,
            )

        return index
