from typing import Optional, Sequence

import os
import uuid
import weaviate
from weaviate.embedded import EmbeddedOptions
from llama_index import Document, ServiceContext, VectorStoreIndex, StorageContext
from llama_index.node_parser import SimpleNodeParser


def import_vector_store_class(vector_store_class_name: str):
    module = __import__("llama_index.vector_stores", fromlist=[vector_store_class_name])
    class_ = getattr(module, vector_store_class_name)
    return class_


class LyzrVectorStoreIndex:
    @staticmethod
    def from_defaults(
        vector_store_type: str = "WeaviateVectorStore",
        documents: Optional[Sequence[Document]] = None,
        service_context: Optional[ServiceContext] = None,
        **kwargs
    ) -> VectorStoreIndex:
        if documents is None and vector_store_type == "SimpleVectorStore":
            raise ValueError("documents must be provided for SimpleVectorStore")

        VectorStoreClass = import_vector_store_class(vector_store_type)

        if vector_store_type == "WeaviateVectorStore":
            weaviate_client = weaviate.Client(
                embedded_options=weaviate.embedded.EmbeddedOptions(),
                additional_headers={"X-OpenAI-Api-Key": os.environ["OPENAI_API_KEY"]},
            )
            kwargs["weaviate_client"] = (
                weaviate_client
                if "weaviate_client" not in kwargs
                else kwargs["weaviate_client"]
            )
            kwargs["index_name"] = (
                f"DB_{uuid.uuid4().hex}" if "index_name" not in kwargs else kwargs["index_name"]
            )

            vector_store = VectorStoreClass(**kwargs)
        else:
            vector_store = VectorStoreClass(**kwargs)

        if documents is None:
            index = VectorStoreIndex.from_vector_store(
                vector_store=vector_store, service_context=service_context
            )
            return index

        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        if documents is not None:
            index = VectorStoreIndex.from_documents(
                documents=documents,
                storage_context=storage_context,
                service_context=service_context,
                show_progress=True,
            )

        return index
