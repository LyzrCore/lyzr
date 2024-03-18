from typing import Optional, Sequence
from llama_index.retrievers import BaseRetriever
from llama_index.indices import VectorStoreIndex


def import_retriever_class(retriever_class_name: str):
    module = __import__("llama_index.retrievers", fromlist=[retriever_class_name])
    class_ = getattr(module, retriever_class_name)
    return class_


class LyzrRetriever:
    @staticmethod
    def from_defaults(
        retriever_type: str = "QueryFusionRetriever",
        base_index: VectorStoreIndex = None,
        **kwargs
    ) -> BaseRetriever:
        RetrieverClass = import_retriever_class(retriever_type)

        if retriever_type == "QueryFusionRetriever":
            print("QueryFusionRetriever")
            retriever = RetrieverClass(
                retrievers=[
                    base_index.as_retriever(
                        vector_store_query_mode="mmr",
                        similarity_top_k=3,
                        vector_store_kwargs={"mmr_threshold": 0.1},
                    ),
                    base_index.as_retriever(
                        vector_store_query_mode="mmr",
                        similarity_top_k=3,
                        vector_store_kwargs={"mmr_threshold": 0.1},
                    ),
                ],
                similarity_top_k=5,
                num_queries=2,
                use_async=False,
                **kwargs
            )
            return retriever
        else:
            retriever = RetrieverClass(**kwargs)
            return retriever
