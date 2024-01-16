from typing import Union, Optional, List

from llama_index import ServiceContext, VectorStoreIndex
from llama_index.embeddings.utils import EmbedType
from llama_index.indices.query.base import BaseQueryEngine

from lyzr.utils.rag_utils import (
    pdf_rag,
    txt_rag,
    docx_rag,
    webpage_rag,
    website_rag,
    youtube_rag,
)


class QABot:
    def __init__(self) -> None:
        return None

    @staticmethod
    def from_instances(
        vector_store_index: VectorStoreIndex, service_context: ServiceContext, **kwargs
    ) -> BaseQueryEngine:
        return vector_store_index.as_query_engine(
            service_context=service_context, **kwargs
        )

    @staticmethod
    def pdf_qa(
        input_dir: Optional[str] = None,
        input_files: Optional[List] = None,
        exclude_hidden: bool = True,
        filename_as_id: bool = True,
        recursive: bool = True,
        required_exts: Optional[List[str]] = None,
        system_prompt: str = None,
        query_wrapper_prompt: str = None,
        embed_model: Union[str, EmbedType] = "default",
        llm_params: dict = None,
        vector_store_params: dict = None,
        service_context_params: dict = None,
        query_engine_params: dict = None,
        retriever_params: dict = None,
    ) -> BaseQueryEngine:
        return pdf_rag(
            input_dir=input_dir,
            input_files=input_files,
            exclude_hidden=exclude_hidden,
            filename_as_id=filename_as_id,
            recursive=recursive,
            required_exts=required_exts,
            system_prompt=system_prompt,
            query_wrapper_prompt=query_wrapper_prompt,
            embed_model=embed_model,
            llm_params=llm_params,
            vector_store_params=vector_store_params,
            service_context_params=service_context_params,
            query_engine_params=query_engine_params,
            retriever_params=retriever_params,
        )

    @staticmethod
    def docx_qa(
        input_dir: Optional[str] = None,
        input_files: Optional[List] = None,
        exclude_hidden: bool = True,
        filename_as_id: bool = True,
        recursive: bool = True,
        required_exts: Optional[List[str]] = None,
        system_prompt: str = None,
        query_wrapper_prompt: str = None,
        embed_model: Union[str, EmbedType] = "default",
        llm_params: dict = None,
        vector_store_params: dict = None,
        service_context_params: dict = None,
        query_engine_params: dict = None,
        retriever_params: dict = None,
    ) -> BaseQueryEngine:
        return docx_rag(
            input_dir=input_dir,
            input_files=input_files,
            exclude_hidden=exclude_hidden,
            filename_as_id=filename_as_id,
            recursive=recursive,
            required_exts=required_exts,
            system_prompt=system_prompt,
            query_wrapper_prompt=query_wrapper_prompt,
            embed_model=embed_model,
            llm_params=llm_params,
            vector_store_params=vector_store_params,
            service_context_params=service_context_params,
            query_engine_params=query_engine_params,
            retriever_params=retriever_params,
        )

    @staticmethod
    def txt_qa(
        input_dir: Optional[str] = None,
        input_files: Optional[List] = None,
        exclude_hidden: bool = True,
        filename_as_id: bool = True,
        recursive: bool = True,
        required_exts: Optional[List[str]] = None,
        system_prompt: str = None,
        query_wrapper_prompt: str = None,
        embed_model: Union[str, EmbedType] = "default",
        llm_params: dict = None,
        vector_store_params: dict = None,
        service_context_params: dict = None,
        query_engine_params: dict = None,
        retriever_params: dict = None,
    ) -> BaseQueryEngine:
        return txt_rag(
            input_dir=input_dir,
            input_files=input_files,
            exclude_hidden=exclude_hidden,
            filename_as_id=filename_as_id,
            recursive=recursive,
            required_exts=required_exts,
            system_prompt=system_prompt,
            query_wrapper_prompt=query_wrapper_prompt,
            embed_model=embed_model,
            llm_params=llm_params,
            vector_store_params=vector_store_params,
            service_context_params=service_context_params,
            query_engine_params=query_engine_params,
            retriever_params=retriever_params,
        )

    @staticmethod
    def webpage_qa(
        url: Optional[str] = None,
        system_prompt: str = None,
        query_wrapper_prompt: str = None,
        embed_model: Union[str, EmbedType] = "default",
        llm_params: dict = None,
        vector_store_params: dict = None,
        service_context_params: dict = None,
        query_engine_params: dict = None,
        retriever_params: dict = None,
    ) -> BaseQueryEngine:
        return webpage_rag(
            url=url,
            system_prompt=system_prompt,
            query_wrapper_prompt=query_wrapper_prompt,
            embed_model=embed_model,
            llm_params=llm_params,
            vector_store_params=vector_store_params,
            service_context_params=service_context_params,
            query_engine_params=query_engine_params,
            retriever_params=retriever_params,
        )

    @staticmethod
    def website_qa(
        url: Optional[str] = None,
        system_prompt: str = None,
        query_wrapper_prompt: str = None,
        embed_model: Union[str, EmbedType] = "default",
        llm_params: dict = None,
        vector_store_params: dict = None,
        service_context_params: dict = None,
        query_engine_params: dict = None,
        retriever_params: dict = None,
    ) -> BaseQueryEngine:
        return website_rag(
            url=url,
            system_prompt=system_prompt,
            query_wrapper_prompt=query_wrapper_prompt,
            embed_model=embed_model,
            llm_params=llm_params,
            vector_store_params=vector_store_params,
            service_context_params=service_context_params,
            query_engine_params=query_engine_params,
            retriever_params=retriever_params,
        )

    @staticmethod
    def youtube_qa(
        urls: List[str] = None,
        system_prompt: str = None,
        query_wrapper_prompt: str = None,
        embed_model: Union[str, EmbedType] = "default",
        llm_params: dict = None,
        vector_store_params: dict = None,
        service_context_params: dict = None,
        query_engine_params: dict = None,
        retriever_params: dict = None,
    ) -> BaseQueryEngine:
        return youtube_rag(
            urls=urls,
            system_prompt=system_prompt,
            query_wrapper_prompt=query_wrapper_prompt,
            embed_model=embed_model,
            llm_params=llm_params,
            vector_store_params=vector_store_params,
            service_context_params=service_context_params,
            query_engine_params=query_engine_params,
            retriever_params=retriever_params,
        )
