from typing import Union, Optional, List

from llama_index import ServiceContext, VectorStoreIndex
from llama_index.embeddings.utils import EmbedType
from llama_index.indices.query.base import BaseQueryEngine

from lyzr.utils.search_utils import (
    add_pdf_,
    add_docx_,
    add_text_,
    add_webpage_,
    add_website_,
    add_youtube_
)


class SearchAgent:
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
    def add_pdf(
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
        return add_pdf_(
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
    def add_docx(
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
        return add_docx_(
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
    def add_text(
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
        return add_text_(
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
    def add_webpage(
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
        return add_webpage_(
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
    def add_website(
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
        return add_website_(
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
    def add_youtube(
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
        return add_youtube_(
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
