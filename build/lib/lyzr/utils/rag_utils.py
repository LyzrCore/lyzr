from typing import Union, Optional, List

from llama_index.embeddings.utils import EmbedType
from llama_index.indices.query.base import BaseQueryEngine
from llama_index.query_engine import RetrieverQueryEngine

from lyzr.base.llm import LyzrLLMFactory
from lyzr.base.retrievers import LyzrRetriever
from lyzr.base.service import LyzrService
from lyzr.base.vector_store import LyzrVectorStoreIndex
from lyzr.utils.document_reading import (
    read_pdf_as_documents,
    read_docx_as_documents,
    read_txt_as_documents,
    read_website_as_documents,
    read_webpage_as_documents,
    read_youtube_as_documents,
)


def pdf_rag(
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
    documents = read_pdf_as_documents(
        input_dir=input_dir,
        input_files=input_files,
        exclude_hidden=exclude_hidden,
        filename_as_id=filename_as_id,
        recursive=recursive,
        required_exts=required_exts,
    )

    llm_params = (
        {
            "model": "gpt-4-0125-preview",
            "temperature": 0,            
        }
        if llm_params is None
        else llm_params
    )
    vector_store_params = (
        {"vector_store_type": "WeaviateVectorStore"}
        if vector_store_params is None
        else vector_store_params
    )
    service_context_params = (
        {} if service_context_params is None else service_context_params
    )
    query_engine_params = {} if query_engine_params is None else query_engine_params

    retriever_params = (
        {"retriever_type": "QueryFusionRetriever"}
        if retriever_params is None
        else retriever_params
    )

    llm = LyzrLLMFactory.from_defaults(**llm_params)

    service_context = LyzrService.from_defaults(
        llm=llm,
        embed_model=embed_model,
        system_prompt=system_prompt,
        query_wrapper_prompt=query_wrapper_prompt,
        **service_context_params,
    )

    vector_store_index = LyzrVectorStoreIndex.from_defaults(
        **vector_store_params,
        documents=documents,
        service_context=service_context,
        similarity_top_k=10,
    )

    # retriever = LyzrRetriever.from_defaults(
    #     **retriever_params, base_index=vector_store_index
    # )

    # query_engine = RetrieverQueryEngine.from_args(retriever, query_engine_params)
    query_engine = vector_store_index.as_query_engine(similarity_top_k=10)

    return query_engine


def txt_rag(
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
    documents = read_txt_as_documents(
        input_dir=input_dir,
        input_files=input_files,
        exclude_hidden=exclude_hidden,
        filename_as_id=filename_as_id,
        recursive=recursive,
        required_exts=required_exts,
    )

    llm_params = (
        {
            "model": "gpt-4-0125-preview",
            "temperature": 0,            
        }
        if llm_params is None
        else llm_params
    )
    vector_store_params = (
        {"vector_store_type": "WeaviateVectorStore"}
        if vector_store_params is None
        else vector_store_params
    )
    service_context_params = (
        {} if service_context_params is None else service_context_params
    )
    query_engine_params = {} if query_engine_params is None else query_engine_params

    retriever_params = (
        {"retriever_type": "QueryFusionRetriever"}
        if retriever_params is None
        else retriever_params
    )

    llm = LyzrLLMFactory.from_defaults(**llm_params)

    service_context = LyzrService.from_defaults(
        llm=llm,
        embed_model=embed_model,
        system_prompt=system_prompt,
        query_wrapper_prompt=query_wrapper_prompt,
        **service_context_params,
    )

    vector_store_index = LyzrVectorStoreIndex.from_defaults(
        **vector_store_params,
        documents=documents,
        service_context=service_context,
        similarity_top_k=10,
    )

    # retriever = LyzrRetriever.from_defaults(
    #     **retriever_params, base_index=vector_store_index
    # )

    # query_engine = RetrieverQueryEngine.from_args(retriever, query_engine_params)
    query_engine = vector_store_index.as_query_engine(similarity_top_k=10)

    return query_engine


def docx_rag(
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
    documents = read_docx_as_documents(
        input_dir=input_dir,
        input_files=input_files,
        exclude_hidden=exclude_hidden,
        filename_as_id=filename_as_id,
        recursive=recursive,
        required_exts=required_exts,
    )

    llm_params = (
        {
            "model": "gpt-4-0125-preview",
            "temperature": 0,            
        }
        if llm_params is None
        else llm_params
    )
    vector_store_params = (
        {"vector_store_type": "WeaviateVectorStore"}
        if vector_store_params is None
        else vector_store_params
    )
    service_context_params = (
        {} if service_context_params is None else service_context_params
    )
    query_engine_params = {} if query_engine_params is None else query_engine_params

    retriever_params = (
        {"retriever_type": "QueryFusionRetriever"}
        if retriever_params is None
        else retriever_params
    )

    llm = LyzrLLMFactory.from_defaults(**llm_params)

    service_context = LyzrService.from_defaults(
        llm=llm,
        embed_model=embed_model,
        system_prompt=system_prompt,
        query_wrapper_prompt=query_wrapper_prompt,
        **service_context_params,
    )

    vector_store_index = LyzrVectorStoreIndex.from_defaults(
        **vector_store_params,
        documents=documents,
        service_context=service_context,
        similarity_top_k=10,
    )

    # retriever = LyzrRetriever.from_defaults(
    #     **retriever_params, base_index=vector_store_index
    # )

    # query_engine = RetrieverQueryEngine.from_args(retriever, query_engine_params)
    query_engine = vector_store_index.as_query_engine(similarity_top_k=10)

    return query_engine


def webpage_rag(
    url: str = None,
    system_prompt: str = None,
    query_wrapper_prompt: str = None,
    embed_model: Union[str, EmbedType] = "default",
    llm_params: dict = None,
    vector_store_params: dict = None,
    service_context_params: dict = None,
    query_engine_params: dict = None,
    retriever_params: dict = None,
) -> BaseQueryEngine:
    documents = read_webpage_as_documents(
        url=url,
    )

    llm_params = (
        {
            "model": "gpt-4-0125-preview",
            "temperature": 0,            
        }
        if llm_params is None
        else llm_params
    )
    vector_store_params = (
        {"vector_store_type": "WeaviateVectorStore"}
        if vector_store_params is None
        else vector_store_params
    )
    service_context_params = (
        {} if service_context_params is None else service_context_params
    )
    query_engine_params = {} if query_engine_params is None else query_engine_params

    retriever_params = (
        {"retriever_type": "QueryFusionRetriever"}
        if retriever_params is None
        else retriever_params
    )

    llm = LyzrLLMFactory.from_defaults(**llm_params)

    service_context = LyzrService.from_defaults(
        llm=llm,
        embed_model=embed_model,
        system_prompt=system_prompt,
        query_wrapper_prompt=query_wrapper_prompt,
        **service_context_params,
    )

    vector_store_index = LyzrVectorStoreIndex.from_defaults(
        **vector_store_params,
        documents=documents,
        service_context=service_context,
        similarity_top_k=10,
    )

    # retriever = LyzrRetriever.from_defaults(
    #     **retriever_params, base_index=vector_store_index
    # )

    # query_engine = RetrieverQueryEngine.from_args(retriever, query_engine_params)
    query_engine = vector_store_index.as_query_engine(similarity_top_k=10)

    return query_engine


def website_rag(
    url: str = None,
    system_prompt: str = None,
    query_wrapper_prompt: str = None,
    embed_model: Union[str, EmbedType] = "default",
    llm_params: dict = None,
    vector_store_params: dict = None,
    service_context_params: dict = None,
    query_engine_params: dict = None,
    retriever_params: dict = None,
) -> BaseQueryEngine:
    documents = read_website_as_documents(
        url=url,
    )

    llm_params = (
        {
            "model": "gpt-4-0125-preview",
            "temperature": 0,            
        }
        if llm_params is None
        else llm_params
    )
    vector_store_params = (
        {"vector_store_type": "WeaviateVectorStore"}
        if vector_store_params is None
        else vector_store_params
    )
    service_context_params = (
        {} if service_context_params is None else service_context_params
    )
    query_engine_params = {} if query_engine_params is None else query_engine_params

    retriever_params = (
        {"retriever_type": "QueryFusionRetriever"}
        if retriever_params is None
        else retriever_params
    )

    llm = LyzrLLMFactory.from_defaults(**llm_params)

    service_context = LyzrService.from_defaults(
        llm=llm,
        embed_model=embed_model,
        system_prompt=system_prompt,
        query_wrapper_prompt=query_wrapper_prompt,
        **service_context_params,
    )

    vector_store_index = LyzrVectorStoreIndex.from_defaults(
        **vector_store_params,
        documents=documents,
        service_context=service_context,
        similarity_top_k=10,
    )

    # retriever = LyzrRetriever.from_defaults(
    #     **retriever_params, base_index=vector_store_index
    # )

    # query_engine = RetrieverQueryEngine.from_args(retriever, query_engine_params)
    query_engine = vector_store_index.as_query_engine(similarity_top_k=10)

    return query_engine


def youtube_rag(
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
    documents = read_youtube_as_documents(
        urls=urls,
    )

    llm_params = (
        {
            "model": "gpt-4-0125-preview",
            "temperature": 0,            
        }
        if llm_params is None
        else llm_params
    )
    vector_store_params = (
        {"vector_store_type": "WeaviateVectorStore"}
        if vector_store_params is None
        else vector_store_params
    )
    service_context_params = (
        {} if service_context_params is None else service_context_params
    )
    query_engine_params = {} if query_engine_params is None else query_engine_params

    retriever_params = (
        {"retriever_type": "QueryFusionRetriever"}
        if retriever_params is None
        else retriever_params
    )

    llm = LyzrLLMFactory.from_defaults(**llm_params)

    service_context = LyzrService.from_defaults(
        llm=llm,
        embed_model=embed_model,
        system_prompt=system_prompt,
        query_wrapper_prompt=query_wrapper_prompt,
        **service_context_params,
    )

    vector_store_index = LyzrVectorStoreIndex.from_defaults(
        **vector_store_params,
        documents=documents,
        service_context=service_context,
        similarity_top_k=10,
    )

    # retriever = LyzrRetriever.from_defaults(
    #     **retriever_params, base_index=vector_store_index
    # )

    # query_engine = RetrieverQueryEngine.from_args(retriever, query_engine_params)
    query_engine = vector_store_index.as_query_engine(similarity_top_k=10)

    return query_engine
