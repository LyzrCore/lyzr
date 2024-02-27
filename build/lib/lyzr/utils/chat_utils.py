from typing import Union, Optional, List

from llama_index.chat_engine.types import BaseChatEngine, ChatMode
from llama_index.embeddings.utils import EmbedType
from llama_index.chat_engine import ContextChatEngine
from llama_index.memory import ChatMemoryBuffer

from lyzr.base.llm import LyzrLLMFactory
from lyzr.base.service import LyzrService
from lyzr.base.vector_store import LyzrVectorStoreIndex
from lyzr.base.retrievers import LyzrRetriever

from lyzr.utils.document_reading import (
    read_pdf_as_documents,
    read_docx_as_documents,
    read_txt_as_documents,
    read_website_as_documents,
    read_webpage_as_documents,
    read_youtube_as_documents,
)


def pdf_chat_(
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
    chat_engine_params: dict = None,
    retriever_params: dict = None,
) -> BaseChatEngine:
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
    chat_engine_params = {} if chat_engine_params is None else chat_engine_params

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
        **vector_store_params, documents=documents, service_context=service_context
    )

    retriever = LyzrRetriever.from_defaults(
        **retriever_params, base_index=vector_store_index
    )

    memory = ChatMemoryBuffer.from_defaults(token_limit=4000)

    chat_engine = ContextChatEngine(
        llm=llm,
        memory=memory,
        retriever=retriever,
        prefix_messages=list(),
        **chat_engine_params,
    )

    return chat_engine


def txt_chat_(
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
    chat_engine_params: dict = None,
    retriever_params: dict = None,
) -> BaseChatEngine:
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
    chat_engine_params = {} if chat_engine_params is None else chat_engine_params

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
        **vector_store_params, documents=documents, service_context=service_context
    )

    retriever = LyzrRetriever.from_defaults(
        **retriever_params, base_index=vector_store_index
    )

    memory = ChatMemoryBuffer.from_defaults(token_limit=4000)

    chat_engine = ContextChatEngine(
        llm=llm,
        memory=memory,
        retriever=retriever,
        prefix_messages=list(),
        **chat_engine_params,
    )

    return chat_engine


def docx_chat_(
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
    chat_engine_params: dict = None,
    retriever_params: dict = None,
) -> BaseChatEngine:
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
    chat_engine_params = {} if chat_engine_params is None else chat_engine_params

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
        **vector_store_params, documents=documents, service_context=service_context
    )

    retriever = LyzrRetriever.from_defaults(
        **retriever_params, base_index=vector_store_index
    )

    memory = ChatMemoryBuffer.from_defaults(token_limit=4000)

    chat_engine = ContextChatEngine(
        llm=llm,
        memory=memory,
        retriever=retriever,
        prefix_messages=list(),
        **chat_engine_params,
    )

    return chat_engine


def webpage_chat_(
    url: str = None,
    system_prompt: str = None,
    query_wrapper_prompt: str = None,
    embed_model: Union[str, EmbedType] = "default",
    llm_params: dict = None,
    vector_store_params: dict = None,
    service_context_params: dict = None,
    chat_engine_params: dict = None,
    retriever_params: dict = None,
) -> BaseChatEngine:
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
    chat_engine_params = {} if chat_engine_params is None else chat_engine_params

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
        **vector_store_params, documents=documents, service_context=service_context
    )

    retriever = LyzrRetriever.from_defaults(
        **retriever_params, base_index=vector_store_index
    )

    memory = ChatMemoryBuffer.from_defaults(token_limit=4000)

    chat_engine = ContextChatEngine(
        llm=llm,
        memory=memory,
        retriever=retriever,
        prefix_messages=list(),
        **chat_engine_params,
    )

    return chat_engine


def website_chat_(
    url: str = None,
    system_prompt: str = None,
    query_wrapper_prompt: str = None,
    embed_model: Union[str, EmbedType] = "default",
    llm_params: dict = None,
    vector_store_params: dict = None,
    service_context_params: dict = None,
    chat_engine_params: dict = None,
    retriever_params: dict = None,
) -> BaseChatEngine:
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
    chat_engine_params = {} if chat_engine_params is None else chat_engine_params

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
        **vector_store_params, documents=documents, service_context=service_context
    )

    retriever = LyzrRetriever.from_defaults(
        **retriever_params, base_index=vector_store_index
    )

    memory = ChatMemoryBuffer.from_defaults(token_limit=4000)

    chat_engine = ContextChatEngine(
        llm=llm,
        memory=memory,
        retriever=retriever,
        prefix_messages=list(),
        **chat_engine_params,
    )

    return chat_engine


def youtube_chat_(
    urls: List[str] = None,
    system_prompt: str = None,
    query_wrapper_prompt: str = None,
    embed_model: Union[str, EmbedType] = "default",
    llm_params: dict = None,
    vector_store_params: dict = None,
    service_context_params: dict = None,
    chat_engine_params: dict = None,
    retriever_params: dict = None,
) -> BaseChatEngine:
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
    chat_engine_params = {} if chat_engine_params is None else chat_engine_params

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
        **vector_store_params, documents=documents, service_context=service_context
    )

    retriever = LyzrRetriever.from_defaults(
        **retriever_params, base_index=vector_store_index
    )

    memory = ChatMemoryBuffer.from_defaults(token_limit=4000)

    chat_engine = ContextChatEngine(
        llm=llm,
        memory=memory,
        retriever=retriever,
        prefix_messages=list(),
        **chat_engine_params,
    )

    return chat_engine
