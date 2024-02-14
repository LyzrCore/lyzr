import logging
from typing import Optional, Union

from llama_index import ServiceContext
from llama_index.callbacks import CallbackManager
from llama_index.embeddings.utils import EmbedType
from llama_index.llms.utils import LLMType
from llama_index.prompts import PromptTemplate
from llama_index.prompts.base import BasePromptTemplate
from llama_index.node_parser import (
    SimpleNodeParser,
)

logger = logging.getLogger(__name__)


class LyzrService:
    @staticmethod
    def from_defaults(
        llm: Optional[LLMType] = "default",
        embed_model: Optional[EmbedType] = "default",
        system_prompt: str = None,
        query_wrapper_prompt: Union[str, BasePromptTemplate] = None,
        **kwargs,
    ) -> ServiceContext:
        if isinstance(query_wrapper_prompt, str):
            query_wrapper_prompt = PromptTemplate(template=query_wrapper_prompt)

        callback_manager: CallbackManager = kwargs.get(
            "callback_manager", CallbackManager()
        )

        service_context = ServiceContext.from_defaults(
            llm=llm,
            embed_model=embed_model,
            system_prompt=system_prompt,
            query_wrapper_prompt=query_wrapper_prompt,
            callback_manager=callback_manager,
            **kwargs,
        )

        return service_context
