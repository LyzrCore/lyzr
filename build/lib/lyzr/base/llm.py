from llama_index.llms import LiteLLM
from llama_index.llms.base import LLM


class LyzrLLMFactory:

    def __init__(self) -> None:
        None
    @staticmethod
    def from_defaults(model: str = "gpt-4-0125-preview", **kwargs) -> LLM:
        return LiteLLM(model=model, **kwargs)
