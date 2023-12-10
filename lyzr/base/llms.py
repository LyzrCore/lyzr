import os
from openai import OpenAI
from typing import Optional, Union
from lyzr.base.prompt import Prompt, get_prompt_text
from lyzr.base.errors import MissingValueError, InvalidValueError


class LLM:
    def __init__(
        self,
        api_key: str,
        model_type: Optional[str] = None,
        model_name: Optional[str] = None,
        model_prompts: Optional[list[dict]] = None,
        **kwargs,
    ):
        self.api_key = api_key
        self.model_type = model_type
        self.model_name = model_name
        self.messages = None
        if model_prompts is not None:
            self.set_messages(model_prompts)
        for param in kwargs:
            setattr(self, param, kwargs[param])
        # llm_params = {
        #     "temperature": 0,
        #     "top_p": 1,
        #     "presence_penalty": 0,
        #     "frequency_penalty": 0,
        #     "stop": None,
        #     "n": 1,
        #     "stream": False,
        #     "functions": None,
        #     "function_call": None,
        #     "logit_bias": None,
        #     "best_of": 1,
        #     "echo": False,
        #     "logprobs": None,
        #     "suffix": None,
        # }
        # for param in llm_params:
        #     setattr(self, param, kwargs.get(param, llm_params[param]))

    def set_messages(
        self,
        model_prompts: Optional[list[dict]] = None,
    ):
        if model_prompts is None and self.prompt is None:
            raise ValueError("Please set a value for the prompt")

        if model_prompts is None:
            return None

        messages = []
        for prompt in model_prompts:
            messages.append(
                {"role": prompt["role"], "content": get_prompt_text(prompt)}
            )
        self.messages = messages
        return self

    def run(self, **kwargs):
        if self.api_key is None:
            raise ValueError(
                "Please provide an API key or set the API_KEY environment variable."
            )

        if self.messages is None:
            if "model_prompts" not in kwargs:
                raise MissingValueError(["model_prompts"])
            self.set_messages(kwargs["model_prompts"])
            del kwargs["model_prompts"]

        params = self.__dict__.copy()
        for param in ["api_key", "model_prompts", "model_type", "model_name"]:
            if param in params:
                del params[param]
        params.update(kwargs)

        if self.model_type == "openai":
            completion = OpenAI(api_key=self.api_key).chat.completions.create(
                model=self.model_name,
                **params,
            )
            return completion


def get_model(
    api_key: Optional[str] = None,
    model_type: Optional[str] = None,
    model_name: Optional[str] = None,
) -> LLM:
    return LLM(
        api_key=api_key or os.getenv("API_KEY"),
        model_type=model_type or os.getenv("MODEL_TYPE") or "openai",
        model_name=model_name or os.getenv("MODEL_NAME") or "gpt-3.5-turbo",
    )
