import os
from openai import OpenAI
from typing import Optional
from lyzr.base.prompt import Prompt


class LLM:
    def __init__(
        self,
        api_key: str,
        model_type: Optional[str] = "openai",
        model_name: Optional[str] = "gpt-3.5-turbo",
        prompt_name: Optional[str] = None,
        prompt: Optional[Prompt] = None,
        **kwargs,
    ):
        self.api_key = api_key
        self.model_type = model_type
        self.model_name = model_name
        if prompt_name is not None:
            self.prompt = Prompt(prompt_name)
        elif prompt is not None:
            self.prompt = prompt
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

    def set_prompt(
        self,
        prompt_name: Optional[str] = None,
        prompt_text: Optional[str] = None,
        **kwargs,
    ):
        if prompt_name is None and self.prompt is None:
            raise ValueError("Please set a value for the prompt")

        if self.prompt is None:
            self.prompt = Prompt(prompt_name, prompt_text)

        self.prompt.text = self.prompt.format(**kwargs)

    def run(self, **kwargs):
        if self.api_key is None:
            raise ValueError(
                "Please provide an API key or set the API_KEY environment variable."
            )

        if "prompt" in kwargs:
            self.set_prompt(kwargs.pop("prompt"), **kwargs)

        empty_variables = self.prompt.get_variables()
        if empty_variables != []:
            raise ValueError(
                f"Please provide values for the following variables: {empty_variables}"
            )
        messages = [{"role": "system", "content": self.prompt.text}]

        params = self.__dict__.copy()
        for param in ["api_key", "model_type", "model_name", "prompt"]:
            del params[param]
        params.update(kwargs)

        if self.model_type == "openai":
            client = OpenAI(api_key=self.api_key)
            completion = client.completions.create(
                model=self.model_name,
                messages=messages,
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
