import os
from openai import OpenAI
from typing import Optional, Literal
from dataanalyzr.enterprise_data_analyzr.base.prompt import get_prompt_text
from dataanalyzr.enterprise_data_analyzr.base.errors import MissingValueError

class LLM:
    def __init__(
        self,
        api_key: str,
        model_type: Optional[str] = None,
        model_name: Optional[str] = None,
        model_prompts: Optional[list[dict]] = None,
        voice: Optional[str] = None,  # Add support for specifying the voice model
        **kwargs,
    ):
        self.api_key = api_key
        self.model_type = model_type or "openai"
        self.model_name = model_name or "gpt-3.5-turbo"
        self.messages = None
        self.voice = voice or "nova"  # Default voice for TTS
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
            raise ValueError("Please provide an API key or set the API_KEY environment variable.")

        if self.messages is None and "model_prompts" in kwargs:
            self.set_messages(kwargs["model_prompts"])
            del kwargs["model_prompts"]

        params = self.__dict__.copy()
        for param in ["api_key", "model_prompts", "model_type", "model_name"]:
            if param in params:
                del params[param]
        params.update(kwargs)

        # Instantiate the OpenAI client
        client = OpenAI(api_key=self.api_key)

        if self.model_type == "openai":
            # Check for Text-to-Speech models
            if self.model_name in ["tts-1", "tts-1-hd"]:
                # Use 'voice' and any other options required by OpenAI's TTS endpoint
                response = client.audio.speech.create(
                    model=self.model_name,
                    voice=self.voice,
                    text=params.get("text"),
                )
                return response

            # Check for transcription models like "whisper"
            elif self.model_name.startswith("whisper"):
                # The transcription API may require a file-like object or binary data
                if "file" not in params:
                    raise MissingValueError("file")
                response = client.audio.transcriptions.create(
                    model=self.model_name,
                    file=params["file"],
                )
                return response

            # Else, handle chat completions
            else:
                completion = client.chat.completions.create(
                    model=self.model_name,
                    messages=self.messages,
                    **params,
                )
                return completion

def get_model(
    api_key: Optional[str] = None,
    model_type: Literal["openai"] = None,
    model_name: Literal["gpt-3.5-turbo", "gpt-4"] = None,
) -> LLM:
    return LLM(
        api_key=api_key or os.getenv("API_KEY"),
        model_type=model_type or os.getenv("MODEL_TYPE") or "openai",
        model_name=model_name or os.getenv("MODEL_NAME") or "gpt-3.5-turbo",
    )
