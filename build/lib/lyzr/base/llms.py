import os
from openai import OpenAI
from typing import Optional, Literal
from lyzr.base.prompt import get_prompt_text
from lyzr.base.errors import MissingValueError, InvalidValueError


class LLM:
    def __init__(
        self,
        api_key: str,
        model_type: Optional[str] = None,
        model_name: Optional[str] = None,
        model_prompts: Optional[list[dict]] = None,
        voice: Optional[
            Literal["echo", "alloy", "fable", "onyx", "nova", "shimmer"]
        ] = None,  # Add support for specifying the voice model
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

    def set_messages(
        self,
        model_prompts: Optional[list[dict]] = None,
        messages: Optional[list[dict]] = None,
    ):
        if model_prompts is None and messages is None and self.messages is None:
            raise ValueError("Please set a value for the prompt")

        if messages is not None:
            self.messages = messages
            return self

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
            if "model_prompts" in kwargs:
                self.set_messages(model_prompts=kwargs["model_prompts"])
            elif "messages" in kwargs:
                self.set_messages(messages=[kwargs["messages"]])
            elif "input" in kwargs:
                self.input = kwargs["input"]
            elif "audiofile" in kwargs:
                self.audiofile = kwargs["audiofile"]
            else:
                raise MissingValueError(["model_prompts", "messages"])

        params = self.__dict__.copy()
        params.update(kwargs)
        for param in [
            "api_key",
            "model_prompts",
            "model_type",
            "model_name",
            "messages",
            "voice",
            "input",
            "audiofile",
        ]:
            if param in params:
                del params[param]

        # Instantiate the OpenAI client
        client = OpenAI(api_key=self.api_key)

        if self.model_type == "openai":
            # Check for Text-to-Speech models
            if self.model_name in ["tts-1", "tts-1-hd"]:
                if self.input is None:
                    raise MissingValueError("input")
                # Use 'voice' and any other options required by OpenAI's TTS endpoint
                response = client.audio.speech.create(
                    model=self.model_name,
                    voice=self.voice,
                    input=self.input,
                    **params,
                )
                return response

            # Check for transcription models like "whisper"
            elif self.model_name.startswith("whisper"):
                if self.audiofile is None:
                    raise MissingValueError("audiofile")
                # The transcription API may require a file-like object or binary data
                response = client.audio.transcriptions.create(
                    model=self.model_name,
                    file=self.audiofile,
                    **params,
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
    model_type: Optional[str] = None,
    model_name: Optional[str] = None,
    **kwargs,
) -> LLM:
    return LLM(
        api_key=api_key or os.getenv("API_KEY"),
        model_type=model_type or os.getenv("MODEL_TYPE") or "openai",
        model_name=model_name or os.getenv("MODEL_NAME") or "gpt-3.5-turbo",
        **kwargs,
    )
