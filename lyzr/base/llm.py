# standard library imports
import logging
from typing import Union, Literal, Sequence

# third-party imports
from llama_index.llms import LiteLLM
from llama_index.llms.base import LLM
from llama_index.llms.base import ChatMessage as LlamaChatMessage

# local imports
from lyzr.base.base import ChatMessage, ChatResponse
from lyzr.base.errors import ImproperUsageError

DEFAULT_LLM = "gpt-4-0125-preview"


class LyzrLLMFactory:
    @staticmethod
    def from_defaults(model: str = DEFAULT_LLM, **kwargs) -> LLM:
        # model_type -> api_type
        # model_name -> model
        # model_prompts -> Sequence[ChatMessage]
        return LiteLLM(model=model, **kwargs)


class LiteLLM(LiteLLM):

    @property
    def _tts_kwargs(self) -> dict:
        return {
            "voice": self.additional_kwargs.pop("voice", None),
        }

    @property
    def _model_type(self) -> Literal["chat", "tts", "stt"]:
        model_name = self._get_model_name()
        if model_name.startswith("tts"):
            return "tts"
        elif model_name.startswith("whisper"):
            return "stt"
        return "chat"

    def set_model_kwargs(
        self, model_kwargs: dict, force: Union[bool, dict] = True
    ) -> dict:
        if isinstance(force, bool):
            force = {arg: force for arg in model_kwargs}
        all_kwargs = self._get_all_kwargs()
        for arg in model_kwargs:
            if (not force.get(arg, True)) and (arg in all_kwargs):
                continue
            if arg in ["temperature", "max_tokens"]:
                self.__dict__[arg] = model_kwargs[arg]
            self.additional_kwargs[arg] = model_kwargs[arg]

    def set_messages(self, messages: Sequence[ChatMessage]):
        self.messages = messages

    def run(self, **kwargs):
        if self._model_type == "chat":
            return self.chat_complete(
                messages=kwargs.pop("messages", self.__dict__.get("messages", None)),
                stream=kwargs.pop("stream", False),
                logger=self.additional_kwargs.pop("logger", kwargs.pop("logger", None)),
                **kwargs,
            )
        elif self._model_type == "tts":
            return self.tts(
                tts_input=kwargs.pop("input", None),
                voice=self._tts_kwargs.pop("voice", kwargs.pop("voice", None)),
                **kwargs,
            )
        elif self._model_type == "stt":
            return self.stt(audiofile=kwargs.pop("audiofile", None), **kwargs)
        return None

    def chat_complete(
        self,
        messages: Sequence[ChatMessage],
        stream: bool = False,
        logger: logging.Logger = None,
        **kwargs,
    ):
        if not messages:
            raise ImproperUsageError("Please provide messages for chat.")
        llama_messages = [
            LlamaChatMessage(role=msg.role.value, content=msg.content)
            for msg in messages
        ]
        if stream:
            response = self._stream_chat(messages=llama_messages, **kwargs)
            return response  # TODO: Convert response to ChatMessage
        else:
            response = self._chat(messages=llama_messages, **kwargs)
            return ChatResponse(
                message=ChatMessage(
                    role=response.message.role, content=response.message.content
                ),
                raw=response.raw,
                delta=response.delta,
                additional_kwargs=response.additional_kwargs,
            )

    def tts(
        self,
        tts_input,
        voice: Literal["echo", "alloy", "fable", "onyx", "nova", "shimmer"],
        **kwargs,
    ):
        if tts_input is None:
            raise ImproperUsageError("Please provide an input text for text-to-speech.")
        if voice is None:
            raise ImproperUsageError("Please provide a voice for text-to-speech.")
        from openai import OpenAI

        client = OpenAI(api_key=self.additional_kwargs.get("api_key", None))
        response = client.audio.speech.create(
            model=self._get_model_name(),
            voice=voice,
            input=tts_input,
            **self._get_all_kwargs(**kwargs),
        )
        return response

    def stt(self, audiofile, **kwargs):
        if audiofile is None:
            raise ImproperUsageError("Please provide an audio file for speech-to-text.")
        from openai import OpenAI

        client = OpenAI(api_key=self.additional_kwargs.get("api_key", None))
        response = client.audio.transcriptions.create(
            model=self._get_model_name(),
            file=audiofile,
            **self._get_all_kwargs(**kwargs),
        )
        return response
