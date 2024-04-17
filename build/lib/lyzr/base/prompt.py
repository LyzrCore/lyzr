import json
from typing import Literal
from lyzr.base.prompt_texts import PROMPT_TEXTS
from lyzr.base.base import ChatMessage, UserMessage, SystemMessage, MessageRole


class PromptRole:
    def __init__(self, allowed_names: list):
        self.allowed_names = allowed_names

    def __get__(self, instance, _):
        return instance._name

    def __set__(self, instance, value):
        if value not in self.allowed_names:
            raise ValueError(f"Prompt type must be one of {self.allowed_names}")
        instance._name = MessageRole(value)


class LyzrPromptFactory:
    """Lyzr prompt factory."""

    prompt_type: MessageRole = PromptRole(["user", "system"])
    sections: dict
    sections_to_use: list

    def __init__(
        self,
        name: str,
        prompt_type: Literal["user", "system"],
        use_sections: list = None,
    ) -> None:
        self.prompt_type = prompt_type
        if name.lower() not in PROMPT_TEXTS:
            raise ValueError(f"Prompt name {name} not found.")
        self.sections = PROMPT_TEXTS[name.lower()][self.prompt_type.value]
        self.sections_to_use = use_sections or []

    def select_sections(self, use_sections: list = None) -> None:
        self.sections_to_use = (
            use_sections
            if (
                (use_sections is not None)
                and isinstance(use_sections, list)
                and (len(use_sections) > 0)
            )
            else self.sections_to_use
        )

    def get_message(self, use_sections: list = None, **kwargs) -> ChatMessage:
        self.select_sections(use_sections)
        if self.sections_to_use == []:
            self.sections_to_use = list(self.sections.keys())
        message = (
            UserMessage() if self.prompt_type == MessageRole.USER else SystemMessage()
        )
        message.content = ""
        for section in self.sections_to_use:
            message.content += self.sections[section].format(**kwargs)
        return message
