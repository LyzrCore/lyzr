"""
Prompt management classes for Lyzr.
"""

from typing import Literal
from lyzr.base.errors import PromptError
from lyzr.base.prompt_texts import DATA_ANALYZR_PROMPTS
from lyzr.base.base import ChatMessage, UserMessage, SystemMessage, MessageRole


class PromptRole:
    """A descriptor class to manage and validate the role of a prompt."""

    def __init__(self, allowed_names: list):
        self.allowed_names = allowed_names

    def __get__(self, instance, _):
        return instance._name

    def __set__(self, instance, value):
        if value not in self.allowed_names:
            raise PromptError(f"Prompt type must be one of {self.allowed_names}")
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
        """
        A factory class for creating and managing prompt messages.

        Attributes:
            prompt_type (MessageRole): The type of the prompt, either "user" or "system".
            sections (dict): A dictionary containing the sections of the prompt.
            sections_to_use (list): A list of sections to be used in the prompt.

        Methods:
            __init__(name: str, prompt_type: Literal["user", "system"], use_sections: list = None) -> None:
                Initializes the LyzrPromptFactory with a given name, prompt type, and optional sections to use.

            select_sections(use_sections: list = None) -> None:
                Selects the sections to be used in the prompt.

            get_message(use_sections: list = None, **kwargs) -> ChatMessage:
                Constructs and returns a ChatMessage object based on the selected sections and additional keyword arguments.
        """
        self.prompt_type = prompt_type
        if name.lower() not in DATA_ANALYZR_PROMPTS:
            raise PromptError(f"Prompt name {name} not found.")
        self.sections = DATA_ANALYZR_PROMPTS[name.lower()][self.prompt_type.value]
        self.sections_to_use = use_sections or []

    def select_sections(self, use_sections: list = None) -> None:
        """
        Selects the sections to be used in the prompt.

        Args:
            use_sections (list, optional): A list of sections to be used. Defaults to None.
        """
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
        """
        Constructs and returns a ChatMessage object based on the selected sections and additional keyword arguments.

        Args:
            use_sections (list, optional): A list of sections to be used. Defaults to None.
            **kwargs: Additional keyword arguments to format the sections.

        Returns:
            ChatMessage: The constructed chat message.
        """
        self.select_sections(use_sections)
        if self.sections_to_use == []:
            self.sections_to_use = list(self.sections.keys())
        message = (
            UserMessage() if self.prompt_type == MessageRole.USER else SystemMessage()
        )
        message.content = ""
        for section in self.sections_to_use:
            message.content += self.sections[section].format(**kwargs)
        message.content = message.content.strip()
        return message
