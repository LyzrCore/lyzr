from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class MessageRole(str, Enum):
    """Message role enum."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    FUNCTION = "function"
    TOOL = "tool"


class ChatMessage(BaseModel):
    """Chat message."""

    role: MessageRole = MessageRole.USER
    content: Optional[str] = ""
    additional_kwargs: dict = Field(default_factory=dict)

    def __str__(self) -> str:
        return f"{self.role.value}: {self.content}"


class UserMessage(ChatMessage):
    """User message."""

    role: MessageRole = MessageRole.USER


class AssistantMessage(ChatMessage):
    """Assistant message."""

    role: MessageRole = MessageRole.ASSISTANT


class SystemMessage(ChatMessage):
    """System message."""

    role: MessageRole = MessageRole.SYSTEM


class ChatResponse(BaseModel):
    """Chat response."""

    message: ChatMessage
    raw: Optional[dict] = None
    delta: Optional[str] = None
    additional_kwargs: dict = {}

    def __str__(self) -> str:
        return str(self.message)
