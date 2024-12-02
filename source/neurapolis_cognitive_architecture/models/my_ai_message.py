from typing import Optional

from neurapolis_cognitive_architecture.enums import MessageRole
from neurapolis_retriever import Reference
from pydantic import Field

from .message import Message


class MyAiMessage(Message):
    role: MessageRole = Field(MessageRole.AI, frozen=True)
    references: Optional[list[Reference]] = Field(None)

    def convert_to_data(self) -> dict:
        return {
            "id": self.id,
            "role": self.role.value,
            "content": self.content,
            "references": (
                None
                if self.references is None
                else [x_reference.convert_to_data() for x_reference in self.references]
            ),
        }

    @classmethod
    def create_from_data(cls, data: dict) -> "MyAiMessage":
        raise NotImplementedError()
