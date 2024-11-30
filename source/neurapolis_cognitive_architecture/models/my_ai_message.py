from typing import Optional

from neurapolis_cognitive_architecture.enums import MessageRole
from neurapolis_retriever import RetrievedFile
from pydantic import Field

from .message import Message


class MyAiMessage(Message):
    role: MessageRole = Field(default=MessageRole.AI, frozen=True)
    retrieved_files: Optional[list[RetrievedFile]] = Field(default=None)

    def convert_to_data(self) -> dict:
        return {
            "id": self.id,
            "role": self.role.value,
            "content": self.content,
            "retrievedFiles": (
                None
                if self.retrieved_files is None
                else [x_file.convert_to_data() for x_file in self.retrieved_files]
            ),
        }
