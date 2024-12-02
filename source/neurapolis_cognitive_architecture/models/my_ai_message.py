from typing import Optional

from neurapolis_cognitive_architecture.enums import MessageRole
from neurapolis_retriever import RetrievedFile
from pydantic import Field

from .message import Message


class MyAiMessage(Message):
    role: MessageRole = Field(MessageRole.AI, frozen=True)
    retrieved_files: Optional[list[RetrievedFile]] = Field(None)

    def convert_to_data(self) -> dict:
        return {
            "id": self.id,
            "role": self.role.value,
            "content": self.content,
            "retrievedFiles": (
                None
                if self.retrieved_files is None
                else [
                    x_retrieved_file.convert_to_data()
                    for x_retrieved_file in self.retrieved_files
                ]
            ),
        }

    @classmethod
    def create_from_data(cls, data: dict) -> "MyAiMessage":
        raise NotImplementedError()
