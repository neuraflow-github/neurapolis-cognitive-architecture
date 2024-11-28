from typing import Optional

from neurapolis_cognitive_architecture.enums import MessageRole
from neurapolis_retriever import FileHit

from .message import Message


class MyAiMessage(Message):
    file_hits: Optional[list[FileHit]] = None

    def __init__(
        self,
        id: str,
        content: str,
        file_hits: Optional[list[FileHit]] = None,
    ):
        super().__init__(id, MessageRole.AI, content)
        self.file_hits = file_hits

    def convert_to_data(self):
        return {
            "id": self.id,
            "role": self.role.value,
            "content": self.content,
            "fileHits": (
                None
                if self.file_hits is None
                else [x_file.convert_to_data() for x_file in self.file_hits]
            ),
        }

    @classmethod
    def create_from_data(cls, ai_message_dto: dict) -> "MyAiMessage":
        raise Exception("NOT_IMPLEMENTED")
