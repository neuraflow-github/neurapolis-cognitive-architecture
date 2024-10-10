from typing import Optional

from neurapolis_retriever.models.file_hit import FileHit

from .message import Message
from .message_role import MessageRole


class AiMessage(Message):
    file_hits: Optional[list[FileHit]] = None

    def __init__(
        self,
        id: str,
        content: str,
        file_hits: Optional[list[FileHit]] = None,
    ):
        super().__init__(id, MessageRole.AI, content)
        self.file_hits = file_hits

    def to_dto(self):
        return {
            "id": self.id,
            "role": self.role.value,
            "content": self.content,
            "file_hits": (
                None
                if self.file_hits is None
                else [x_file.to_dto() for x_file in self.file_hits]
            ),
        }

    @classmethod
    def from_dto(cls, ai_message_dto: dict) -> "AiMessage":
        raise Exception("NOT_IMPLEMENTED")
