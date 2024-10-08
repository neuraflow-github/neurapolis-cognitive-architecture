from typing import Optional

from neurapolis_common.models.dto import Dto
from neurapolis_retriever.models.file_info import FileInfo

from .message_role import MessageRole


class Message(Dto["Message"]):
    id: str
    role: MessageRole
    content: str
    date_filter: Optional[dict] = None
    files: Optional[list[FileInfo]] = None

    def __init__(
        self,
        id: str,
        role: MessageRole,
        content: str,
        date_filter: Optional[dict] = None,
        files: Optional[list[FileInfo]] = None,
    ):
        self.id = id
        self.role = role
        self.content = content
        self.date_filter = date_filter
        self.files = files

    def to_dto(self):
        return {
            "id": self.id,
            "role": self.role.value,
            "content": self.content,
            "date_filter": self.date_filter,  # HACK
            "files": (
                None
                if self.files is None
                else [x_file.to_dto() for x_file in self.files]
            ),
        }

    @classmethod
    def from_dto(cls, message_dto: dict) -> "Message":
        raise Exception("NOT_IMPLEMENTED")
