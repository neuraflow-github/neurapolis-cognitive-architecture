from typing import Optional

from neurapolis_retriever.models.file_info import FileInfo
from pydantic import BaseModel, Field

from .message_role import MessageRole


class Message(BaseModel):
    id: str = Field()
    role: MessageRole = Field()
    content: str = Field()
    date_filter: Optional[dict] = Field(default=None)
    files: Optional[list[FileInfo]] = Field(default=None)

    def to_dto(self):
        return {
            "id": self.id,
            "role": self.role.value,
            "content": self.content,
            "date_filter": self.date_filter,
            "files": None if self.files else [x_file.to_dto() for x_file in self.files],
        }
