from typing import Optional

from neurapolis_retriever.models.date_filter import DateFilter

from .message import Message
from .message_role import MessageRole


class UserMessage(Message):
    date_filter: Optional[dict]
    quality_preset: str

    def __init__(
        self, id: str, content: str, date_filter: Optional[dict], quality_preset: str
    ):
        super().__init__(id, MessageRole.USER, content)
        self.date_filter = date_filter
        self.quality_preset = quality_preset

    def to_dto(self):
        raise Exception("NOT_IMPLEMENTED")

    @classmethod
    def from_dto(cls, user_message_dto: dict) -> "UserMessage":
        return cls(
            id=user_message_dto["id"],
            content=user_message_dto["content"],
            date_filter=(
                None
                if user_message_dto["date_filter"] is None
                else DateFilter.from_dto(user_message_dto["date_filter"])
            ),
            quality_preset=user_message_dto["quality_preset"],
        )
