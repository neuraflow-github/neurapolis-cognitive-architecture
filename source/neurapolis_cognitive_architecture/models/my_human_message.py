from typing import Optional

from langchain_core.messages import HumanMessage
from neurapolis_cognitive_architecture.enums import MessageRole
from neurapolis_retriever import DateFilter, QualityPreset

from .message import Message


class MyHumanMessage(Message, HumanMessage):
    date_filter: Optional[DateFilter]
    quality_preset: QualityPreset

    def __init__(
        self,
        id: str,
        content: str,
        date_filter: Optional[DateFilter],
        quality_preset: QualityPreset,
    ):
        super().__init__(id, MessageRole.USER, content)
        self.date_filter = date_filter
        self.quality_preset = quality_preset

    def convert_to_data(self):
        return {
            "id": self.id,
            "role": self.role.value,
            "content": self.content,
            "dateFilter": (
                None if self.date_filter is None else self.date_filter.convert_to_data()
            ),
            "qualityPreset": self.quality_preset.value,
        }

    @classmethod
    def create_from_data(cls, user_message_dto: dict) -> "MyHumanMessage":
        return cls(
            id=user_message_dto["id"],
            content=user_message_dto["content"],
            date_filter=(
                None
                if user_message_dto.get("dateFilter") is None
                else DateFilter.create_from_data(user_message_dto["dateFilter"])
            ),
            quality_preset=QualityPreset(user_message_dto["qualityPreset"]),
        )
