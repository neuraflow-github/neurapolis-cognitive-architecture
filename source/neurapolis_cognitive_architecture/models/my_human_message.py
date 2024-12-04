from typing import Optional

from langchain_core.messages import HumanMessage
from neurapolis_cognitive_architecture.enums import MessageRole
from neurapolis_retriever import DateFilter, QualityPreset
from pydantic import Field

from .message import Message


class MyHumanMessage(Message, HumanMessage):
    role: MessageRole = Field(MessageRole.HUMAN, frozen=True)
    date_filter: Optional[DateFilter] = Field(None)
    quality_preset: QualityPreset = Field()

    def convert_to_data(self) -> dict:
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
    def create_from_data(cls, data: dict) -> "MyHumanMessage":
        return cls(
            id=data["id"],
            content=data["content"],
            date_filter=(
                None
                if data.get("dateFilter") is None
                else DateFilter.create_from_data(data["dateFilter"])
            ),
            quality_preset=QualityPreset(data["qualityPreset"]),
        )
