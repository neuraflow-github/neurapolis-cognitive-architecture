from abc import ABC, abstractmethod

from neurapolis_cognitive_architecture.enums import MessageRole
from neurapolis_common import Dto
from pydantic import BaseModel, Field


class Message(BaseModel, Dto["Message"], ABC):
    id: str = Field()
    role: MessageRole = Field()
    content: str = Field()

    @abstractmethod
    def convert_to_data(self) -> dict:
        pass

    @classmethod
    @abstractmethod
    def create_from_data(cls, data: dict) -> "Message":
        pass
