from abc import ABC, abstractmethod

from neurapolis_cognitive_architecture.enums import MessageRole
from neurapolis_common import Dto


class Message(Dto["Message"], ABC):
    id: str
    role: MessageRole
    content: str

    def __init__(
        self,
        id: str,
        role: MessageRole,
        content: str,
    ):
        self.id = id
        self.role = role
        self.content = content

    @abstractmethod
    def convert_to_data(self) -> dict:
        pass

    @classmethod
    @abstractmethod
    def create_from_data(cls, message_dto: dict) -> "Message":
        pass
