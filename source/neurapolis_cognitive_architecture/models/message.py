from abc import ABC, abstractmethod

from neurapolis_common.models.dto import Dto

from ..enums.message_role import MessageRole


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
    def to_dto(self) -> dict:
        pass

    @classmethod
    @abstractmethod
    def from_dto(cls, message_dto: dict) -> "Message":
        pass
