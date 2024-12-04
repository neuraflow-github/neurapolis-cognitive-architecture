from abc import ABC

from neurapolis_cognitive_architecture.enums import MessageRole
from neurapolis_common import Dto
from pydantic import BaseModel, Field


class Message(BaseModel, Dto["Message"], ABC):
    id: str = Field()
    role: MessageRole = Field()
    content: str = Field()
