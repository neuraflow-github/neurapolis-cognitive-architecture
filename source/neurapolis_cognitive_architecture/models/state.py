from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages

from .cognitive_architecture_config import CognitiveArchitectureConfig


class State(TypedDict):
    config: CognitiveArchitectureConfig
    messages: Annotated[list[BaseMessage], add_messages] = []
