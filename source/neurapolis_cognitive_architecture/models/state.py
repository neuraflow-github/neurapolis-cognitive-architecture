from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages] = []
    # messages: list[BaseMessage]
