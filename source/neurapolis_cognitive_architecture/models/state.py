from typing import Annotated

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages
from pydantic import BaseModel


class State(BaseModel):
    messages: Annotated[list[BaseMessage], add_messages]
