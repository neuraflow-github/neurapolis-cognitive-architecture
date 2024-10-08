from langgraph.graph import add_messages
from langchain_core.messages import HumanMessage, BaseMessage
from typing import TypedDict, Annotated, Sequence, Optional
from neurapolis_retriever.models.date_filter import DateFilter
from pydantic import Field, ConfigDict


class FilteredBaseMessage(HumanMessage):
    date_filter: Optional[DateFilter] = Field(default=None)
    model_config = ConfigDict(arbitrary_types_allowed=True)


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
