from typing import Any, Optional

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from neurapolis_cognitive_architecture.models.state import State
from neurapolis_cognitive_architecture.utilities import get_last_message_of_type
from neurapolis_retriever.main import NeurapolisRetriever
from neurapolis_retriever.models.date_filter import DateFilter
from neurapolis_retriever.models.file_hit import FileHit
from neurapolis_retriever.models.loader_update import LoaderUpdate
from neurapolis_retriever.models.quality_preset import QualityPreset


async def retriever_node(state: State, config) -> State:
    last_human_message = get_last_message_of_type(state.messages, HumanMessage)
    last_tool_message = get_last_message_of_type(state.messages, ToolMessage)
    # TODO get tool message and query

    retriever = NeurapolisRetriever()

    file_hits: Optional[list[FileHit]] = None
    async for x_event in retriever.retrieve(
        last_human_message.content,
        last_human_message.date_filter,
        last_human_message.quality_preset,
    ):
        if isinstance(x_event, LoaderUpdate):
            await config["configurable"]["send_loader_update_to_client"](x_event)
        elif isinstance(x_event, list):
            file_hits = x_event

    file_hits_data: list[dict[str, Any]] = []
    for x_file_hit in file_hits:
        file_hits_data.append(x_file_hit.model_dump())

    function_message = ToolMessage(
        content=file_hits_data,
        name=last_tool_message.name,
        tool_call_id=last_tool_message.tool_call_id,
    )

    return {"messages": [function_message]}


@tool
async def retriever_tool(query: str, date_filter: dict, config) -> str:
    retriever = NeurapolisRetriever()

    retriever = NeurapolisRetriever()

    file_hits: Optional[list[FileHit]] = None
    async for x_event in retriever.retrieve(
        query,
        date_filter,
        quality_preset,
    ):
        if isinstance(x_event, LoaderUpdate):
            await config["configurable"]["send_loader_update_to_client"](x_event)
        elif isinstance(x_event, list):
            file_hits = x_event

    file_hits_data: list[dict[str, Any]] = []
    for x_file_hit in file_hits:
        file_hits_data.append(x_file_hit.model_dump())

    return file_hits_data


tools = [retriever_tool]

tool_node = ToolNode(tools)
