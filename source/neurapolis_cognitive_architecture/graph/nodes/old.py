from typing import Any, Optional

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from neurapolis_cognitive_architecture.models import State
from neurapolis_common import get_last_message_of_type
from neurapolis_retriever import LoaderUpdate, NeurapolisRetriever, RetrievedFile


async def retriever_node(state: State, config) -> State:
    last_human_message = get_last_message_of_type(state.messages, HumanMessage)
    last_tool_message = get_last_message_of_type(state.messages, ToolMessage)
    # TODO get tool message and query

    retriever = NeurapolisRetriever()

    retrieved_files: Optional[list[RetrievedFile]] = None
    async for x_event in retriever.retrieve(
        last_human_message.content,
        last_human_message.date_filter,
        last_human_message.quality_preset,
    ):
        if isinstance(x_event, LoaderUpdate):
            await config["configurable"]["send_loader_update_to_client"](x_event)
        elif isinstance(x_event, list):
            retrieved_files = x_event

    retrieved_files_data: list[dict[str, Any]] = []
    for x_retrieved_file in retrieved_files:
        retrieved_files_data.append(x_retrieved_file.model_dump())

    function_message = ToolMessage(
        content=retrieved_files_data,
        name=last_tool_message.name,
        tool_call_id=last_tool_message.tool_call_id,
    )

    return {"messages": [function_message]}


@tool
async def retriever_tool(query: str, date_filter: dict, config) -> str:
    retriever = NeurapolisRetriever()

    retriever = NeurapolisRetriever()

    retrieved_files: Optional[list[RetrievedFile]] = None
    async for x_event in retriever.retrieve(
        query,
        date_filter,
        quality_preset,
    ):
        if isinstance(x_event, LoaderUpdate):
            await config["configurable"]["send_loader_update_to_client"](x_event)
        elif isinstance(x_event, list):
            retrieved_files = x_event

    retrieved_files_data: list[dict] = []
    for x_retrieved_file in retrieved_files:
        retrieved_files_data.append(x_retrieved_file.model_dump())

    return retrieved_files_data


tools = [retriever_tool]

tool_node = ToolNode(tools)
