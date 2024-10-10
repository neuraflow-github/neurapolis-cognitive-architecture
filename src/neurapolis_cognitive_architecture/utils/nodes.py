import datetime
import json
from functools import lru_cache
from typing import List

import pytz
from langchain import hub
from langchain_aws import ChatBedrock
from langchain_core.documents import Document
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableLambda
from langchain_openai import AzureChatOpenAI

from neurapolis_cognitive_architecture.utils.tools import tools


def should_continue(state):
    messages = state["messages"]
    last_message = messages[-1]
    return "continue" if last_message.tool_calls else "end"


from pydantic import BaseModel


async def call_model(state, config):
    messages = state["messages"]

    model = ChatBedrock(
        model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
        model_kwargs=dict(temperature=0),
        name="agent",
    )

    prompt = hub.pull("neurapolis-ca-dev")

    model = prompt | model.bind_tools(tools)
    response = await model.ainvoke(messages)
    return {"messages": [response]}


async def call_tool(state, config):

    print(
        "\033[94mconfig",
        config["configurable"]["send_loader_update_to_client"],
        "\033[0m",
    )
    last_message = state["messages"][-1]
    from typing import cast

    from neurapolis_cognitive_architecture.utils.state import FilteredBaseMessage

    last_human_message = cast(
        FilteredBaseMessage,
        next((msg for msg in reversed(state["messages"]) if msg.type == "human"), None),
    )
    tool_call = last_message.tool_calls[0]
    query = tool_call["args"]["full_query"]

    from neurapolis_retriever.main import NeurapolisRetriever
    from neurapolis_retriever.models.date_filter import DateFilter
    from neurapolis_retriever.models.file_hit import FileHit
    from neurapolis_retriever.models.loader_update import LoaderUpdate
    from neurapolis_retriever.models.quality_preset import QualityPreset

    retriever = NeurapolisRetriever()
    date_filter = last_human_message.date_filter
    file_hits = []

    async for x_event in retriever.retrieve(
        query,
        date_filter,
        QualityPreset.LOW,
    ):
        if isinstance(x_event, LoaderUpdate):
            loader_update = x_event
            print("---")
            print(f"Retriever Step: {loader_update.retriever_step}")
            print(f"Search Count: {loader_update.search_count}")
            print(f"Hit Count: {loader_update.hit_count}")
            print(f"Relevant Hit Count: {loader_update.relevant_hit_count}")
            print("Log Entries:")
            await config["configurable"]["send_loader_update_to_client"](loader_update)
            for x_log_entry in loader_update.log_entries:
                print(f"  - {x_log_entry.text}")

            print("---")
        elif isinstance(x_event, list):
            file_hits = x_event
            print("---")
            print("Retrieved File Infos:")
            for x_file_hit in file_hits:
                print(f"  - {x_file_hit.name}: {x_file_hit.description}")
            print("---")
            break

    # Serialize file_hits before adding to the content
    serialized_file_hits = [file_hit.model_dump() for file_hit in file_hits]

    function_message = ToolMessage(
        content=serialized_file_hits,
        name=tool_call["name"],
        tool_call_id=tool_call["id"],
    )

    print("function_message", function_message)

    return {"messages": [function_message]}
