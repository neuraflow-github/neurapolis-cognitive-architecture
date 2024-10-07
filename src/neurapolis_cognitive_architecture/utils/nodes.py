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
    # print("config", config)
    last_message = state["messages"][-1]
    tool_call = last_message.tool_calls[0]
    query = tool_call["args"]["full_query"]

    async def get_relevant_content(query):
        from neurapolis_retriever.graph import graph
        from neurapolis_retriever.models.loader_update import LoaderUpdate
        from neurapolis_retriever.state.state import State

        file_infos = []

        from neurapolis_retriever.retriever_config import retriever_config

        retriever_config.planner_vector_search_limit = 1
        retriever_config.planner_keyword_search_limit = 1
        retriever_config.planner_min_relevant_hit_count = 1
        retriever_config.planner_max_search_count = 1
        retriever_config.sub_planner_keyword_search_limit = 1
        retriever_config.sub_planner_relevant_hits_limit = 1

        async for x_event in graph.astream_events(State(query=query), version="v1"):
            pass
            # if isinstance(x_event, dict) and x_event.get("event") == "on_chain_end":
            #     state_data = x_event.get("data", {}).get("output")
            #     if state_data and isinstance(state_data, dict):
            #         state = State(**state_data)
            #         loader_update = LoaderUpdate(
            #             retriever_step=state.retriever_step,
            #             search_count=state.search_count,
            #             hit_count=state.hit_count,
            #             relevant_hit_count=state.relevant_hit_count,
            #             log_entries=[],
            #         )
            #         config["send_loader_update_to_client"](loader_update)

        # Mock file inputs for testing purposes
        file_infos = [
            {
                "id": "mock_id_1",
                "name": "Mock Document 1",
                "description": "This is a mock document about playgrounds.",
                "text": "The latest playground in our city was built in 2023.",
                "created_at": "2024-03-15T10:00:00Z",
                "pdf_url": "https://example.com/mock_document_1.pdf",
                "highlight_areas": [],
            },
            {
                "id": "mock_id_2",
                "name": "Mock Document 2",
                "description": "Another mock document about city development.",
                "text": "City plans include building a new playground next year.",
                "created_at": "2024-03-16T14:30:00Z",
                "pdf_url": "https://example.com/mock_document_2.pdf",
                "highlight_areas": [],
            },
        ]

        return file_infos

    relevant_content = await get_relevant_content(query)

    print("\033[94m" + str(relevant_content) + "\033[0m")

    function_message = ToolMessage(
        content=relevant_content,
        name=tool_call["name"],
        tool_call_id=tool_call["id"],
    )

    return {"messages": [function_message]}
