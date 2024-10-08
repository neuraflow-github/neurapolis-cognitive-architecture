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
    # print("config", config)
    last_message = state["messages"][-1]
    tool_call = last_message.tool_calls[0]
    query = tool_call["args"]["full_query"]

    async def get_file_infos(query):
        from neurapolis_retriever.graph import graph
        from neurapolis_retriever.models.file_info import FileInfo
        from neurapolis_retriever.state.state import State
        from neurapolis_retriever.retriever_config import retriever_config

        retriever_config.planner_vector_search_limit = 1
        retriever_config.planner_keyword_search_limit = 1
        retriever_config.planner_min_relevant_hit_count = 1
        retriever_config.planner_max_search_count = 1
        retriever_config.sub_planner_keyword_search_limit = 1
        retriever_config.sub_planner_relevant_hits_limit = 1
        retriever_config.initial_retriever_vector_search_top_k = 30
        retriever_config.reranked_retriever_vector_search_top_k = 5
        retriever_config.initial_retriever_keyword_search_top_k = 30
        retriever_config.reranked_retriever_keyword_search_top_k = 5

        file_infos = []

        async for x_event in graph.astream_events(State(query=query), version="v1"):
            if isinstance(x_event, dict) and x_event.get("event") == "on_chain_end":
                # print("x_event", x_event)
                event_name = x_event.get("name")
                print("event_name", event_name)
                from neurapolis_retriever.models.retriever_step import RetrieverStep

                if event_name == RetrieverStep.FINISHER.value:
                    state = x_event.get("data", {}).get("output")

                    if state:
                        for x_search in state.searches:
                            for x_hit in x_search.hits:
                                if x_hit.grading.is_relevant:
                                    file_info = FileInfo(
                                        id=x_hit.related_file.id,
                                        name=x_hit.related_file.name,
                                        description=x_hit.grading.description,
                                        text=x_hit.text,
                                        created_at=x_hit.related_file.created,
                                        pdf_url=x_hit.related_file.download_url,
                                        highlight_areas=[],
                                    )
                                    file_infos.append(file_info)
                                    if len(file_infos) >= 20:
                                        break
                            if len(file_infos) >= 20:
                                break
                    break

        return file_infos

    file_infos = await get_file_infos(query)

    # Serialize file_infos before adding to the content
    serialized_file_infos = [file_info.model_dump() for file_info in file_infos]

    # print("\033[94m" + str(content) + "\033[0m")

    function_message = ToolMessage(
        content=serialized_file_infos,
        name=tool_call["name"],
        tool_call_id=tool_call["id"],
    )

    print("function_message", function_message)

    return {"messages": [function_message]}
