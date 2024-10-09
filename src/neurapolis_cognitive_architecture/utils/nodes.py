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
    print("\033[94mconfig", config, "\033[0m")
    last_message = state["messages"][-1]
    from neurapolis_cognitive_architecture.utils.state import FilteredBaseMessage
    from typing import cast

    last_human_message = cast(
        FilteredBaseMessage,
        next((msg for msg in reversed(state["messages"]) if msg.type == "human"), None),
    )
    tool_call = last_message.tool_calls[0]
    query = tool_call["args"]["full_query"]

    async def get_file_infos(query):
        from datetime import datetime
        from typing import AsyncIterator, Optional
        from neurapolis_retriever.graph import graph
        from neurapolis_retriever.models.date_filter import DateFilter
        from neurapolis_retriever.models.file_info import FileInfo
        from neurapolis_retriever.models.loader_update import LoaderUpdate
        from neurapolis_retriever.models.retriever_step import RetrieverStep
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

        retriever_step_values = [step.value for step in RetrieverStep]

        def get_attr_or_key(obj, attr, default=None):
            if hasattr(obj, attr):
                return getattr(obj, attr)
            elif isinstance(obj, dict) and attr in obj:
                return obj[attr]
            else:
                return default

        try:
            async for x_event in graph.astream_events(
                State(
                    query=query,
                    date_filter=DateFilter(
                        start_at=datetime.now(),
                        end_at=datetime.now(),
                    ),
                ),
                include_names=retriever_step_values,
                include_types=["on_chain_start", "on_chain_end"],
                version="v1",
            ):
                event_name = x_event["name"]
                try:
                    retriever_step = RetrieverStep(event_name)
                except:
                    continue
                event_type = x_event["event"]
                if event_type == "on_chain_start":
                    state = x_event["data"]["input"]
                    if type(state) != State:
                        continue
                    if retriever_step == RetrieverStep.FINISHER:
                        continue
                    loader_update = LoaderUpdate(
                        retriever_step,
                        state.search_count,
                        state.hit_count,
                        state.relevant_hit_count,
                        [],
                    )
                    yield loader_update
                elif event_type == "on_chain_end":
                    state = x_event["data"]["output"]
                    if type(state) != State:
                        continue
                    if retriever_step != RetrieverStep.FINISHER:
                        continue
                    file_infos = []
                    if state:
                        for x_search in get_attr_or_key(state, "searches", []):
                            if not x_search:
                                continue
                            x_hits = get_attr_or_key(x_search, "hits", [])
                            for x_hit in x_hits:
                                if not x_hit:
                                    continue
                                x_grading = get_attr_or_key(x_hit, "grading")
                                if not x_grading:
                                    continue
                                is_relevant = get_attr_or_key(
                                    x_grading, "is_relevant", False
                                )
                                if is_relevant:
                                    try:
                                        related_file = get_attr_or_key(
                                            x_hit, "related_file", {}
                                        )
                                        file_info = FileInfo(
                                            id=get_attr_or_key(related_file, "id"),
                                            name=get_attr_or_key(related_file, "name"),
                                            description=get_attr_or_key(
                                                x_grading, "description"
                                            ),
                                            text=get_attr_or_key(x_hit, "text"),
                                            created_at=get_attr_or_key(
                                                related_file, "created"
                                            ),
                                            pdf_url=get_attr_or_key(
                                                related_file, "download_url"
                                            ),
                                            highlight_areas=[],
                                        )
                                        file_infos.append(file_info)
                                        if len(file_infos) >= 20:
                                            break
                                    except Exception as e:
                                        print(f"An error occurred: {e}")
                                        continue
                            if len(file_infos) >= 20:
                                break
                        yield file_infos
                        break
        except Exception as e:
            print(f"An error occurred during file retrieval: {e}")
            yield []

    file_infos = []
    async for result in get_file_infos(query):
        if isinstance(result, list):
            file_infos = result
            break
        # Handle LoaderUpdate if needed

    # Serialize file_infos before adding to the content
    serialized_file_infos = [file_info.model_dump() for file_info in file_infos]

    function_message = ToolMessage(
        content=serialized_file_infos,
        name=tool_call["name"],
        tool_call_id=tool_call["id"],
    )

    print("function_message", function_message)

    return {"messages": [function_message]}
