from functools import lru_cache
from langchain_openai import AzureChatOpenAI
from neurapolis_cognitive_architecture.utils.tools import tools
from langchain import hub
from langchain_aws import ChatBedrock
from langchain_core.runnables import RunnableLambda
from typing import List
from langchain_core.documents import Document
from langchain_core.messages import ToolMessage
import json
import datetime
import pytz


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
    last_message = state["messages"][-1]
    tool_call = last_message.tool_calls[0]
    query = tool_call["args"]["full_query"]

    def extract_relevant_file_info(searches):
        relevant_files = []

        for search in searches:
            hits = search.hits if hasattr(search, "hits") else []
            for hit in hits:
                if (
                    hasattr(hit, "grading")
                    and hit.grading is not None
                    and getattr(hit.grading, "is_relevant", False)
                ):
                    if hasattr(hit, "related_file") and hit.related_file is not None:
                        file_info = {
                            "name": hit.related_file.name,
                            "access_url": hit.related_file.access_url,
                            "extracted_text": (
                                hit.related_file.extracted_text
                                if hasattr(hit.related_file, "extracted_text")
                                else None
                            ),
                        }
                        relevant_files.append(file_info)

        return relevant_files

    def get_relevant_content(query):
        from neurapolis_retriever.graph import graph as graph_retriever

        results = graph_retriever.invoke({"query": query})
        searches = results.get("searches", [])
        relevant_files = extract_relevant_file_info(searches)
        return relevant_files

    relevant_content = get_relevant_content(query)

    function_message = ToolMessage(
        content=str(relevant_content),
        name=tool_call["name"],
        tool_call_id=tool_call["id"],
    )

    return {"messages": [function_message]}
