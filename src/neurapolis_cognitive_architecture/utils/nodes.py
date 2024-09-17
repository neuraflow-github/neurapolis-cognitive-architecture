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

    def get_page_contents(query):

        from neurapolis_retriever.graph import graph as graph_retriever

        results = graph_retriever.invoke({"query": query})
        searches = results.get("searches", [])
        page_contents = ""
        for search in searches[:1]:
            hits = search.hits if hasattr(search, "hits") else []
            for hit in hits:
                print(hit)
                if hasattr(hit, "related_file") and hit.related_file is not None:
                    if hasattr(hit.related_file, "extracted_text"):
                        page_contents += hit.related_file.extracted_text + "\n"
                    if hasattr(hit.related_file, "access_url"):
                        page_contents += f"Access URL: {hit.related_file.access_url}\n"
        return page_contents.strip()

    page_contents = get_page_contents(query)

    function_message = ToolMessage(
        content=page_contents, name=tool_call["name"], tool_call_id=tool_call["id"]
    )

    return {"messages": [function_message]}
