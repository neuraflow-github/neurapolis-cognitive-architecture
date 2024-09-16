from typing import Optional, Any, Dict
import uuid
import asyncio
import nest_asyncio
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from cognitive_architecture.agent import graph

nest_asyncio.apply()

messages = []
session = {}  # Store session information, including thread_id for pairs


async def process_user_input(user_input):
    messages.append(user_input)
    state = {"messages": messages}

    if "thread_id" not in session:
        session["thread_id"] = str(uuid.uuid4())  # Generate a new thread_id for pairs

    config = {
        "thread_id": session["thread_id"]
    }  # Include thread_id in config for pairs

    async for event in graph.astream_events(state, config=config, version="v1"):
        kind = event["event"]

        if kind == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            if isinstance(chunk.content, list):
                for item in chunk.content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        print(item["text"], end="", flush=True)
            else:
                print(chunk.content, end="", flush=True)
        elif kind == "on_tool_start":
            print("\n--")
            print(
                f"Starting tool: {event['name']} with inputs: {event['data'].get('input')}"
            )
        elif kind == "on_tool_end":
            print(f"Done tool: {event['name']}")
            print(f"Tool output was: {event['data'].get('output')}")
            print("--\n")


async def main():
    try:
        while True:
            user_input = input("\nEnter your question (or 'exit' to quit): ")
            if user_input.lower() == "exit":
                break
            await process_user_input(HumanMessage(content=user_input))
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    asyncio.run(main())
