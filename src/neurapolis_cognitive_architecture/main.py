import asyncio
from datetime import datetime
from typing import Callable, Optional
from uuid import uuid4

from neurapolis_retriever.models.date_filter import DateFilter
from neurapolis_retriever.models.file_highlight_area import FileHighlightArea
from neurapolis_retriever.models.file_info import FileInfo
from neurapolis_retriever.models.loader_update import LoaderUpdate
from neurapolis_retriever.models.retriever_step import RetrieverStep
from neurapolis_retriever.models.text_loader_log_entry import TextLoaderLogEntry

from neurapolis_cognitive_architecture.models.message import Message
from neurapolis_cognitive_architecture.models.message_role import MessageRole


class NeurapolisCognitiveArchitecture:
    async def query(
        self,
        thread_id: str,
        query: str,
        date_filter: Optional[DateFilter],
        send_loader_update_to_client: Callable[[LoaderUpdate], None],
        send_message_to_client: Callable[[Message], None],
    ):
        from neurapolis_cognitive_architecture.agent import graph

        state = {"messages": []}
        config = {"thread_id": thread_id}

        from langchain_core.messages import HumanMessage

        state["messages"].append(HumanMessage(content=query))

        async for event in graph.astream(
            state,
            {
                "send_loader_update_to_client": send_loader_update_to_client,
                "send_message_to_client": send_message_to_client,
                "thread_id": thread_id,
            },
        ):
            print(event)
            if isinstance(event, dict) and "agent" in event:
                agent_messages = event["agent"].get("messages", [])
                for msg in agent_messages:
                    if hasattr(msg, "content") and msg.content:
                        await send_message_to_client(
                            Message(str(uuid4()), MessageRole.AI, msg.content, None, [])
                        )


if __name__ == "__main__":
    neurapolis_cognitive_architecture = NeurapolisCognitiveArchitecture()
    thread_id = str(uuid4())

    async def print_message(message: Message):
        print(f"AI: {message.content}")

    asyncio.run(
        neurapolis_cognitive_architecture.query(
            thread_id,
            "Wann wurde der letzte Spielplatz erbaut?",
            None,
            print,
            print_message,
        )
    )
