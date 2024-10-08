import asyncio
from datetime import datetime
from typing import Callable, Optional
from uuid import uuid4

from neurapolis_retriever.models.date_filter import DateFilter
from neurapolis_retriever.models.file_highlight_area import FileHighlightArea
from neurapolis_retriever.models.file_hit import FileHit
from neurapolis_retriever.models.loader_update import LoaderUpdate
from neurapolis_retriever.models.retriever_step import RetrieverStep
from neurapolis_retriever.models.text_loader_log_entry import TextLoaderLogEntry

from neurapolis_cognitive_architecture.models.ai_message import AiMessage
from neurapolis_cognitive_architecture.models.message import Message
from neurapolis_cognitive_architecture.models.message_role import MessageRole
from neurapolis_cognitive_architecture.models.user_message import UserMessage


class NeurapolisCognitiveArchitecture:
    async def query(
        self,
        thread_id: str,
        user_message: UserMessage,
        send_loader_update_to_client: Callable[[LoaderUpdate], None],
        send_message_to_client: Callable[[str], None],
    ):

        # print("thread_id", thread_id)

        from langchain_core.callbacks import BaseCallbackHandler
        from langchain_core.messages import AIMessage as LcAIMessage
        from langchain_core.messages import HumanMessage, ToolMessage

        from neurapolis_cognitive_architecture.agent import graph
        from neurapolis_cognitive_architecture.utils.state import FilteredBaseMessage

        state = {
            "messages": [
                FilteredBaseMessage(
                    content=user_message.content, date_filter=user_message.date_filter
                )
            ]
        }
        previous_tool_call = None
        async for event in graph.astream(
            state,
            {
                "configurable": {
                    "thread_id": thread_id,
                    "send_loader_update_to_client": send_loader_update_to_client,
                    "send_message_to_client": send_message_to_client,
                },
            },
            subgraphs=True,
        ):
            # print(event)

            if isinstance(event, tuple) and len(event) == 2:
                print("0")
                action, data = event
                for step, step_data in data.items():
                    if step in RetrieverStep._member_names_:
                        pass
                        # loader_update = LoaderUpdate(
                        #     retriever_step=RetrieverStep(step),
                        #     search_count=step_data.get("search_count", 0),
                        #     hit_count=step_data.get("hit_count", 0),
                        #     relevant_hit_count=step_data.get("relevant_hit_count", 0),
                        #     log_entries=[
                        #         TextLoaderLogEntry(
                        #             str(uuid4()), text=f"Processing step {step}"
                        #         )
                        #     ],
                        # )
                        # print("\033[94mloader_update", loader_update, "\033[0m")
                        # await send_loader_update_to_client(loader_update)
                    else:
                        print("1")
                        if (
                            isinstance(data, dict)
                            and "agent" in data
                            and "messages" in data["agent"]
                            and isinstance(data["agent"]["messages"], list)
                            and len(data["agent"]["messages"]) > 0
                            and isinstance(data["agent"]["messages"][0], LcAIMessage)
                            and not data["agent"]["messages"][0].tool_calls
                        ):
                            print("2")
                            ai_message = data["agent"]["messages"][0]
                            content = ai_message.content

                            print("2")
                            if previous_tool_call is not None:
                                file_hits = [
                                    FileHit(**item)
                                    for item in previous_tool_call.content
                                ]
                            else:
                                file_hits = []

                            print("3")
                            message = AiMessage(
                                id="msg_" + str(uuid4()),
                                content=content,
                                file_hits=file_hits,
                            )
                            print("4")
                            await send_message_to_client(message)
                            print("5")
                            previous_tool_call = None
                        if (
                            isinstance(data, dict)
                            and "action" in data
                            and "messages" in data["action"]
                            and isinstance(data["action"]["messages"], list)
                            and len(data["action"]["messages"]) > 0
                            and isinstance(data["action"]["messages"][0], ToolMessage)
                        ):
                            previous_tool_call = data["action"]["messages"][0]


if __name__ == "__main__":
    import asyncio

    async def main():
        import uuid

        thread_id = str(uuid.uuid4())
        neurapolis_cognitive_architecture = NeurapolisCognitiveArchitecture()
        await neurapolis_cognitive_architecture.query(
            thread_id, "Wann wurde der letzte Spielplatz erbaut??", None, None, None
        )

    asyncio.run(main())
