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
        send_message_to_client: Callable[[str], None],
    ):

        from langchain_core.callbacks import BaseCallbackHandler
        from langchain_core.messages import HumanMessage

        from neurapolis_cognitive_architecture.agent import graph

        state = {"messages": [HumanMessage(content=query)]}
        async for event in graph.astream(
            state,
            {
                "configurable": {
                    "thread_id": thread_id,
                },
                "send_loader_update_to_client": send_loader_update_to_client,
                "send_message_to_client": send_message_to_client,
            },
            subgraphs=True,
        ):
            print(event)
            # file_info = FileInfo(
            #     id="123",
            #     name="example.pdf",
            #     description="An example PNG file",
            #     text="This is the content of the PDF file.",
            #     created_at=datetime.now(),
            #     pdf_url="https://example.com/example.pdf",
            #     highlight_areas=[
            #         FileHighlightArea(
            #             page_index=0,
            #             left_percentage=10.0,
            #             top_percentage=20.0,
            #             width_percentage=30.0,
            #             height_percentage=5.0,
            #         )
            #     ],
            # )
            # message = Message(
            #     "msg_123",
            #     MessageRole.AI,
            #     "Here's the information you requested.",
            #     None,
            #     [file_info],
            # )
            # await send_message_to_client(message)
            # break

            if isinstance(event, tuple) and len(event) == 2:
                action, data = event
                for step, step_data in data.items():
                    if step in RetrieverStep._member_names_:
                        loader_update = LoaderUpdate(
                            retriever_step=RetrieverStep(step),
                            search_count=step_data.get("search_count", 0),
                            hit_count=step_data.get("hit_count", 0),
                            relevant_hit_count=step_data.get("relevant_hit_count", 0),
                            log_entries=[
                                TextLoaderLogEntry(
                                    str(uuid4()),
                                    f"Processing step {step}",
                                ),
                            ],
                        )
                        print("\033[94mloader_update", loader_update, "\033[0m")
                        await send_loader_update_to_client(loader_update)
                    else:
                        # Check if the event matches the expected format
                        from langchain_core.messages import ToolMessage

                        if (
                            isinstance(data, dict)
                            and "action" in data
                            and "messages" in data["action"]
                            and isinstance(data["action"]["messages"], list)
                            and len(data["action"]["messages"]) > 0
                            and isinstance(data["action"]["messages"][0], ToolMessage)
                        ):
                            tool_message = data["action"]["messages"][0]
                            content = tool_message.content

                            # Create FileInfo objects from the content
                            file_infos = [
                                FileInfo(
                                    id=item["id"],
                                    name=item["name"],
                                    description=item["description"],
                                    text=item["text"],
                                    created_at=datetime.fromisoformat(
                                        item["created_at"].rstrip("Z")
                                    ),
                                    pdf_url=item["pdf_url"],
                                    highlight_areas=[],  # Assuming no highlight areas for now
                                )
                                for item in content
                            ]

                            # Create and send the message
                            message = Message(
                                "msg_" + str(uuid4()),
                                MessageRole.AI,
                                "Here's the information you requested.",
                                None,
                                file_infos,
                            )
                            await send_message_to_client(message)


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
