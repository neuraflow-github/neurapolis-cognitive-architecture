import asyncio
from datetime import datetime
from typing import Callable

from neurapolis_retriever.models.file_highlight_area import FileHighlightArea
from neurapolis_retriever.models.file_info import FileInfo
from neurapolis_retriever.models.loader_log_entry import LoaderLogEntry
from neurapolis_retriever.models.loader_update import LoaderUpdate
from neurapolis_retriever.models.retriever_step import RetrieverStep

from neurapolis_cognitive_architecture.models.message import Message
from neurapolis_cognitive_architecture.models.message_role import MessageRole


class NeurapolisCognitiveArchitecture:
    async def query(
        self,
        query: str,
        send_loader_update_to_client: Callable[[LoaderUpdate], None],
        send_message_to_client: Callable[[str], None],
    ):
        # graph.invoke(
        #     {
        #         "query": query,
        #     },
        #     {
        #         "send_loader_update_to_client": send_loader_update_to_client,
        #         "send_message_to_client": send_message_to_client,
        #     },
        # )

        # Mock loader updates
        for i in range(5):
            await asyncio.sleep(1)  # Simulate some processing time
            loader_update = LoaderUpdate(
                retriever_step=RetrieverStep.SEARCHING,
                search_count=i + 1,
                hit_count=i * 2,
                relevant_hit_count=i,
                log_entries=[
                    LoaderLogEntry(message=f"Processing step {i + 1}", level="INFO")
                ],
            )
            send_loader_update_to_client(loader_update)
        # Mock message with file info
        file_info = FileInfo(
            id="123",
            name="example.pdf",
            description="An example PDF file",
            text="This is the content of the PDF file.",
            created_at=datetime.now(),
            pdf_url="https://example.com/example.pdf",
            highlight_areas=[
                FileHighlightArea(page=1, x=100, y=200, width=300, height=50)
            ],
        )
        message = Message(
            id="msg_123",
            role=MessageRole.AI,
            content="Here's the information you requested.",
            files=[file_info],
        )
        send_message_to_client(message)
        pass
