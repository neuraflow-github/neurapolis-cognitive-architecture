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
                retriever_step=RetrieverStep.PLANNER,
                search_count=i + 1,
                hit_count=i * 2,
                relevant_hit_count=i,
                log_entries=[
                    TextLoaderLogEntry(
                        str(uuid4()),
                        "Processing step " + str(i + 1),
                    ),
                ],
            )
            await send_loader_update_to_client(loader_update)
        # Mock message with file info
        file_info = FileInfo(
            id="123",
            name="example.pdf",
            description="An example PDF file",
            text="This is the content of the PDF file.",
            created_at=datetime.now(),
            pdf_url="https://ris.freiburg.de/documents.php?document_type_id=4&submission_id=1003006100000&id=69&inline=true",
            highlight_areas=[
                FileHighlightArea(
                    page_index=0,
                    left_percentage=10.0,
                    top_percentage=20.0,
                    width_percentage=30.0,
                    height_percentage=5.0,
                )
            ],
        )
        message = Message(
            "msg_123",
            MessageRole.AI,
            "Here's the information you requested.",
            None,
            [file_info],
        )
        await send_message_to_client(message)
        pass
