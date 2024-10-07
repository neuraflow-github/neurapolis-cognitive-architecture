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

<<<<<<< HEAD
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

                        # else

                        # ((), {'action': {'messages': [ToolMessage(content=[{'id': 'mock_id_1', 'name': 'Mock Document 1', 'description': 'This is a mock document about playgrounds.', 'text': 'The latest playground in our city was built in 2023.', 'created_at': '2024-03-15T10:00:00Z', 'pdf_url': 'https://example.com/mock_document_1.pdf', 'highlight_areas': []}, {'id': 'mock_id_2', 'name': 'Mock Document 2', 'description': 'Another mock document about city development.', 'text': 'City plans include building a new playground next year.', 'created_at': '2024-03-16T14:30:00Z', 'pdf_url': 'https://example.com/mock_document_2.pdf', 'highlight_areas': []}], name='search_docs', tool_call_id='toolu_bdrk_01CL6k3ELeZQxk2VbgPHaYvX')]}})

        # Mock loader updates and message with file info code removed


if __name__ == "__main__":
    import asyncio

    async def main():
        import uuid

        thread_id = str(uuid.uuid4())
        neurapolis_cognitive_architecture = NeurapolisCognitiveArchitecture()
        await neurapolis_cognitive_architecture.query(
            thread_id, "Wann wurde der letzte Spielplatz erbaut??", None, None, None
=======
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
>>>>>>> dc47fb8d454c06d71f8263ddf54c656f50aa18b8
        )

    asyncio.run(main())
