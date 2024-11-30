import asyncio
from typing import Any, Optional

from langchain_core.runnables.config import RunnableConfig
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from neurapolis_retriever import (
    DateFilter,
    LoaderUpdate,
    NeurapolisRetriever,
    RetrievedFile,
)


@tool
def retriever_tool(
    query: str, date_filter: DateFilter, runnable_config: RunnableConfig
) -> str:
    """Mit diesem Nachschlage-Tool kannst du Fakten nachschlagen. Es enthält alle Daten aus dem RIS.

    Args:
        query: Die Suchanfrage, nach der gesucht werden soll
        date_filter: Ein Filter für das Datum der Dokumente
    """
    print("retriever_tool", query, date_filter, runnable_config)

    retriever = NeurapolisRetriever()

    async def _retrieve():
        retrieved_files: Optional[list[RetrievedFile]] = None
        async for x_event in retriever.retrieve(
            query,
            date_filter,
            runnable_config.quality_preset,
        ):
            if isinstance(x_event, LoaderUpdate):
                await runnable_config["configurable"]["send_loader_update_to_client"](
                    x_event
                )
            elif isinstance(x_event, list):
                retrieved_files = x_event
        return retrieved_files

    retrieved_files = asyncio.run(_retrieve())

    retrieved_files_data: list[dict[str, Any]] = []
    for x_retrieved_file in retrieved_files:
        retrieved_files_data.append(x_retrieved_file.model_dump())

    capped_retrieved_files = retrieved_files[:10]
    inner_xml = RetrievedFile.format_multiple_to_inner_llm_xml(capped_retrieved_files)
    xml = f"<{RetrievedFile.get_llm_xml_tag_name_prefix()}>\n{inner_xml}\n</{RetrievedFile.get_llm_xml_tag_name_prefix()}>"

    return xml, retrieved_files_data


tools = [retriever_tool]

tools_node = ToolNode(tools)
