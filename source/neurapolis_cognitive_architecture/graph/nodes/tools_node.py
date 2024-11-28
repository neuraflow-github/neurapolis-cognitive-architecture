from typing import Any, Optional

from langchain_core.runnables.config import RunnableConfig
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from neurapolis_retriever import DateFilter, FileHit, LoaderUpdate, NeurapolisRetriever


@tool
async def retriever_tool(
    query: str, date_filter: DateFilter, config: RunnableConfig
) -> str:
    """Mit diesem Nachschlage-Tool kannst du Fakten nachschlagen. Es enthält alle Daten aus dem RIS.

    Args:
        query: Die Suchanfrage, nach der gesucht werden soll
        date_filter: Ein Filter für das Datum der Dokumente
    """
    retriever = NeurapolisRetriever()

    file_hits: Optional[list[FileHit]] = None
    async for x_event in retriever.retrieve(
        query,
        date_filter,
        config.quality_preset,
    ):
        if isinstance(x_event, LoaderUpdate):
            await config["configurable"]["send_loader_update_to_client"](x_event)
        elif isinstance(x_event, list):
            file_hits = x_event

    file_hits_data: list[dict[str, Any]] = []
    for x_file_hit in file_hits:
        file_hits_data.append(x_file_hit.model_dump())

    capped_file_hits = file_hits[:10]
    inner_xml = FileHit.format_multiple_to_inner_llm_xml(capped_file_hits)
    xml = f"<{FileHit.get_llm_xml_tag_name_prefix()}>\n{inner_xml}\n</{FileHit.get_llm_xml_tag_name_prefix()}>"

    return xml, file_hits_data


tools = [retriever_tool]

tools_node = ToolNode(tools)
