import asyncio
import logging
from typing import AsyncGenerator, Callable, Optional

from langchain_core.tools import tool
from langchain_core.tools.base import BaseTool
from neurapolis_cognitive_architecture.config import config
from neurapolis_retriever import (
    DateFilter,
    LoaderUpdate,
    NeurapolisRetriever,
    QualityPreset,
    Reference,
)

logger = logging.getLogger()


# Langgraph way of injecting tool args: https://langchain-ai.github.io/langgraph/how-tos/pass-run-time-values-to-tools/?h=tools#define-the-tools
# Or inject tool args: https://python.langchain.com/docs/how_to/tool_runtime/#hiding-arguments-from-the-model
@tool(response_format="content_and_artifact")
async def retriever_tool(query: str):
    """
    Mit dem Nachschlagetool kannst du relevante Informationen aus dem RIS herausfinden.

    Args:
        query: Die Suchanfrage an das Nachschlagetool.
    """

    pass


tools: list[BaseTool] = [retriever_tool]


async def retrieve(
    query: str,
    date_filter: Optional[DateFilter],
    quality_preset: QualityPreset,
    send_loader_update_to_client: Callable[[LoaderUpdate], None],
) -> AsyncGenerator[tuple[str, list[Reference]], None]:
    retriever = NeurapolisRetriever()
    references: list[Reference] = []

    async for x_event in retriever.retrieve(query, date_filter, quality_preset):
        if isinstance(x_event, LoaderUpdate):
            # Use asyncio.create_task to not block
            await send_loader_update_to_client(x_event)
            await asyncio.sleep(0)  # Give control back to event loop
        elif isinstance(x_event, list):
            references = x_event

    capped_references = references[: config.reference_limit]
    inner_xml = Reference.format_multiple_to_inner_llm_xml(capped_references)
    xml = f"<{Reference.get_llm_xml_tag_name_prefix()}>\n{inner_xml}\n</{Reference.get_llm_xml_tag_name_prefix()}>"

    return xml, references
