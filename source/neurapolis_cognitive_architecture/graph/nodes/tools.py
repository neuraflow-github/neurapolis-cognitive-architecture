import logging
from typing import Callable, Optional

from langchain_core.tools import InjectedToolArg, tool
from neurapolis_cognitive_architecture.config import config
from neurapolis_retriever import (
    DateFilter,
    LoaderUpdate,
    NeurapolisRetriever,
    QualityPreset,
    RetrievedFile,
)
from typing_extensions import Annotated

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


# Langgraph way of injecting tool args: https://langchain-ai.github.io/langgraph/how-tos/pass-run-time-values-to-tools/?h=tools#define-the-tools
# Or inject tool args: https://python.langchain.com/docs/how_to/tool_runtime/#hiding-arguments-from-the-model
@tool(response_format="content_and_artifact")
async def retrieve(
    query: str,
    date_filter: Annotated[Optional[DateFilter], InjectedToolArg],
    quality_preset: Annotated[QualityPreset, InjectedToolArg],
    send_loader_update_to_client: Annotated[
        Callable[[LoaderUpdate], None], InjectedToolArg
    ],
) -> any:
    """
    Mit dem Nachschlagetool kannst du relevante Informationen aus dem RIS herausfinden.

    Args:
        query: Die Suchanfrage an das Nachschlagetool.
    """

    logger.info(f"ToolNode: Started retrieving")

    retriever = NeurapolisRetriever()

    retrieved_files: list[RetrievedFile] = []
    for x_event in retriever.retrieve(
        query,
        date_filter,
        quality_preset,
    ):
        if isinstance(x_event, LoaderUpdate):
            await send_loader_update_to_client(
                x_event
            )  # Do not await this, to not block the graph
        elif isinstance(x_event, list):
            retrieved_files = x_event

    capped_retrieved_files = retrieved_files[: config.retrieved_file_limit]
    inner_xml = RetrievedFile.format_multiple_to_inner_llm_xml(capped_retrieved_files)
    xml = f"<{RetrievedFile.get_llm_xml_tag_name_prefix()}>\n{inner_xml}\n</{RetrievedFile.get_llm_xml_tag_name_prefix()}>"

    logger.info(f"ToolNode: Finished retrieving")

    return xml, retrieved_files


tools = [retrieve]
