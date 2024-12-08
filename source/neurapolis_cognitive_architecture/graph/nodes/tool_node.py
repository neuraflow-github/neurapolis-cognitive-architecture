import asyncio
import logging

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_core.tools.base import BaseTool
from langgraph.prebuilt import InjectedState, ToolNode
from neurapolis_cognitive_architecture.config import config as my_config
from neurapolis_cognitive_architecture.models import MyHumanMessage, State
from neurapolis_common import get_last_message_of_type
from neurapolis_retriever import LoaderUpdate, NeurapolisRetriever, Reference
from typing_extensions import Annotated

logger = logging.getLogger()


# Langgraph way of injecting tool args: https://langchain-ai.github.io/langgraph/how-tos/pass-run-time-values-to-tools/?h=tools#define-the-tools
# Or inject tool args: https://python.langchain.com/docs/how_to/tool_runtime/#hiding-arguments-from-the-model
@tool(response_format="content_and_artifact")
async def retrieve(
    query: str,
    config: RunnableConfig,
    state: Annotated[State, InjectedState],
) -> any:
    """
    Mit dem Nachschlagetool kannst du relevante Informationen aus dem RIS herausfinden.

    Args:
        query: Die Suchanfrage an das Nachschlagetool.
    """

    last_human_message: MyHumanMessage = get_last_message_of_type(
        state["messages"], MyHumanMessage
    )

    retriever = NeurapolisRetriever()

    references: list[Reference] = []
    async for x_event in retriever.retrieve(
        query,
        last_human_message.date_filter,
        last_human_message.quality_preset,
    ):
        print("UPDATE")
        if isinstance(x_event, LoaderUpdate):
            print("SENDING LOADER UPDATE TO CLIENT")
            await config["configurable"]["send_loader_update_to_client"](x_event)
            await asyncio.sleep(
                0
            )  # Give control back to event loop, so the loader update gets sent
            pass
        elif isinstance(x_event, list):
            references = x_event

    capped_references = references[: my_config.reference_limit]
    inner_xml = Reference.format_multiple_to_inner_llm_xml(capped_references)
    xml = f"<{Reference.get_llm_xml_tag_name_prefix()}>\n{inner_xml}\n</{Reference.get_llm_xml_tag_name_prefix()}>"

    return xml, references


tools: list[BaseTool] = [retrieve]

tool_node = ToolNode(tools)
