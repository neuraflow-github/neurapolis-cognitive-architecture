import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

from langchain_core.messages import AIMessage, ToolMessage
from neurapolis_cognitive_architecture.models import GraphConfig, State
from neurapolis_common import get_last_message_of_type
from neurapolis_retriever import (
    DateFilter,
    LoaderUpdate,
    NeurapolisRetriever,
    QualityPreset,
    RetrievedFile,
)

logger = logging.getLogger()


class RetrieverNode:
    def retrieve(state: State, runnable_config: GraphConfig) -> State:
        # logger.info(f"{self.__class__.__name__}: Started retrieving")

        async def _async_retrieve():
            last_ai_message: AIMessage = get_last_message_of_type(
                state.messages, AIMessage
            )
            tool_call = last_ai_message.tool_calls[0]
            date_filter = (
                None
                if tool_call["args"]["date_filter"] is None
                else DateFilter.model_validate(tool_call["args"]["date_filter"])
            )
            quality_preset = QualityPreset(tool_call["args"]["quality_preset"])

            retrieved_files: list[RetrievedFile] = []
            retriever = NeurapolisRetriever()
            async for x_event in retriever.retrieve(
                tool_call["args"]["query"],
                date_filter,
                quality_preset,
            ):
                if isinstance(x_event, LoaderUpdate):
                    await runnable_config["send_loader_update_to_client"](x_event)
                elif isinstance(x_event, list):
                    retrieved_files = x_event

            capped_retrieved_files = retrieved_files[:10]
            inner_xml = RetrievedFile.format_multiple_to_inner_llm_xml(
                capped_retrieved_files
            )
            xml = f"<{RetrievedFile.get_llm_xml_tag_name_prefix()}>\n{inner_xml}\n</{RetrievedFile.get_llm_xml_tag_name_prefix()}>"

            tool_message = ToolMessage(
                tool_call_id=tool_call["id"],
                content=xml,
                artifact=retrieved_files,
            )
            state.messages.append(tool_message)
            return state

        # Get the current event loop
        loop = asyncio.get_event_loop()

        # Run the coroutine and wait for its result
        future = asyncio.run_coroutine_threadsafe(_async_retrieve(), loop)
        result = future.result()

        # logger.info(f"{self.__class__.__name__}: Finished retrieving")
        return result
