import logging

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables.config import RunnableConfig
from neurapolis_cognitive_architecture.models import State
from neurapolis_common import get_last_message_of_type
from neurapolis_retriever import (
    DateFilter,
    LoaderUpdate,
    NeurapolisRetriever,
    QualityPreset,
    RetrievedFile,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


class RetrieverNode:
    async def retrieve(self, state: State, config: RunnableConfig) -> State:
        logger.info(f"{self.__class__.__name__}: Started retrieving")

        last_ai_message: AIMessage = get_last_message_of_type(state.messages, AIMessage)
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
                config["configurable"]["send_loader_update_to_client"](
                    x_event
                )  # Do not await to not block
            elif isinstance(x_event, list):
                retrieved_files = x_event

        # retrieved_file_datas: list[dict] = []
        # for x_retrieved_file in retrieved_files:
        #     retrieved_file_datas.append(x_retrieved_file.model_dump())

        capped_retrieved_files = retrieved_files[:10]
        inner_xml = RetrievedFile.format_multiple_to_inner_llm_xml(
            capped_retrieved_files
        )
        xml = f"<{RetrievedFile.get_llm_xml_tag_name_prefix()}>\n{inner_xml}\n</{RetrievedFile.get_llm_xml_tag_name_prefix()}>"

        tool_message = ToolMessage(
            xml,
            tool_call_id=tool_call["id"],
            # artifact=retrieved_files,
        )
        state.messages.append(tool_message)

        logger.info(f"{self.__class__.__name__}: Finished retrieving")

        return state
