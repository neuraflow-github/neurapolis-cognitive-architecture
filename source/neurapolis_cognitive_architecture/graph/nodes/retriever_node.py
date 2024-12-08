import copy
import logging

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from neurapolis_cognitive_architecture.models import MyHumanMessage, State
from neurapolis_common import get_last_message_of_type

from .tool_node import retrieve

logger = logging.getLogger()


class RetrieverNode:
    async def retrieve(self, state: State, config: RunnableConfig) -> dict:
        logger.info(f"{self.__class__.__name__}: Started")

        last_my_human_message = get_last_message_of_type(
            state["messages"], MyHumanMessage
        )
        last_ai_message = get_last_message_of_type(state["messages"], AIMessage)
        tool_call = last_ai_message.tool_calls[0]

        xml, references = await retrieve(
            tool_call["args"]["query"],
            last_my_human_message.date_filter,
            last_my_human_message.quality_preset,
            config["configurable"]["send_loader_update_to_client"],
        )

        tool_message = ToolMessage(
            xml, artifact=references, tool_call_id=tool_call["id"]
        )
        logger.info(f"{self.__class__.__name__}: Finished")
        return {"messages": [tool_message]}
