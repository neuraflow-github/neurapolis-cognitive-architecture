import copy

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from neurapolis_cognitive_architecture.models import MyHumanMessage, State
from neurapolis_common import get_last_message_of_type

from .tools import retrieve


class RetrieverNode:
    async def retrieve(self, state: State, config: RunnableConfig) -> dict:
        last_my_human_message: MyHumanMessage = get_last_message_of_type(
            state["messages"], MyHumanMessage
        )
        last_ai_message: AIMessage = get_last_message_of_type(
            state["messages"], AIMessage
        )

        tool_call = last_ai_message.tool_calls[0]

        # Inject the necessary args
        copied_tool_call = copy.deepcopy(tool_call)
        copied_tool_call["args"]["date_filter"] = last_my_human_message.date_filter
        copied_tool_call["args"][
            "quality_preset"
        ] = last_my_human_message.quality_preset
        copied_tool_call["args"]["send_loader_update_to_client"] = config[
            "configurable"
        ]["send_loader_update_to_client"]

        tool_message: ToolMessage = await retrieve.ainvoke(copied_tool_call)

        return {"messages": [tool_message]}
