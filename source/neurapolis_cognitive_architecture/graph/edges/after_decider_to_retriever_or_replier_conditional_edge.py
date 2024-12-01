from typing import Literal

from langchain_core.messages import AIMessage, ToolCall
from neurapolis_cognitive_architecture.enums import GraphStep
from neurapolis_cognitive_architecture.models import State
from neurapolis_common import get_last_message_of_type


def after_decider_to_retriever_or_replier_conditional_edge(
    state: State,
) -> Literal["RETRIEVER", "REPLIER"]:
    """
    This edge checks if there is a tool call after the last human message. If there is and it is a retriever tool call, it directs to the retriever node, otherwise to the replier node.
    """
    last_ai_message: AIMessage = get_last_message_of_type(state.messages, AIMessage)

    if len(last_ai_message.tool_calls) > 0:
        tool_call: ToolCall = last_ai_message.tool_calls[0]
        if tool_call["name"] == GraphStep.RETRIEVER.value:
            return GraphStep.RETRIEVER.value

    return GraphStep.REPLIER.value
