from typing import Callable

from neurapolis_cognitive_architecture.graph.graph import graph
from neurapolis_cognitive_architecture.models.graph_config import GraphConfig
from neurapolis_cognitive_architecture.models.my_ai_message import MyAiMessage
from neurapolis_cognitive_architecture.models.my_human_message import MyHumanMessage
from neurapolis_cognitive_architecture.models.state import State
from neurapolis_cognitive_architecture.utilities.get_last_message_of_type import (
    get_last_message_of_type,
)
from neurapolis_retriever.models.loader_update import LoaderUpdate

# TODO how to pass through config with datefilter and quality preset, or get access to it in tool via last user message
# TODO Fix all dtos
# TODO what about state naming or event naming
# TODO threads
# TODO Put tags into the response, render those or catch events in markdown viewer, then use to go to or tag the file hit cards
# TODO Clean up file hits like in zeitkapsel + multi highlight
# TODO clean up messages


class NeurapolisCognitiveArchitecture:
    async def query(
        self,
        thread_id: str,
        user_message: MyHumanMessage,
        send_loader_update_to_client: Callable[[LoaderUpdate], None],
        send_ai_message_to_client: Callable[[MyAiMessage], None],
    ):
        # print("thread_id", thread_id)

        messages = [user_message]
        state = State(messages=messages)
        graph_config = GraphConfig(
            send_loader_update_to_client=send_loader_update_to_client
        )
        result_state = graph.invoke(state, graph_config)
        last_ai_message = get_last_message_of_type(result_state.messages, MyAiMessage)
        send_ai_message_to_client(last_ai_message)
