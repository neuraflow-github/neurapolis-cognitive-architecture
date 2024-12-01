from typing import Callable

from neurapolis_cognitive_architecture.graph import graph
from neurapolis_cognitive_architecture.models import (
    GraphConfig,
    MyAiMessage,
    MyHumanMessage,
    State,
)
from neurapolis_common import get_last_message_of_type
from neurapolis_retriever import LoaderUpdate


class NeurapolisCognitiveArchitecture:
    async def query(
        self,
        thread_id: str,
        human_message: MyHumanMessage,
        send_loader_update_to_client: Callable[[LoaderUpdate], None],
        send_ai_message_to_client: Callable[[MyAiMessage], None],
    ):
        messages: list[MyHumanMessage] = [human_message]
        state = State(messages=messages)

        graph_config = GraphConfig(
            send_loader_update_to_client=send_loader_update_to_client,
            configurable={"thread_id": thread_id},
        )

        result_state = graph.invoke(state, graph_config)

        print("result_state", result_state)

        last_ai_message: MyAiMessage = get_last_message_of_type(
            result_state.messages, MyAiMessage
        )
        send_ai_message_to_client(last_ai_message)
