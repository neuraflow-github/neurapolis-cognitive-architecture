from typing import Callable

from langchain_core.runnables.config import RunnableConfig
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from neurapolis_cognitive_architecture.graph import graph_builder
from neurapolis_cognitive_architecture.models import (
    GraphConfig,
    MyAiMessage,
    MyHumanMessage,
    State,
)
from neurapolis_common import config as common_config
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

        runnable_config = RunnableConfig(
            configurable=GraphConfig(
                thread_id=thread_id,
                send_loader_update_to_client=send_loader_update_to_client,
            )
        )

        # PostgresSaver docs: https://langchain-ai.github.io/langgraph/how-tos/persistence_postgres
        async with AsyncPostgresSaver.from_conn_string(
            common_config.db_connection_string
        ) as async_postgres_saver:
            graph = graph_builder.compile(checkpointer=async_postgres_saver)
            result_state = await graph.ainvoke(state, runnable_config)

        print("result_state", result_state)

        last_ai_message: MyAiMessage = get_last_message_of_type(
            result_state.messages, MyAiMessage
        )
        send_ai_message_to_client(last_ai_message)
