from typing import Callable, Optional
from uuid import uuid4

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.runnables.config import RunnableConfig
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from neurapolis_cognitive_architecture.graph import graph_builder
from neurapolis_cognitive_architecture.models import (
    GraphConfig,
    MyAiMessage,
    MyHumanMessage,
    State,
)
from neurapolis_cognitive_architecture.utilities import (
    get_cognitive_architecture_config_by_quality_preset,
)
from neurapolis_common import config as common_config
from neurapolis_common import get_last_message_of_type
from neurapolis_retriever import LoaderUpdate, Reference


class NeurapolisCognitiveArchitecture:
    async def query(
        self,
        thread_id: str,
        human_message: MyHumanMessage,
        send_loader_update_to_client: Callable[[LoaderUpdate], None],
        send_ai_message_to_client: Callable[[MyAiMessage], None],
    ):
        messages: list[MyHumanMessage] = [human_message]

        cognitive_architecture_config = (
            get_cognitive_architecture_config_by_quality_preset(
                human_message.quality_preset
            )
        )

        state = State(config=cognitive_architecture_config, messages=messages)

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
            result_state: State = await graph.ainvoke(state, runnable_config)

        last_ai_message: AIMessage = get_last_message_of_type(
            result_state["messages"], AIMessage
        )

        last_ai_message_index = result_state["messages"].index(last_ai_message)
        previous_message: Optional[BaseMessage] = None
        if last_ai_message_index > 0:
            previous_message = result_state["messages"][last_ai_message_index - 1]

        references: list[Reference] = []
        if (
            isinstance(previous_message, ToolMessage)
            and previous_message.artifact is not None
        ):
            references = previous_message.artifact

        my_ai_message = MyAiMessage(
            id=str(uuid4()),
            content=last_ai_message.content,
            references=references,
        )
        await send_ai_message_to_client(my_ai_message)
