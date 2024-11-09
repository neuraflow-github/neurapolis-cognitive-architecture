from typing import Callable, TypedDict

from neurapolis_cognitive_architecture.models.my_ai_message import MyAiMessage


class GraphConfig(TypedDict):
    send_ai_message_to_client: Callable[[MyAiMessage], None]
