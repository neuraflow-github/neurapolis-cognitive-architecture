from typing import Any, Callable, TypedDict

from .my_ai_message import MyAiMessage


class GraphConfig(TypedDict):
    send_ai_message_to_client: Callable[[MyAiMessage], None]
    configurable: dict[str, Any]
