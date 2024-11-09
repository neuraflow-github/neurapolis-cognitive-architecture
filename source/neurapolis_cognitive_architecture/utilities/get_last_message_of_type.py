from typing import Optional

from langchain_core.messages import BaseMessage


def get_last_message_of_type(
    messages: list[BaseMessage], type: type[BaseMessage]
) -> Optional[BaseMessage]:
    for x_message in reversed(messages):
        if not isinstance(x_message, type):
            continue

        return x_message
    return None
