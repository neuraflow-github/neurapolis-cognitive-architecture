from langchain_aws import ChatBedrock
from langchain_core.messages import BaseMessage, ToolMessage


def truncate_messages(
    messages: list[BaseMessage], token_limit: int
) -> list[BaseMessage]:
    """
    Truncates messages to fit within a token limit. It starts with the system message and then goes from the newest message to the oldest. It tries to stuff as many messages as possible into the token limit. When it can not stuff a tool message anymore it skips it. But when it can not stuff a human or ai message anymore it stops.
    """

    system_message = messages[0]

    reversed_other_messages = reversed(messages[1:])

    truncated_messages: list[BaseMessage] = [system_message]
    llm = ChatBedrock()
    for x_message in reversed_other_messages:
        candidate_messages = truncated_messages + [x_message]
        token_count = llm.get_num_tokens_from_messages(candidate_messages)

        if token_count <= token_limit:
            truncated_messages.append(x_message)
        else:
            if isinstance(x_message, ToolMessage):
                # Skip the tool message
                continue
            else:
                break

    return truncated_messages
