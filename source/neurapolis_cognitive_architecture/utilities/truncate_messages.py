from langchain_core.messages import BaseMessage, ToolMessage
from langchain_core.prompt_values import ChatPromptValue
from langchain_openai import AzureChatOpenAI
from neurapolis_common import config as common_config


def truncate_messages(
    chat_prompt_value: ChatPromptValue, token_limit: int
) -> ChatPromptValue:
    """
    Truncates messages to fit within a token limit. It starts with the system message and then goes from the newest message to the oldest. It tries to stuff as many messages as possible into the token limit. When it can not stuff a tool message anymore it skips it. But when it can not stuff a human or ai message anymore it stops.
    """

    messages = chat_prompt_value.to_messages()

    system_message = messages[0]

    reversed_other_messages = reversed(messages[1:])

    truncated_messages: list[BaseMessage] = [system_message]
    llm = AzureChatOpenAI(
        azure_endpoint=common_config.azure_openai_endpoint,
        api_version=common_config.openai_api_version,
        api_key=common_config.azure_openai_api_key,
        azure_deployment="gpt-4o",
        model="gpt-4o",  # Needed for token counting
    )
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

    new_chat_prompt_value = ChatPromptValue(messages=truncated_messages)

    return new_chat_prompt_value
