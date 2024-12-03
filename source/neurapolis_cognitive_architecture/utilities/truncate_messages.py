from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
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

    reversed_other_messages = list(reversed(messages[1:]))

    llm = AzureChatOpenAI(
        azure_endpoint=common_config.azure_openai_endpoint,
        api_version=common_config.openai_api_version,
        api_key=common_config.azure_openai_api_key,
        azure_deployment="gpt-4o",
        model="gpt-4o",  # Needed for token counting
    )

    truncated_messages: list[BaseMessage] = [system_message]
    large_context_message_count = 0
    for x_message_index, x_message in enumerate(reversed_other_messages):
        if isinstance(x_message, ToolMessage):
            # The current message is a tool result tool message. We only want to add up to 3 of the large tool result tool messages
            if large_context_message_count >= 3:
                continue

            next_message = (
                reversed_other_messages[x_message_index + 1]
                if x_message_index < len(reversed_other_messages) - 1
                else None
            )  # This is the next message we would work on. So it is an older message in the chat history than the current message. We need it to check if the tool result tool message fits in together with its tool call ai message

            # Check that the current tool result tool message fits in together with its tool calling ai message
            tool_result_tool_message_and_tool_calling_ai_message = [
                next_message,
                x_message,
            ]
            candidate_truncated_messages = (
                [truncated_messages[0]]
                + tool_result_tool_message_and_tool_calling_ai_message
                + truncated_messages[1:]
            )
            token_count = llm.get_num_tokens_from_messages(candidate_truncated_messages)

            if token_count <= token_limit:
                # Only add the tool result tool message. The next iteration of the loop will add the tool calling ai message. That one will fit anyways, because we already checked above, that both will fit in
                truncated_messages.insert(1, x_message)
                large_context_message_count += 1
            else:
                # The tool result tool message and its tool calling ai message will not fit in the token limit. Do not add those messages and after this do add anymore tool result or tool calling ai messages. Here we only skip the current tool result tool message. The next iteration, will work on the tool calling ai message and will check if we added the tool result tool message before or not. Depending on that it will either add the tool calling ai message or skip it too.
                large_context_message_count = 10

        else:
            # The current message is a human or ai message
            if isinstance(x_message, AIMessage) and len(x_message.tool_calls) > 0:
                # The current message is a tool calling ai message.
                # Check that the last message we added to the truncated messages is a tool result tool message before we add a tool calling ai message. Maybe we did not add the tool result tool message because of space or limit, so we should also not add the tool calling ai message
                if len(truncated_messages) == 0 or not isinstance(
                    truncated_messages[1], ToolMessage
                ):
                    continue

            candidate_truncated_messages = (
                [truncated_messages[0]] + [x_message] + truncated_messages[1:]
            )
            token_count = llm.get_num_tokens_from_messages(candidate_truncated_messages)

            if token_count <= token_limit:
                truncated_messages.insert(1, x_message)
            else:
                break

    new_chat_prompt_value = ChatPromptValue(messages=truncated_messages)

    # for x_message in truncated_messages:
    #     print(f"{x_message.type}: {x_message.content[:100]}")

    return new_chat_prompt_value
