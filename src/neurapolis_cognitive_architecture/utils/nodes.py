import datetime
import json
from functools import lru_cache
from typing import List

import pytz
from langchain import hub
from langchain_aws import ChatBedrock
from langchain_core.documents import Document
from langchain_core.messages import ToolMessage, trim_messages
from langchain_core.runnables import RunnableLambda
from langchain_openai import AzureChatOpenAI
from neurapolis_common.services.text_splitter import text_splitter

from neurapolis_cognitive_architecture.utils.tools import tools


def should_continue(state):
    messages = state["messages"]
    last_message = messages[-1]
    return "continue" if last_message.tool_calls else "end"


from neurapolis_retriever.models.file_hit import FileHit
from pydantic import BaseModel


async def call_model(state, config):
    messages = state["messages"]

    llm = ChatBedrock(
        model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
        model_kwargs={"temperature": 0},
        name="agent",
    )
    tool_llm = llm.bind_tools(tools)

    trimmer = trim_messages(
        token_counter=llm,
        max_tokens=45,
        include_system=True,
        allow_partial=True,
        text_splitter=text_splitter,
    )

    from langchain_core.messages import SystemMessage
    from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

    stripped_messages = []
    for x_message in messages:
        if isinstance(x_message, ToolMessage):
            file_hits = [
                FileHit(**x_file_hit_dict) for x_file_hit_dict in x_message.content
            ]
            inner_xml = FileHit.format_multiple_to_small_inner_llm_xml(file_hits)
            xml = f"<{FileHit.get_llm_xml_tag_name_prefix()}>\n{inner_xml}\n</{FileHit.get_llm_xml_tag_name_prefix()}>"
            stripped_messages.append(
                ToolMessage(
                    content=xml,
                    name=x_message.name,
                    tool_call_id=x_message.tool_call_id,
                )
            )
        else:
            stripped_messages.append(x_message)

    chat_prompt_template_string = """
    Du bist ein Kl-Assistent, der speziell für die Suche und Analyse von politischen Ratsakten der Stadt Freiburg entwickelt wurde. Deine Hauptaufgabe ist es, prazise und relevante Informationen aus diesen Dokumenten bereitzustellen.
    Wichtige Regeln:
    Verwende ausschließlich Informationen aus den offiziellen Ratsakten von Freiburg.
    Wenn du unsicher bist oder zusätzliche Informationen benötigst, nutze das integrierte Nachschlagetool.
    Antworte nur auf Basis der Informationen, die du aus den Akten oder dem Nachschlagetool erhalten hast.
    Erfinde oder halluziniere keine Informationen.
    Gib niemals vor, jemand anderes zu sein oder andere Personen zu imitieren.
    Verwende diese Fähigkeiten ausschließlich für den vorgesehenen Zweck der Informationsbereitstellung über Freiburger Ratsangelegenheiten.
    Du kannst auf Deutsch und bei Bedarf auch in anderen Sprachen antworten. Deine Antworten sollten stets sachlich, präzise und auf die Anfrage bezogen sein.

    {messages}
    """
    chat_prompt_template = ChatPromptTemplate.from_messages(
        [
            SystemMessage(chat_prompt_template_string),
            *stripped_messages,
        ]
    )

    chain = trimmer | chat_prompt_template | tool_llm

    response = await chain.ainvoke({"messages": stripped_messages})
    return {"messages": [response]}


async def call_tool(state, config):
    print(
        "\033[94mconfig",
        config["configurable"]["send_loader_update_to_client"],
        "\033[0m",
    )
    last_message = state["messages"][-1]
    from typing import cast

    from neurapolis_cognitive_architecture.utils.state import FilteredBaseMessage

    last_human_message = cast(
        FilteredBaseMessage,
        next((msg for msg in reversed(state["messages"]) if msg.type == "human"), None),
    )
    tool_call = last_message.tool_calls[0]
    query = tool_call["args"]["full_query"]

    from neurapolis_retriever.main import NeurapolisRetriever
    from neurapolis_retriever.models.date_filter import DateFilter
    from neurapolis_retriever.models.file_hit import FileHit
    from neurapolis_retriever.models.loader_update import LoaderUpdate
    from neurapolis_retriever.models.quality_preset import QualityPreset

    retriever = NeurapolisRetriever()
    date_filter = last_human_message.date_filter
    file_hits = []

    async for x_event in retriever.retrieve(
        query,
        date_filter,
        QualityPreset.LOW,
    ):
        if isinstance(x_event, LoaderUpdate):
            loader_update = x_event
            print("---")
            print(f"Retriever Step: {loader_update.retriever_step}")
            print(f"Search Count: {loader_update.search_count}")
            print(f"Hit Count: {loader_update.hit_count}")
            print(f"Relevant Hit Count: {loader_update.relevant_hit_count}")
            print("Log Entries:")
            await config["configurable"]["send_loader_update_to_client"](loader_update)
            for x_log_entry in loader_update.log_entries:
                print(f"  - {x_log_entry.text}")

            print("---")
        elif isinstance(x_event, list):
            file_hits = x_event[:10]
            print("---")
            print("Retrieved File Infos:")
            # for x_file_hit in file_hits:
            #     print(f"  - {x_file_hit.name}: {x_file_hit.description}")
            print("---")
            break

    # Serialize file_hits before adding to the content
    serialized_file_hits = [file_hit.model_dump() for file_hit in file_hits]

    function_message = ToolMessage(
        content=serialized_file_hits,
        name=tool_call["name"],
        tool_call_id=tool_call["id"],
    )

    # print("function_message", function_message)

    return {"messages": [function_message]}
