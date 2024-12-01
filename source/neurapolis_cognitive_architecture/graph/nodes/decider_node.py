import logging
from datetime import datetime
from operator import itemgetter
from typing import Optional
from uuid import uuid4

from langchain_aws import ChatBedrock
from langchain_core.messages import AIMessage, ToolCall
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable, RunnableLambda
from neurapolis_cognitive_architecture.config import config
from neurapolis_cognitive_architecture.enums import GraphStep
from neurapolis_cognitive_architecture.models import MyHumanMessage, State
from neurapolis_cognitive_architecture.utilities import truncate_messages
from neurapolis_common import UserMetadata
from neurapolis_common import config as common_config
from neurapolis_common import get_last_message_of_type
from pydantic import BaseModel, Field

logger = logging.getLogger()


class DeciderLlmDataModel(BaseModel):
    explanation: str = Field(
        description="Eine kurze (1 Satz) Erklärung, warum nachgeschlagen werden sollte oder nicht."
    )
    should_use_retriever: bool = Field(
        description="Soll das Nachschlagetool verwendet werden?"
    )
    query: Optional[str] = Field(description="Die Suchanfrage an das Nachschlagetool")


class DeciderNode:
    _chain: Runnable

    def __init__(self):
        prompt_template_string = """Generell:

- Du bist Teil einer Retrieval Augmented Generation Anwendung. Diese besteht aus einem KI-Agenten, welcher aus mehreren LLM-Modulen besteht, welche zusammenarbeiten, um Nutzeranfragen zum Rats Informationssystem (RIS) zu beantworten.
- Das RIS ist ein internes System für Politiker und städtische Mitarbeiter, das ihnen bei ihrer Arbeit hilft. Es ist eine Datenbank, welche Informationen einer bestimmten Stadt über Organisationen, Personen, Sitzungen, Dateien usw. enthält.
- Ein menschlicher Mitarbeiter kommt zu dem KI-Agenten mit einer Frage, dessen Antworten sich in der Datenbank verstecken und ihr müsst die Frage so gut wie möglich beantworten.
- Zur einfachen Durchsuchbarkeit wurden viele Daten durch ein Embeddingmodel als Vektoren embedded.


Aufgabe:

- Du bist der "Decider"-Mitarbeiter in dem KI-Agenten.
- Deine Aufgabe ist es, zu entscheiden, ob für die Nutzeranfrage das Nachschlagetool verwendet werden muss.
- Der Standardfall ist, dass nachgeschlagen werden muss, da die meisten Nutzeranfragen auf Informationen abzielen, welche sich im RIS befinden.
- Der "Replier"-Mitarbeiter, welcher am Ende die Nutzeranfrage beantwortet, darf nur nachgeschlagene Informationen aus dem RIS und Informationen, welche sich schon im Chatverlauf befinden, verwenden.
- Es muss nur nicht nachschlagen werden,
    - wenn es sich um eine simple Konversationsfrage wie "Hallo, wie geht es dir?" oder "Welchen Tag haben wir heute?" handelt.
    - für Zusammenfassungen oder Ähnlichem von Informationen, welche sich schon im Chatverlauf befinden.
- Die Suchanfrage an das Nachschlagetool muss in deutsch sein, auch wenn die Nutzeranfrage in einer anderen Sprache ist.
- Die Suchanfrage sollte außerdem alle Informationen aus der Nutzeranfrage enthalten, aber prägnant sein. Wenn der Nutzer z. B. Dinge doppelt und dreifach beschreibt, dann sollte die Suchanfrage die Information nur einmal enthalten.


Beispiele:

- Beispiel 1:
    - Nutzeranfrage: "Was ist die Adresse der Stadthalle?"
    - Antwort:
        - should_use_retriever: True
        - query: "Wie lautet die Adresse der Stadthalle?"
- Beispiel 2:
    - Nutzeranfrage: "Hallo, wie geht es dir?"
    - Antwort:
        - should_use_retriever: False
        - query: None
- Beispiel 3:
    - Nutzeranfrage: "Gebe mir die Baugenehmigung für den Bau des Krankenhauses an der Königstraße. An der König"
    - Antwort:
        - should_use_retriever: True
        - query: "Baugenehmigung für den Bau des Krankenhauses an der Königstraße"

        
Aktuelles Datum und Uhrzeit: {formatted_current_datetime}


Nutzer Metadaten:

<Nutzer Metadaten>
{user_metadata}
</Nutzer Metadaten>"""
        system_message = ChatPromptTemplate.from_template(prompt_template_string)
        chat_prompt_template = ChatPromptTemplate(
            [
                system_message,
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        llm = ChatBedrock(
            aws_access_key_id=common_config.aws_access_key_id,
            aws_secret_access_key=common_config.aws_secret_access_key,
            region=common_config.aws_region,
            model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
            temperature=0,
            # timeout=120,  # 2 minutes
        )
        structured_llm = llm.with_structured_output(DeciderLlmDataModel)
        self._chain = (
            {
                "formatted_current_datetime": lambda x: datetime.now().strftime(
                    "%d.%m.%Y %H:%M"
                ),
                "user_metadata": lambda x: x["user_metadata"].format_to_inner_llm_xml(),
                "messages": itemgetter("messages"),
            }
            | chat_prompt_template
            | RunnableLambda(
                lambda x: truncate_messages(
                    x, token_limit=config.context_window_token_limit
                )
            )
            | structured_llm
        )

    def decide(self, state: State) -> State:
        logger.info(f"{self.__class__.__name__}: Started deciding")

        last_human_message: MyHumanMessage = get_last_message_of_type(
            state.messages, MyHumanMessage
        )

        decider_llm_data_model: DeciderLlmDataModel = self._chain.invoke(
            {
                "user_metadata": UserMetadata(
                    city_name="Freiburg",
                    user_name="Lorem Ipsum",
                ),
                "messages": state.messages,
            }
        )

        # For safety check also that the query is not None
        if (
            decider_llm_data_model.should_use_retriever
            and decider_llm_data_model.query is not None
        ):
            tool_call = ToolCall(
                id=str(uuid4()),
                name=GraphStep.RETRIEVER.value,
                args={
                    "query": decider_llm_data_model.query,
                    "date_filter": last_human_message.date_filter,
                    "quality_preset": last_human_message.quality_preset,
                },
            )
            ai_message = AIMessage(
                "Calling retriever tool",
                tool_calls=[tool_call],
            )
            state.messages.append(ai_message)

        logger.info(f"{self.__class__.__name__}: Finished deciding")

        return state
