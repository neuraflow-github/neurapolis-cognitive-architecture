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
from neurapolis_cognitive_architecture.models import MyHumanMessage
from neurapolis_cognitive_architecture.utilities import truncate_messages
from neurapolis_common import UserMetadata
from neurapolis_common import config as common_config
from neurapolis_common import get_last_message_of_type
from neurapolis_retriever.models import State
from pydantic import BaseModel, Field

logger = logging.getLogger()


class DeciderLlmDataModel(BaseModel):
    should_use_retriever: bool = Field(
        description="Gibt an, ob der Retriever verwendet werden soll"
    )
    query: Optional[str] = Field(description="Die Anfrage an den Retriever")


class DeciderNode:
    _chain: Runnable

    def __init__(self):
        # TODO: Prompt, Trim of the tool messages first.
        prompt_template_string = """Generell:

- Du bist Teil einer Retrieval Augmented Generation Anwendung. Diese besteht aus einem KI-Agenten, welcher aus mehreren LLM-Modulen besteht, welche zusammenarbeiten, um Nutzeranfragen zum Rats Informationssystem (RIS) zu beantworten.
- Das RIS ist ein internes System für Politiker und städtische Mitarbeiter, das ihnen bei ihrer Arbeit hilft. Es ist eine Datenbank, welche Informationen einer bestimmten Stadt über Organisationen, Personen, Sitzungen, Dateien usw. enthält.
- Ein menschlicher Mitarbeiter kommt zu dem KI-Agenten mit einer Frage, dessen Antworten sich in der Datenbank verstecken und ihr müsst die Frage so gut wie möglich beantworten.
- Zur einfachen Durchsuchbarkeit wurden viele Daten durch ein Embeddingmodel als Vektoren embedded.


Aufgabe:

- Du bist der "Planner"-Mitarbeiter in dem KI-Agenten.
- Du bist der erste, der die Nutzeranfrage verarbeitet.
- Eine Nutzeranfrage kann auf mehrere Themenbereiche abzielen und wenn der "Retriever"-Mitarbeiter eine Suche mit einem gemischten Themenbereich auf der Datenbank durchführt, bekommt er sehr gemischte Treffer, wodurch die Nutzeranfragen nicht gut beantwortet werden kann.
- Deine Aufgabe ist es, die Nutzeranfrage in spezifischere, genauere, abzielendere Suchanfragen zu konvertieren.
- Diese sollten zwar spezifisch sein, aber trotzdem noch einen sehr guten Detailgrad haben und ganze Sätz sein. Sehr kurze oder allgemeine Sätze führen dazu, dass der "Retriever"-Mitarbeiter sehr viele allgemeine Treffer bekommt, wodurch die Antwort auf die Nutzeranfrage dann sehr schwammig wird, oder oft die gleiche ist, weil oft eher die allgemeinen Dokumente aus der Datenbank zurückgegeben werden.
- Die Suchanfragen können auch auf bestimmte Keywords wie Aktenzeichen, Referenzen, Nummern, Namen, Orte, Straßennamen, Gebäudenamen, Firmennamen, etc. abzielen. Keywords sollen aber auf keinen Fall sehr allgemeine, geläufige Wörter sein (z. B. "Zeitungsarchiv") und auch nicht nur ein Datum.
- Gebe mindestens zwei Suchanfragen an und maximal {planner_search_limit}.
- Deine Antwort geht an den "Retriever"-Mitarbeiter, welcher dann mit den Suchanfragen Dokumente aus der Datenbank holt.
- Du wirst für die gleiche Nutzeranfrage mehrmals durchlaufen, damit du mehrere Chancen hast, die perfekten Suchanfragen zu finden.
- Damit du weißt wie du dich verbessern kannst, wird dir vom "Relevance-Grader"-Mitarbeiter Feedback gegeben, ob die Suchanfragen, die du vorgeschlagen hast, relevante Artikel hervorgebracht haben oder nicht.
- Wenn du Feedback bekommst, versuche nochmal für jeden irrelevanten Themenbereich eine KOMPLETT NEUE Suchanfrage zu finden.


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
            timeout=120,  # 2 minutes
        )
        structured_llm = llm.with_structured_output(DeciderLlmDataModel)
        self._chain = (
            {
                "formatted_current_datetime": lambda x: datetime.now().strftime(
                    "%d.%m.%Y %H:%M"
                ),
                "user_metadata": lambda x: x.user_metadata.format_to_inner_llm_xml(),
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
                "query": last_human_message.content,
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
                "Toolcall", tool_calls=[tool_call], content=decider_llm_data_model.query
            )
            state.messages.append(ai_message)

        logger.info(f"{self.__class__.__name__}: Finished deciding")

        return state
