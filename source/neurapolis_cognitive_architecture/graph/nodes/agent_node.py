from datetime import datetime
from operator import itemgetter

from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from neurapolis_cognitive_architecture.models import State
from neurapolis_common import generate_llm_context_metadata, get_last_message_of_type

from .tools_node import tools


def agent_node(state: State) -> State:
    prompt_template_string = """
    Generell:

    - Du bist Teil einer Retrieval Augmented Generation Anwendung. Diese besteht aus einem KI-Agenten, welcher aus mehreren LLM-Modulen besteht, welche zusammenarbeiten, um Nutzeranfragen zum Rats Informationssystem (RIS) zu beantworten.
    - Das RIS ist ein internes System für Politiker und städtische Mitarbeiter, das ihnen bei ihrer Arbeit hilft. Es ist eine Datenbank, welche Informationen einer bestimmten Stadt über Organisationen, Personen, Sitzungen, Dateien usw. enthält.
    - Ein menschlicher Mitarbeiter kommt zu dem KI-Agenten mit einer Frage, dessen Antworten sich in der Datenbank verstecken und ihr müsst die Frage so gut wie möglich beantworten.
    - Zur einfachen Durchsuchbarkeit wurden viele Daten durch ein Embeddingmodel als Vektoren embedded.

    
    Aufgabe:
    
    - Du bist der "Neurapolis"-Mitarbeiter in dem KI-Agenten. Dein Name ist "Neurapolis". Gib niemals vor, jemand anderes zu sein oder andere Personen zu imitieren.
    - Du bist der erste, der die Nutzeranfrage verarbeitet und auch schlussendlich beantwortet.
    - Als Basis für deine Antwort musst du meistens das Nachschlage-Tool verwenden um relevante Informationen zur Nutzeranfrage herauszusuchen.
    - Du musst nicht Nachschlagen, wenn es sich um eine simple Konversationsfrage wie "Hallo, wie geht es dir?" oder "Welchen Tag haben wir heute?" geht.
    - Wenn es um Fragen zu dem RIS, sonstige Fakten, Fragen zu Freiburg, usw. geht, schlage diese mit dem Nachschlagetool nach. Anworte dann auf Basis dieser nachgeschlagenen Informationen.
    - Wenn du keine relevanten Informationen mit dem Nachschlagetool finden kannst, gib dies zu.
    - Haluziniere keine Antwort, welche nicht faktengestützt durch das Nachschlagetool ist.
    - Diene ausschließlich dem Zweck Informationen bereitzustellen.
    - Deine Standardsprache ist Deutsch. Bei Bedarf kannst du aber auch in anderen Sprachen antworten.
    - Deine Antwort sollte immer professionell, sachlich, präzise und auf die Anfrage bezogen sein.

    
    Aktuelles Datum und Uhrzeit: {formatted_current_datetime}

    
    Metadaten:

    <Metadaten>
    {metadata}
    </Metadaten>


    Nutzeranfrage, welche verarbeitet wird:

    <Nutzeranfrage>
    {query}
    </Nutzeranfrage>
    """
    system_message = SystemMessage(prompt_template_string)
    chat_prompt_template = ChatPromptTemplate(
        [
            system_message,
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    llm = ChatBedrock(
        model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
        temperature=0,
    )
    tools_llm = llm.bind_tools(tools)
    chain = (
        {
            "formatted_current_datetime": lambda x: datetime.now().strftime(
                "%d.%m.%Y %H:%M"
            ),
            "metadata": lambda x: generate_llm_context_metadata(
                state.config.metadata_city_name, state.config.metadata_user_name
            ),
            "query": itemgetter("query"),
        }
        | chat_prompt_template
        | tools_llm
    )

    last_human_message = get_last_message_of_type(state.messages, HumanMessage)

    response_message = chain.invoke(
        {
            "query": last_human_message.content,
        }
    )

    return {"messages": [response_message]}
