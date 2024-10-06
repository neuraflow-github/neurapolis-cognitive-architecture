from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tracers import LangChainTracer
from langchain_core.tracers.run_collector import RunCollectorCallbackHandler
from langsmith import Client
import streamlit as st
from streamlit_feedback import streamlit_feedback
import time
import uuid
from langserve import RemoteRunnable

from langchain import callbacks
import asyncio

import os
from dotenv import load_dotenv

load_dotenv()

# Import secrets from environment variables
LANGCHAIN_API_KEY = os.environ.get("LANGCHAIN_API_KEY")
LANGCHAIN_ENDPOINT = os.environ.get("LANGCHAIN_ENDPOINT")
LANGCHAIN_PROJECT = os.environ.get("LANGCHAIN_PROJECT")
CHAIN_URL = os.environ.get("CHAIN_URL")

st.set_page_config(page_title="Ratsinformationssystem", page_icon="🏛️")
st.title("Ratsinformationssystem")

langchain_api_key = LANGCHAIN_API_KEY
project = LANGCHAIN_PROJECT

langchain_endpoint = "https://eu.api.smith.langchain.com"
client = Client(api_url=langchain_endpoint, api_key=langchain_api_key)
ls_tracer = LangChainTracer(project_name=project, client=client)
run_collector = RunCollectorCallbackHandler()

if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = str(uuid.uuid4())

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(
            content="Willkommen beim Ratsinformationssystem! Wie kann ich Ihnen bei Ihrer Suche nach Informationen über städtische Verwaltungsaktivitäten, Entscheidungen oder damit verbundene Einrichtungen helfen?"
        ),
    ]

cfg = RunnableConfig(
    {
        "configurable": {
            "session_id": st.session_state["session_id"],
            "thread_id": st.session_state["thread_id"],
        }
    }
)
cfg["callbacks"] = [ls_tracer, run_collector]


# Display chat history
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.write(message.content)

# User input
if input := st.chat_input(
    placeholder="z.B. 'Welche Beschlüsse wurden zur Stadtentwicklung in den letzten 5 Jahren gefasst?'"
):
    st.session_state.chat_history.append(HumanMessage(content=input))
    st.chat_message("user").write(input)

    with st.chat_message("assistant"):
        with callbacks.collect_runs() as cb:
            response_placeholder = st.empty()
            full_response = ""

            remote = RemoteRunnable(CHAIN_URL)

            async def process_events(full_response):
                async for event in remote.astream_events(
                    {
                        "messages": st.session_state.chat_history,
                    },
                    cfg,
                    version="v1",
                ):
                    kind = event["event"]
                    if kind == "on_chat_model_stream" and event["name"] == "agent":
                        chunk = event["data"]["chunk"]
                        if isinstance(chunk.content, list):
                            for item in chunk.content:
                                if (
                                    isinstance(item, dict)
                                    and item.get("type") == "text"
                                ):
                                    content = item["text"]
                                    full_response += content
                                    response_placeholder.markdown(full_response + "▌")
                        else:
                            full_response += chunk.content
                            response_placeholder.markdown(full_response + "▌")
                    elif kind == "on_chat_model_end":
                        full_response += "\n"
                        response_placeholder.markdown(full_response + "▌")
                    elif kind == "on_tool_start":
                        tool_info = f"\nStarting tool: {event['name']} with inputs: {event['data'].get('input')}\n"
                        full_response += tool_info
                        response_placeholder.markdown(full_response + "▌")
                    elif kind == "on_tool_end":
                        tool_output = f"\nTool {event['name']} finished. Output: {event['data'].get('output')}\n"
                        full_response += tool_output
                        response_placeholder.markdown(full_response + "▌")
                return full_response

            full_response = asyncio.run(process_events(full_response))
            response_placeholder.markdown(full_response)
        if cb.traced_runs:
            st.session_state["last_run"] = cb.traced_runs[0].id

    st.session_state.chat_history.append(AIMessage(content=full_response))


@st.cache_data(ttl="2h", show_spinner=False)
def get_run_url(run_id):
    time.sleep(1)
    return client.read_run(run_id).url


if st.session_state.get("last_run"):
    run_url = get_run_url(st.session_state.last_run)

    feedback = streamlit_feedback(
        feedback_type="faces",
        optional_text_label="[Optional] Bitte erläutern Sie Ihre Bewertung:",
        key=f"feedback_{st.session_state.last_run}",
    )

    if feedback:
        scores = {"😀": 1, "🙂": 0.75, "😐": 0.5, "🙁": 0.25, "😞": 0}
        client.create_feedback(
            st.session_state.last_run,
            feedback["type"],
            score=scores[feedback["score"]],
            comment=feedback.get("text", None),
        )
        st.toast("Vielen Dank für Ihr Feedback zur Informationsqualität!", icon="📊")
