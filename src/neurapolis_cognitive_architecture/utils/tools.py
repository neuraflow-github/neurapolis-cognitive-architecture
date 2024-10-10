import os
from typing import Optional, Type

from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.vectorstores import Neo4jVector
from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
from langchain_openai import OpenAIEmbeddings
from pydantic import BaseModel, Field

# Load environment variables from .env file
load_dotenv()


class RetrieverInput(BaseModel):
    # TODO What is urgency?
    full_query: str = Field(
        description="""When using get_information, create a complete sentence that captures the user's request in the context of a city council's Retrieval Augmented Generation system. Use the same language as the input. Examples:

"What are the current waste management policies in the city?"
"How can residents participate in the next city council meeting?"
"What are the zoning regulations for new construction projects?"
"What initiatives are in place to improve public transportation?"

This approach ensures context-aware, specific queries related to city council matters without redundant mentions of the city name.

Additionally, include the following information in your query if applicable:
- Topic: Specify the main topic or category (e.g., 'waste management', 'public participation', 'zoning', 'transportation')
- Urgency: Indicate the urgency of the query (e.g., 'high', 'medium', 'low')
- Time frame: Specify the time frame for the information needed (e.g., 'current', 'past year', 'next month')

Example with additional information:
"What are the current waste management policies in the city? Topic: waste management, Urgency: medium, Time frame: current"
"""
    )


class CustomRetrieverTool(BaseTool):
    name: str = "get_information"
    description: str = """
    Searches and retrieves information from official council documents of the city of Freiburg.

    Always use this tool when asked about specific entities, names,
    places, or facts related to Freiburg's council matters,
    even if you think you might know the answer.

    Before using this tool, inform the user empathetically by partially
    repeating the theme of the query. For example:

    Query: "Current events in Freiburg"
    Reply: "I'd be glad to help you discover what's happening in Freiburg. Let me search the latest council documents for current events."

    Query: "What's the status of bicycle path planning?"
    Reply: "Bicycle infrastructure is an important topic for Freiburg. I'll check the recent council documents for the latest updates on bicycle path planning."

    Apply this empathetic and informative approach across all queries,
    tailoring your response to show understanding of the user's needs
    before retrieving information from the council documents.

    Remember to use only information from official Freiburg council documents,
    and if unsure or needing additional details, always use this integrated lookup tool.
    Never invent or hallucinate information, and use these capabilities solely for
    providing information about Freiburg council matters."""
    args_schema: Type[BaseModel] = RetrieverInput
    return_direct: bool = True

    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        raise NotImplementedError()

    async def _arun(
        self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError()


tools = [CustomRetrieverTool()]
