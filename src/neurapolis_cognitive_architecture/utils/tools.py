from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Optional, Type
from langchain_core.callbacks.manager import (
    CallbackManagerForToolRun,
    AsyncCallbackManagerForToolRun,
)
from langchain_community.vectorstores import Neo4jVector
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()


class RetrieverInput(BaseModel):
    full_query: str = Field(
        description="""When using search_docs, create a complete sentence that captures the user's request in the context of a city council's Retrieval Augmented Generation system. Use the same language as the input. Examples:

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
    name: str = "search_docs"
    description: str = (
        """Searches and returns documents from relevant city council sources. This tool is crucial for answering any factual questions about local governance, city policies, public services, or verifying details within the given context of the city council. Always use this tool when asked about specific ordinances, city projects, public meetings, or facts related to local government operations, even if you think you might know the answer."""
    )
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
