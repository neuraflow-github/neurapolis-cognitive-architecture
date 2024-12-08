from langgraph.graph import StateGraph
from langgraph.prebuilt import tools_condition
from neurapolis_cognitive_architecture.enums import GraphStep
from neurapolis_cognitive_architecture.models import GraphConfig, State

from .nodes import AgentNode, tool_node

graph_builder = StateGraph(State, GraphConfig)

graph_builder.add_node(GraphStep.AGENT.value, AgentNode().agent)
graph_builder.add_node(GraphStep.RETRIEVER.value, tool_node)

graph_builder.set_entry_point(GraphStep.AGENT.value)

graph_builder.add_conditional_edges(
    GraphStep.AGENT.value,
    tools_condition,
    {"tools": GraphStep.RETRIEVER.value, "__end__": "__end__"},
)
graph_builder.add_edge(GraphStep.RETRIEVER.value, GraphStep.AGENT.value)

# graph_builder.set_finish_point(GraphStep.AGENT.value)
