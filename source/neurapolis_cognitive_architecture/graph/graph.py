from langgraph.graph import StateGraph
from langgraph.prebuilt import tools_condition
from neurapolis_cognitive_architecture.enums.cogitive_architecture_step import (
    CognitiveArchitectureStep,
)
from neurapolis_cognitive_architecture.graph.nodes.agent_node import agent_node
from neurapolis_cognitive_architecture.graph.nodes.tools_node import tools_node
from neurapolis_cognitive_architecture.models.graph_config import GraphConfig
from neurapolis_cognitive_architecture.models.state import State

graph_builder = StateGraph(State, GraphConfig)

graph_builder.add_node(CognitiveArchitectureStep.AGENT.value, agent_node)
graph_builder.add_node(CognitiveArchitectureStep.TOOLS.value, tools_node)

graph_builder.set_entry_point(CognitiveArchitectureStep.AGENT.value)

graph_builder.add_conditional_edges(
    CognitiveArchitectureStep.AGENT.value, tools_condition
)
graph_builder.add_edge(
    CognitiveArchitectureStep.TOOLS.value, CognitiveArchitectureStep.AGENT.value
)

graph_builder.set_finish_point(CognitiveArchitectureStep.AGENT.value)

graph = graph_builder.compile()
