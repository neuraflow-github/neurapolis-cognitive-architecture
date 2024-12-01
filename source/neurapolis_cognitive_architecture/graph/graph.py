from langgraph.graph import StateGraph
from neurapolis_cognitive_architecture.enums import GraphStep
from neurapolis_cognitive_architecture.models import GraphConfig, State

from .edges import after_decider_to_retriever_or_replier_conditional_edge
from .nodes import DeciderNode, ReplierNode, RetrieverNode

graph_builder = StateGraph(State, GraphConfig)

graph_builder.add_node(GraphStep.DECIDER.value, DeciderNode().decide)
graph_builder.add_node(GraphStep.RETRIEVER.value, RetrieverNode().retrieve)
graph_builder.add_node(GraphStep.REPLIER.value, ReplierNode().reply)

graph_builder.set_entry_point(GraphStep.DECIDER.value)

graph_builder.add_conditional_edges(
    GraphStep.DECIDER.value, after_decider_to_retriever_or_replier_conditional_edge
)
graph_builder.add_edge(GraphStep.RETRIEVER.value, GraphStep.REPLIER.value)

graph_builder.set_finish_point(GraphStep.REPLIER.value)

graph = graph_builder.compile()
