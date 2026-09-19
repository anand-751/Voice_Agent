from langgraph.graph import END, StateGraph

from .nodes import make_responder_node, make_router_node, make_tool_node
from .state import AgentState


def route_after_router(state):
	action = state["decision"].get("action", "chitchat")
	return "responder" if action in ("chitchat", "end_call") else "tools"


def build_graph(app):
	graph = StateGraph(AgentState)
	graph.add_node("router", make_router_node(app))
	graph.add_node("tools", make_tool_node(app))
	graph.add_node("responder", make_responder_node(app))

	graph.set_entry_point("router")
	graph.add_conditional_edges("router", route_after_router)
	graph.add_edge("tools", "responder")
	graph.add_edge("responder", END)
	return graph.compile()
