# graph/pedagogy_flow.py
from langgraph.graph import StateGraph, END
from state import AgentGraphState

# 1. Import the agent node functions for the new architecture
from agents import (
    initial_report_generation_node,
)

# 2. Define standardized node names
NODE_INITIAL_REPORT_GENERATION = "initial_report_generation"



def create_pedagogy_subgraph():
    """
    Creates a LangGraph subgraph for the pedagogy flow.
    This flow follows the standard RAG -> Generator -> Formatter architecture
    to determine the next best task for the student.
    """
    workflow = StateGraph(AgentGraphState)

    # 3. Add the nodes to the subgraph
    workflow.add_node(NODE_INITIAL_REPORT_GENERATION, initial_report_generation_node)

    # 4. Define the entry point and the sequential flow
    workflow.set_entry_point(NODE_INITIAL_REPORT_GENERATION)
    workflow.add_edge(NODE_INITIAL_REPORT_GENERATION, END) # End of the subgraph

    # 5. Compile and return the subgraph
    return workflow.compile()
