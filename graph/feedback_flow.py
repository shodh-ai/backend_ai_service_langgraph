# graph/feedback_flow.py
from langgraph.graph import StateGraph, END
from state import AgentGraphState

# 1. Import the agent node functions
from agents.feedback_RAG_document_node import feedback_RAG_document_node
from agents.feedback_generator import feedback_generator_node
from agents.feedback_output_formatter import feedback_output_formatter_node

# 2. Define standardized node names
NODE_FEEDBACK_RAG = "feedback_rag"
NODE_FEEDBACK_GENERATOR = "feedback_generator"
NODE_FEEDBACK_OUTPUT_FORMATTER = "feedback_output_formatter"



def create_feedback_subgraph():
    """
    Creates a LangGraph subgraph for the feedback flow.
    This flow follows the standard RAG -> Generator -> Formatter architecture.
    """
    workflow = StateGraph(AgentGraphState)

    # 3. Add the nodes to the subgraph
    workflow.add_node(NODE_FEEDBACK_RAG, feedback_RAG_document_node)
    workflow.add_node(NODE_FEEDBACK_GENERATOR, feedback_generator_node)
    workflow.add_node(NODE_FEEDBACK_OUTPUT_FORMATTER, feedback_output_formatter_node)


    # 4. Define the entry point and the sequential flow
    workflow.set_entry_point(NODE_FEEDBACK_RAG)
    workflow.add_edge(NODE_FEEDBACK_RAG, NODE_FEEDBACK_GENERATOR)
    workflow.add_edge(NODE_FEEDBACK_GENERATOR, NODE_FEEDBACK_OUTPUT_FORMATTER)
    # The flow-specific formatter is the final step in this subgraph
    workflow.add_edge(NODE_FEEDBACK_OUTPUT_FORMATTER, END)

    # 5. Compile and return the subgraph
    return workflow.compile()
