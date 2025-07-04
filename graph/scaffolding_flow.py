# graph/scaffolding_flow.py
from langgraph.graph import StateGraph, END
from state import AgentGraphState

# 1. Import the agent node functions
from agents.scaffolding_RAG_document_node import scaffolding_RAG_document_node
from agents.scaffolding_generator import scaffolding_generator_node
from agents.scaffolding_output_formatter import scaffolding_output_formatter_node

# 2. Define standardized node names
NODE_SCAFFOLDING_RAG = "scaffolding_rag"
NODE_SCAFFOLDING_GENERATOR = "scaffolding_generator"
NODE_SCAFFOLDING_OUTPUT_FORMATTER = "scaffolding_output_formatter"



def create_scaffolding_subgraph():
    """
    Creates a LangGraph subgraph for the scaffolding flow.
    This flow follows the standard RAG -> Generator -> Formatter architecture.
    """
    workflow = StateGraph(AgentGraphState)

    # 3. Add the nodes to the subgraph
    workflow.add_node(NODE_SCAFFOLDING_RAG, scaffolding_RAG_document_node)
    workflow.add_node(NODE_SCAFFOLDING_GENERATOR, scaffolding_generator_node)
    workflow.add_node(NODE_SCAFFOLDING_OUTPUT_FORMATTER, scaffolding_output_formatter_node)


    # 4. Define the entry point and the sequential flow
    workflow.set_entry_point(NODE_SCAFFOLDING_RAG)
    workflow.add_edge(NODE_SCAFFOLDING_RAG, NODE_SCAFFOLDING_GENERATOR)
    workflow.add_edge(NODE_SCAFFOLDING_GENERATOR, NODE_SCAFFOLDING_OUTPUT_FORMATTER)
    # The flow-specific formatter is the final step in this subgraph
    workflow.add_edge(NODE_SCAFFOLDING_OUTPUT_FORMATTER, END)

    # 5. Compile and return the subgraph
    return workflow.compile()
