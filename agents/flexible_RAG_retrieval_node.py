# agents/flexible_RAG_retrieval_node.py
import logging
from state import AgentGraphState
from graph.utils import query_knowledge_base # Import the new utility

logger = logging.getLogger(__name__)

async def flexible_RAG_retrieval_node(state: AgentGraphState) -> dict:
    """
    Queries a VectorDB based on a dynamic query configuration provided in the graph state.
    This node is a generic, reusable tool.
    """
    logger.info("--- Executing Flexible RAG Retrieval Node ---")
    
    rag_config = state.get("rag_query_config")

    if not rag_config or not isinstance(rag_config, dict):
        logger.warning("RAG Node: `rag_query_config` is missing or invalid in state. Skipping retrieval.")
        return {"rag_retrieved_documents": []}

    query_text = rag_config.get("query_text")
    category = rag_config.get("category")
    top_k = rag_config.get("top_k", 3)

    if not query_text or not category:
        logger.error(f"RAG Node: `query_text` or `category` missing in rag_query_config. Config: {rag_config}")
        return {"rag_retrieved_documents": []}
    
    # Call the utility function to perform the actual search
    retrieved_documents = await query_knowledge_base(
        query_string=query_text,
        category=category,
        top_k=top_k
    )
    
    logger.info(f"RAG Node: Retrieved {len(retrieved_documents)} documents for consumer '{state.get('rag_target_consumer')}'.")
    
    # Return the results to be merged into the graph state
    # We also clear the query config since it has been used.
    return {
        "rag_retrieved_documents": retrieved_documents,
        "rag_query_config": None 
    }