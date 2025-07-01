# graph/teaching_RAG_document_node.py

import logging
from state import AgentGraphState
from graph.utils import query_knowledge_base # We will create a shared utility for the DB query

logger = logging.getLogger(__name__)

async def teaching_RAG_document_node(state: AgentGraphState) -> dict:
    """
    Queries the unified knowledge base to find the most relevant 'teaching'
    documents based on the current student context and learning objective.
    """
    logger.info("---Executing Teaching RAG Node---")

    # --- Dynamically construct the query based on the current lesson step ---
    plan = state.get("pedagogical_plan")
    current_index = state.get("current_plan_step_index", 0)
    current_step_focus = ""

    if plan and isinstance(plan, list) and 0 <= current_index < len(plan):
        current_step = plan[current_index]
        if isinstance(current_step, dict):
            current_step_focus = current_step.get('focus', '')
        logger.info(f"Current lesson step focus: {current_step_focus}")

    # Define which parts of the state form the query for 'teaching'.
    # The query is now more specific if a lesson step is active.
    query_parts = [
        f"Topic/Focus for this step: {current_step_focus}",
        f"Overall Learning Objective: {state.get('Learning_Objective_Focus', '')}",
        f"Student Proficiency: {state.get('STUDENT_PROFICIENCY', '')}",
        f"General concepts to cover: {state.get('LESSON_FOR_STUDENT', '')}",
    ]
    query_string = " \n ".join(filter(None, query_parts)).strip()

    if not query_string:
        logger.warning("Teaching RAG Node: Query string is empty. Skipping vector search.")
        return {"rag_document_data": []}

    # --- Get the category for filtering ---
    # The category for filtering is the specific lesson_id for the current teaching session.
    # This ensures we retrieve the exact teaching strategy document for the current learning objective.
    current_context = state.get("current_context", {})
    category = current_context.get("lesson_id")

    if not category:
        logger.warning("No lesson_id found in current_context. Cannot perform a targeted RAG query.")
        return {"rag_document_data": []}

    logger.info(f"Querying knowledge base with category: {category}")

    # --- Query the Knowledge Base ---
    # The utility function will handle the actual ChromaDB query.
    # We are now passing the determined category for filtering.
    retrieved_documents = await query_knowledge_base(
        query_string=query_string,
        category=category
    )

    return {"rag_document_data": retrieved_documents}