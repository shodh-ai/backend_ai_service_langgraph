import logging
from typing import Dict, Any
from state import AgentGraphState

logger = logging.getLogger(__name__)

async def context_merger_node(state: AgentGraphState) -> Dict[str, Any]:
    """
    Intelligently merges the 'incoming_context' from the latest request
    into the 'current_context' which is persisted across turns.

    This ensures that long-term state (like a pedagogical plan) is not
    lost, while new information from the current turn is incorporated.
    """
    logger.info("--- Merging Session Context ---")
    
    # The full context from the previous turn (or empty if new session)
    persisted_context = state.get("current_context", {})
    
    # The context from the latest request
    new_context = state.get("incoming_context")

    if new_context:
        logger.debug(f"Merging new context: {new_context}")
        # A simple update is sufficient for a shallow merge.
        # For deep-nested objects, a deep merge might be needed in the future.
        persisted_context.update(new_context)
        logger.debug(f"Resulting context: {persisted_context}")

    return {"current_context": persisted_context}
