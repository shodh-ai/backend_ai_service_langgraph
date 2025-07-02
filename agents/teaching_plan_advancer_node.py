# agents/teaching_plan_advancer_node.py
import logging
from state import AgentGraphState

logger = logging.getLogger(__name__)

async def teaching_plan_advancer_node(state: AgentGraphState) -> dict:
    """
    A simple utility node that increments the current_plan_step_index in the state.
    This is called after a student confirms understanding of the current step.
    """
    logger.info("--- Executing Teaching Plan Advancer Node ---")
    
    current_index = state.get("current_plan_step_index", 0)
    new_index = current_index + 1
    
    logger.info(f"Advancing lesson plan from step {current_index} to {new_index}.")
    
    # Get the existing plan to ensure it's not lost from the state
    plan = state.get("pedagogical_plan")
    
    return {
        "current_plan_step_index": new_index,
        "pedagogical_plan": plan
    }