# langgraph-service/agents/modelling_plan_advancer_node.py
import logging
from state import AgentGraphState

logger = logging.getLogger(__name__)

async def modelling_plan_advancer_node(state: AgentGraphState) -> dict:
    """
    Increments the current step index of the modelling plan.
    """
    logger.info("--- Executing Modelling Plan Advancer Node ---")
    
    current_index = state.get("current_plan_step_index", 0)
    new_index = current_index + 1
    
    logger.info(f"Advancing modelling plan from step {current_index} to {new_index}.")
    
    # Get the existing plan to ensure it's not lost from the state
    plan = state.get("modelling_plan")
    
    # Persist the plan and the new index in the state.
    # This ensures that even if other nodes only return partial state updates,
    # the core plan and our position in it are preserved.
    return {
        "current_plan_step_index": new_index,
        "modelling_plan": plan
    }
