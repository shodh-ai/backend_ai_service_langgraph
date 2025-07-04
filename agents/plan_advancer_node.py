import logging
from typing import Dict

from state import AgentGraphState

logger = logging.getLogger(__name__)

def plan_advancer_node(state: AgentGraphState) -> Dict:
    """
    Advances the pedagogical plan to the next step by incrementing the step index.
    """
    user_id = state["user_id"]
    current_context = state.get("current_context", {})
    
    if not current_context or 'current_plan_step_index' not in current_context:
        logger.warning(f"PlanAdvancerNode: 'current_plan_step_index' not found in current_context for user {user_id}. Cannot advance plan.")
        return {}

    current_index = current_context.get('current_plan_step_index', 0)
    new_index = current_index + 1

    logger.info(f"PlanAdvancerNode: Advancing plan for user {user_id} from step {current_index} to {new_index}.")

    # Update the index in the current_context
    updated_context = current_context.copy()
    updated_context['current_plan_step_index'] = new_index

    return {"current_context": updated_context}
