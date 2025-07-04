import logging
from typing import Dict, Any

from state import AgentGraphState

logger = logging.getLogger(__name__)

async def initiate_module_execution_node(state: AgentGraphState) -> dict:
    """
    Announces the generated lesson plan to the student and prepares to navigate
    to the first step, then ends the turn.
    """
    logger.info("--- Initiating Module Execution ---")
    try:
        plan = state.get("pedagogical_plan")
        lo_title = state.get("current_lo_to_address", {}).get("title", "the topic")

        if not plan or not isinstance(plan, list) or len(plan) == 0:
            logger.warning("Initiate Module Execution: No plan found or plan is empty. Ending turn.")
            return {
                "output_content": {
                    "text_response": "I seem to have had a problem creating the lesson plan. Let's try that again.",
                    "ui_actions": []
                }
            }

        # Announce the plan
        first_step = plan[0]
        focuses = [step.get("focus", f"step {i+1}") for i, step in enumerate(plan)]
        
        # Create a more natural sentence from the list of focuses
        if len(focuses) > 1:
            plan_summary = ", then ".join(focuses[:-1]) + f", and finally {focuses[-1]}"
        else:
            plan_summary = focuses[0]

        announcement = (
            f"Great! To learn about '{lo_title}', our plan is to first {plan_summary}. "
            f"Let's begin with the first step: {first_step.get('focus')}."
        )

        # Prepare navigation to the first module
        target_page = first_step.get("target_page")
        ui_actions = []
        if target_page:
            ui_actions.append({
                "action": "NAVIGATE_TO_PAGE",
                "payload": {"page_id": target_page}
            })
            announcement += f" I'm taking you to the right page now."

        return {
            "output_content": {
                "text_response": announcement,
                "ui_actions": ui_actions
            },
            # This action should be saved
            "last_action_was": "PLAN_ANNOUNCEMENT" 
        }

    except Exception as e:
        logger.error(f"Initiate Module Execution: CRITICAL FAILURE: {e}", exc_info=True)
        return {
            "output_content": {
                "text_response": "I encountered an error while starting our lesson. Please try again.",
                "ui_actions": []
            }
        }
