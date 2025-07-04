# langgraph-service/agents/pedagogy_output_formatter.py
import logging
from state import AgentGraphState
import copy

logger = logging.getLogger(__name__)

async def pedagogy_output_formatter_node(state: AgentGraphState) -> dict:
    """
    Takes the layered output from the pedagogy generator, formats the interactive
    sequence for the client, and saves the rich content into the state for future use.
    """
    logger.info("---Executing Pedagogy Output Formatter (Layered Content Version)---")


    payload = state.get("intermediate_pedagogy_payload")
    if not payload:
        logger.warning("Pedagogy formatter received no intermediate payload.")

    pedagogy_payload = state.get("task_suggestion_llm_output")

    if not pedagogy_payload or not pedagogy_payload.get("task_suggestion_tts"):
        logger.warning("No valid pedagogy content found in state. Providing a fallback response.")

        return {
            "final_text_for_tts": "I seem to have lost my train of thought. Could we try that again?",
            "final_ui_actions": []
        }

    try:
        # --- 1. Process the interactive sequence for the client ---
        sequence = payload.get("sequence", [])
        final_ui_actions = []
        interruption_context = {}

        tts_parts = [item.get("content", "") for item in sequence if item.get("type") == "tts"]
        final_text_for_tts = " ".join(tts_parts).strip()

        last_listen_step = next((item for item in reversed(sequence) if item.get("type") == "listen"), None)

        if last_listen_step:
            final_ui_actions.append({
                "action_type": "PROMPT_FOR_USER_INPUT",
                "parameters": {"prompt_text": last_listen_step.get("prompt_if_silent", "Listening...")}
            })
            interruption_context = {
                "expected_intent": last_listen_step.get("expected_intent"),
                "source_node": "pedagogy_module"
            }

        # --- 2. Save the rich, non-sequence content for future context ---
        new_context = copy.deepcopy(state.get("current_context", {}))
        
        new_context['last_pedagogy_context'] = {
            "main_explanation": payload.get("main_explanation"),
            "simplified_explanation": payload.get("simplified_explanation"),
            "clarifications": payload.get("clarifications", {})
        }
        
        logger.info(f"Formatted TTS: '{final_text_for_tts[:100]}...' and {len(final_ui_actions)} UI actions.")

        return {
            "final_text_for_tts": final_text_for_tts,
            "final_ui_actions": final_ui_actions,
            "current_context": new_context,
            "interruption_context": interruption_context
        }

    except Exception as e:
        logger.error(f"PedagogyOutputFormatter: CRITICAL FAILURE: {e}", exc_info=True)
        return {
            "final_text_for_tts": "I ran into a technical problem while preparing my response.",
            "final_ui_actions": []
        }
